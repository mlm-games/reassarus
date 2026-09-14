//! Rendering pipeline for processing events into layers

use crate::renderer::RenderContext;
use crate::utils::{DirtyRegion, RenderError};
#[cfg(feature = "analysis-integration")]
use reassarus_core::analysis::ScriptAnalysis;
use reassarus_core::parser::{Event, Script};
use smallvec::SmallVec;

#[cfg(feature = "nostd")]
use alloc::{string::String, vec::Vec};
#[cfg(not(feature = "nostd"))]
use std::{string::String, vec::Vec};

pub mod animation;
pub mod compositing;
#[cfg(feature = "vector")]
pub mod drawing;
pub mod effects;
#[cfg(feature = "shaping")]
pub mod font_loader;
#[cfg(feature = "shaping")]
pub mod shaping;
pub mod tag_processor;
pub mod text_segmenter;
pub mod transform;
pub mod validation;

#[cfg(feature = "shaping")]
mod software_pipeline;
#[cfg(feature = "shaping")]
pub use software_pipeline::SoftwarePipeline;

/// Pipeline trait for processing events
pub trait Pipeline: Send + Sync {
    /// Prepare the pipeline with a script
    fn prepare_script(
        &mut self,
        script: &Script,
        #[cfg(feature = "analysis-integration")] analysis: Option<&ScriptAnalysis>,
        #[cfg(not(feature = "analysis-integration"))] _analysis: Option<()>,
    ) -> Result<(), RenderError>;

    /// Get the current script
    fn script(&self) -> Option<&Script<'_>>;

    /// Process events into intermediate layers
    fn process_events(
        &mut self,
        events: &[&Event],
        time_cs: u32,
        context: &RenderContext,
    ) -> Result<Vec<IntermediateLayer>, RenderError>;

    /// Process events at a millisecond timestamp (native renderer clock).
    ///
    /// Millisecond entry interpolates `\move`/`\t`/`\fad`/`\fade`/karaoke
    /// smoothly; the centisecond entry quantizes to 10ms steps. The default
    /// impl quantizes (`ms / 10`); override for native precision.
    fn process_events_ms(
        &mut self,
        events: &[&Event],
        time_ms: u64,
        context: &RenderContext,
    ) -> Result<Vec<IntermediateLayer>, RenderError> {
        self.process_events(
            events,
            (time_ms / 10).min(u64::from(u32::MAX)) as u32,
            context,
        )
    }

    /// Compute dirty regions for incremental rendering
    fn compute_dirty_regions(
        &self,
        events: &[&Event],
        time_cs: u32,
        prev_time_cs: u32,
    ) -> Result<Vec<DirtyRegion>, RenderError>;

    /// Millisecond variant of [`compute_dirty_regions`](Pipeline::compute_dirty_regions).
    fn compute_dirty_regions_ms(
        &self,
        events: &[&Event],
        time_ms: u64,
        prev_time_ms: u64,
    ) -> Result<Vec<DirtyRegion>, RenderError> {
        self.compute_dirty_regions(
            events,
            (time_ms / 10).min(u64::from(u32::MAX)) as u32,
            (prev_time_ms / 10).min(u64::from(u32::MAX)) as u32,
        )
    }
}

/// Pipeline stage for processing
#[derive(Debug, Clone, Copy)]
pub enum PipelineStage {
    /// Text shaping stage
    Shaping,
    /// Drawing command processing
    Drawing,
    /// Effect application
    Effects,
    /// Layer compositing
    Compositing,
}

/// Intermediate layer representation
pub enum IntermediateLayer {
    /// Rasterized bitmap layer
    Raster(RasterData),
    /// Vector graphics layer
    Vector(VectorData),
    /// Text layer (boxed: `TextData` is ~296B vs ~80B for the other
    /// variants, so storing it inline bloats every layer value)
    Text(Box<TextData>),
}

impl IntermediateLayer {
    /// Check if layer intersects with a dirty region
    pub fn intersects_region(&self, region: &DirtyRegion) -> bool {
        match self {
            Self::Raster(data) => {
                let layer_bounds = (data.x, data.y, data.x + data.width, data.y + data.height);
                region.intersects(layer_bounds)
            }
            Self::Vector(data) => {
                if let Some(bounds) = &data.bounds {
                    region.intersects(*bounds)
                } else {
                    true
                }
            }
            Self::Text(data) => {
                let (w, h) = match &data.measured {
                    Some(m) => (
                        m.spaced_width(data.spacing, data.text.chars().count()),
                        m.height,
                    ),
                    // Hand-built layer without shaping metrics: fall back to a
                    // generous estimate so the region is never under-covered.
                    None => (200.0, data.font_size * 1.5),
                };
                let approx_bounds = (
                    data.x as u32,
                    data.y as u32,
                    (data.x + w) as u32,
                    (data.y + h) as u32,
                );
                region.intersects(approx_bounds)
            }
        }
    }
}

/// Raster layer data
pub struct RasterData {
    /// Pixel data (RGBA)
    pub pixels: Vec<u8>,
    /// Layer X position
    pub x: u32,
    /// Layer Y position
    pub y: u32,
    /// Layer width
    pub width: u32,
    /// Layer height
    pub height: u32,
    /// Opacity (0-255)
    pub opacity: u8,
}

/// Vector graphics layer data
pub struct VectorData {
    /// Path to draw (lyon path IR; only present with `vector`)
    #[cfg(feature = "vector")]
    pub path: Option<lyon_path::Path>,
    /// Fill color (RGBA)
    pub color: [u8; 4],
    /// Stroke information
    pub stroke: Option<StrokeInfo>,
    /// Bounding box
    pub bounds: Option<(u32, u32, u32, u32)>,
}

/// Stroke information for vector graphics
pub struct StrokeInfo {
    /// Stroke color (RGBA)
    pub color: [u8; 4],
    /// Stroke width
    pub width: f32,
}

/// Text layer data
pub struct TextData {
    /// Text content
    pub text: String,
    /// Font family
    pub font_family: String,
    /// Font size in pixels
    pub font_size: f32,
    /// Text color (RGBA)
    pub color: [u8; 4],
    /// X position
    pub x: f32,
    /// Y position
    pub y: f32,
    /// Text effects
    pub effects: SmallVec<[TextEffect; 4]>,
    /// Letter spacing in pixels
    pub spacing: f32,
    /// Shaping-measured bounds, populated by the pipeline from the same
    /// cached run it lays out with. Backends must prefer this over
    /// glyph-count estimates for clip/blur/layer rects and dirty regions;
    /// `None` only for hand-built layers (tests, drawings path).
    pub measured: Option<MeasuredBounds>,
}

/// Shaping-measured text bounds in the layer's local pixels.
#[derive(Clone, Copy, Debug)]
pub struct MeasuredBounds {
    /// Total shaped advance (excludes [`TextData::spacing`], which backends
    /// add per glyph: rendered width is `width + spacing * (glyphs - 1)`).
    pub width: f32,
    /// Ascent+descent box (Windows metrics, libass-compatible).
    pub height: f32,
    /// Baseline offset from the layer top.
    pub baseline: f32,
    /// Ascent in pixels.
    pub ascent: f32,
    /// Descent in pixels (negative).
    pub descent: f32,
}

impl MeasuredBounds {
    /// Rendered width including inter-glyph spacing (`glyphs` = glyph count).
    #[must_use]
    pub fn spaced_width(&self, spacing: f32, glyphs: usize) -> f32 {
        self.width + spacing * glyphs.saturating_sub(1) as f32
    }
}

/// Text effect enumeration
#[derive(Clone, Debug)]
pub enum TextEffect {
    /// Bold text
    Bold,
    /// Italic text
    Italic,
    /// Underline
    Underline,
    /// Strikethrough
    Strikethrough,
    /// Outline with color and per-axis widths (`\bordx`/`\bordy`).
    ///
    /// Both axes are preserved from evaluation through the IR. Rasterizers
    /// approximate with `max(width_x, width_y)` until anisotropic outlines
    /// are supported; do not collapse the axes at emission.
    Outline {
        color: [u8; 4],
        width_x: f32,
        width_y: f32,
    },
    /// Shadow with color and offset
    Shadow {
        color: [u8; 4],
        x_offset: f32,
        y_offset: f32,
    },
    /// Blur effect
    Blur { radius: f32 },
    /// Edge blur effect (only blurs outline/edges)
    EdgeBlur { radius: f32 },
    /// Karaoke effect. `secondary` is the not-yet-sung (secondary) colour;
    /// the sung colour is the layer's primary `color`.
    Karaoke {
        progress: f32,
        style: u8,
        secondary: [u8; 4],
    },
    /// 3D rotation (in degrees). `origin`, when set, is the rotation centre in
    /// screen-space pixels (`\org`); otherwise the text's own centre is used.
    Rotation {
        x: f32,
        y: f32,
        z: f32,
        origin: Option<(f32, f32)>,
    },
    /// Shear/skew transformation
    Shear { x: f32, y: f32 },
    /// Scale transformation
    Scale { x: f32, y: f32 },
    /// Clip region
    Clip {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        inverse: bool,
    },
    /// Vector (drawing) clip region (`\clip(m ...)`), in render coordinates.
    /// Backends tessellate/mask it directly; `inverse` is `\iclip`.
    /// Only present with `vector` (lyon path IR).
    #[cfg(feature = "vector")]
    VectorClip {
        path: lyon_path::Path,
        inverse: bool,
    },
    /// Opaque box behind the text (`BorderStyle: 3`), drawn in the outline
    /// colour with `padding` pixels around the glyph bounds.
    OpaqueBox { color: [u8; 4], padding: f32 },
}
