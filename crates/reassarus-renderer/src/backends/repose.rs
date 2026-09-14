//! GPU backend emitting a Repose [`Scene`](repose_core::Scene).
//!
//! The `tiny-skia` software backend stays the parity reference: it owns
//! layout, shaping and effect evaluation. This backend only converts the
//! resulting [`IntermediateLayer`](crate::pipeline::IntermediateLayer)s into
//! Repose scene nodes, so ASS playback composes with the rest of a Repose UI
//! without any ASS-specific code inside the framework.
//!
//! [`layers_to_scene`] is pure CPU and needs no GPU, which is what the parity
//! tests exercise. [`ReposeBackend::composite_layers`] additionally resolves
//! the scene to RGBA through `OffscreenRenderer`, so it needs a WGPU adapter
//! at render time (llvmpipe via `mesa-vulkan-drivers` is enough for CI) and
//! fails loudly without one — select `Software` explicitly for
//! headless-without-GPU environments.
//!
//! Mapping (v1):
//!
//! - `Text` fill → `SceneNode::Text`; `Bold`/`Italic`/underline/strike ride
//!   on the node itself.
//! - `Outline` → overlaid stroke `Text` (under) + fill `Text` (over), since
//!   `DrawStyle` is fill-XOR-stroke per node.
//! - `Shadow` → offset duplicate `Text` behind the main one.
//! - `Blur` → `BeginLayer`/`EndLayer` with a blur radius around the whole
//!   run (fill, outline, shadow). `EdgeBlur` (`\be`) wraps just the outline
//!   stroke so the fill stays sharp; the two are never merged.
//! - `Karaoke` style 0 (`\k`) flips the whole run between sung/unsung;
//!   styles 1-3 sweep via a clipped sung window, like the software
//!   reference.
//! - `Rotation` (`\frz`) → `PushTransform`; `Scale` (ASS percentages) is
//!   normalized to multipliers there — Y is already baked into the font
//!   size during shaping, so only X is applied. `Shear` (`\fax`/`\fay`)
//!   maps directly onto the transform's shear factors. `Rotation.x/y`
//!   (`\frx`/`\fry`) take the perspective path instead: libass's exact 3D
//!   rotation (order, signs, shear coupling) about the `\org` pivot
//!   (default: run centre) with its perspective row, so the subtree
//!   flattens and composites projectively (differentially fit against
//!   libass 0.17.5, ≤2px).
//! - `Clip` → `PushClip`/`PopClip` (`Intersect`/`Difference` for
//!   `\clip`/`\iclip`).
//! - `Vector` → tessellated `VectorMesh`es: always a fill pass, plus a
//!   separate stroke pass in `stroke.color` when a border is set.
//! - `Raster` → uploaded with `register_image_rgba8` and emitted as an
//!   `Image` node by [`ReposeBackend::composite_layers`]. The pure
//!   [`layers_to_scene`] (used by parity tests, no GPU) still counts these
//!   in [`BuiltScene::skipped_raster`].
//! - `OpaqueBox` → backing `Rect` node.

use core::f32::consts::PI;
use std::sync::Arc;

use repose_core::{
    BlendMode, Brush, ClipOp, Color, DrawStyle, FontStyle, FontWeight, ImageFit, PaintDesc, Px,
    Rect, Scene, SceneNode, StrokeCap, StrokeJoin, TextAlign, TextDecoration, TextExtraStyle,
    Transform, VectorMeshData, VectorVertex,
};

use super::{BackendFeature, BackendType, RenderBackend};
use crate::pipeline::{
    IntermediateLayer, Pipeline, RasterData, SoftwarePipeline, TextData, TextEffect, VectorData,
};
use crate::renderer::RenderContext;
use crate::utils::{DirtyRegion, RenderError};

/// GPU renderer for [`IntermediateLayer`]s via a Repose [`Scene`].
///
/// Construction is cheap and needs no GPU; the WGPU adapter is acquired
/// lazily on the first [`composite_layers`](RenderBackend::composite_layers)
/// call (and re-sized from the [`RenderContext`]) so merely selecting the
/// backend never fails headless.
pub struct ReposeBackend {
    width: u32,
    height: u32,
    offscreen: Option<repose_render_wgpu::offscreen::OffscreenRenderer>,
}

impl ReposeBackend {
    /// Create a backend for frames of the context's size.
    #[must_use]
    pub fn new(context: &RenderContext) -> Self {
        Self {
            width: context.width().max(1),
            height: context.height().max(1),
            offscreen: None,
        }
    }

    /// Build the Repose scene for `layers` without needing a GPU.
    #[must_use]
    pub fn build_scene(layers: &[IntermediateLayer], width: u32, height: u32) -> BuiltScene {
        layers_to_scene(layers, width, height)
    }
}

impl RenderBackend for ReposeBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Repose
    }

    fn create_pipeline(&self) -> Result<Box<dyn Pipeline>, RenderError> {
        Ok(Box::new(SoftwarePipeline::new()))
    }

    fn composite_layers(
        &mut self,
        layers: &[IntermediateLayer],
        context: &RenderContext,
    ) -> Result<Vec<u8>, RenderError> {
        let width = context.width().max(1);
        let height = context.height().max(1);
        // The scene needs the offscreen renderer for raster uploads, so make
        // sure it exists (and matches the frame size) before building it.
        if self.offscreen.is_none() {
            let renderer =
                repose_render_wgpu::offscreen::OffscreenRenderer::new_blocking(width, height, 4)
                    .map_err(|e| {
                        RenderError::BackendError(format!(
                            "repose backend needs a WGPU adapter (mesa-vulkan-drivers suffices for headless/CI); render failed: {e:#}"
                        ))
                    })?;
            self.offscreen = Some(renderer);
            self.width = width;
            self.height = height;
        }
        let offscreen = self.offscreen.as_mut().expect("initialised above");
        if width != self.width || height != self.height {
            offscreen.ensure_size(width, height).map_err(|e| {
                RenderError::BackendError(format!("repose offscreen resize failed: {e:#}"))
            })?;
            self.width = width;
            self.height = height;
        }
        let built = {
            let renderer = offscreen.renderer_mut();
            build_scene(layers, width, height, &mut |data: &RasterData| {
                upload_raster(renderer, data)
            })
        };
        // A failed vector tessellation drops a drawing entirely: surface it
        // instead of rendering a silently incomplete frame. (Raster uploads
        // resolve through the live renderer here, so `skipped_raster` stays 0
        // on this path; only the GPU-less `layers_to_scene` counts those.)
        if built.skipped_tess > 0 {
            return Err(RenderError::BackendError(format!(
                "repose backend dropped {} vector layer(s): tessellation failed",
                built.skipped_tess
            )));
        }
        let scene = built.scene;
        offscreen.render_rgba(&scene, None).map_err(|e| {
            RenderError::BackendError(format!("repose offscreen render failed: {e:#}"))
        })
    }

    fn composite_layers_incremental(
        &mut self,
        layers: &[IntermediateLayer],
        dirty_regions: &[DirtyRegion],
        previous_frame: &[u8],
        context: &RenderContext,
    ) -> Result<Vec<u8>, RenderError> {
        let _ = (dirty_regions, previous_frame);
        self.composite_layers(layers, context)
    }

    fn supports_feature(&self, feature: BackendFeature) -> bool {
        matches!(feature, BackendFeature::HardwareAcceleration)
    }
}

/// Output of [`layers_to_scene`] with loss accounting.
///
/// Anything the scene graph cannot express without a live renderer (raster
/// uploads, failed tessellation) is counted here so parity tests fail loudly
/// instead of comparing a silently degraded scene.
#[derive(Debug)]
pub struct BuiltScene {
    /// The composed scene, ready for `render_scene_to_encoder` or offscreen
    /// readback via `OffscreenRenderer`.
    pub scene: Scene,
    /// `Raster` layers skipped (need live renderer image handles; only
    /// non-zero for the GPU-less [`layers_to_scene`], never for
    /// [`RenderBackend::composite_layers`]).
    pub skipped_raster: usize,
    /// Vector layers whose tessellation failed.
    pub skipped_tess: usize,
}

/// Convert pipeline layers into a Repose [`Scene`].
///
/// `width`/`height` are the frame size in physical pixels; the scene clears
/// to transparent so subtitles composite over video.
///
/// Pure CPU: needs no GPU, which is what the parity tests exercise. Raster
/// layers are counted in [`BuiltScene::skipped_raster`] — use
/// [`RenderBackend::composite_layers`] for the upload path.
#[must_use]
pub fn layers_to_scene(layers: &[IntermediateLayer], width: u32, height: u32) -> BuiltScene {
    build_scene(layers, width, height, &mut |_| None)
}

/// [`layers_to_scene`] with a raster-image upload hook.
///
/// Called in layer order so z-order is preserved. Returning `None` counts
/// the layer in [`BuiltScene::skipped_raster`].
fn build_scene(
    layers: &[IntermediateLayer],
    width: u32,
    height: u32,
    upload_raster: &mut dyn FnMut(&RasterData) -> Option<SceneNode>,
) -> BuiltScene {
    let _ = (width, height);
    let mut out = BuiltScene {
        scene: Scene {
            clear_color: Color::from_rgba(0, 0, 0, 0),
            nodes: Vec::new(),
        },
        skipped_raster: 0,
        skipped_tess: 0,
    };
    let mut layers_ctx = LayerCtx { next_layer_id: 1 };
    for layer in layers {
        match layer {
            IntermediateLayer::Raster(data) => match upload_raster(data) {
                Some(node) => out.scene.nodes.push(node),
                None => out.skipped_raster += 1,
            },
            IntermediateLayer::Vector(data) => {
                if !emit_vector(&mut out, data) {
                    out.skipped_tess += 1;
                }
            }
            IntermediateLayer::Text(data) => emit_text(&mut out, &mut layers_ctx, data),
        }
    }
    out
}

/// Upload a raster layer to the GPU and wrap it in an `Image` node.
///
/// Returns `None` for empty layers. The software reference draws raster
/// layers verbatim (ignoring `opacity`), so the tint stays opaque white.
fn upload_raster(
    renderer: &mut repose_render_wgpu::WgpuSceneRenderer,
    data: &RasterData,
) -> Option<SceneNode> {
    if data.width == 0 || data.height == 0 || data.pixels.is_empty() {
        return None;
    }
    let handle = renderer.register_image_rgba8(data.width, data.height, &data.pixels, true);
    Some(SceneNode::Image {
        rect: Rect {
            x: data.x as f32,
            y: data.y as f32,
            w: data.width as f32,
            h: data.height as f32,
        },
        handle,
        tint: Color::from_rgba(255, 255, 255, 255),
        fit: ImageFit::FillBounds,
    })
}

/// Per-scene counter for graphics-layer ids.
struct LayerCtx {
    next_layer_id: u32,
}

impl LayerCtx {
    fn alloc(&mut self) -> u32 {
        let id = self.next_layer_id;
        self.next_layer_id = self.next_layer_id.wrapping_add(1).max(1);
        id
    }
}

/// Text bounds from the pipeline's shaping-measured metrics, falling back to
/// the old glyph-count estimate for hand-built layers (tests) only.
fn text_rect(data: &TextData) -> Rect {
    let (w, h) = match &data.measured {
        Some(m) => (
            m.spaced_width(data.spacing, data.text.chars().count()),
            m.height,
        ),
        None => {
            let glyphs = data.text.chars().count().max(1) as f32;
            (
                (glyphs * data.font_size * 0.6 + data.spacing * glyphs).max(1.0),
                (data.font_size * 1.4).max(1.0),
            )
        }
    };
    Rect {
        x: data.x,
        y: data.y,
        w: w.max(1.0),
        h: h.max(1.0),
    }
}

/// Expand a layer rect to cover `rect` under `transform` (plus `pad` px on
/// every side for outline/blur bleed), so rotated/projected layer content
/// never clips at the flat text rect. Uses the same origin-free affine +
/// perspective-divide semantics the renderer applies; non-finite projections
/// fall back to the padded flat rect.
fn expand_layer_rect(rect: Rect, transform: Option<Transform>, pad: f32) -> Rect {
    let padded = Rect {
        x: rect.x - pad,
        y: rect.y - pad,
        w: rect.w + pad * 2.0,
        h: rect.h + pad * 2.0,
    };
    let Some(t) = transform else {
        return padded;
    };
    let corners = [
        (rect.x, rect.y),
        (rect.x + rect.w, rect.y),
        (rect.x + rect.w, rect.y + rect.h),
        (rect.x, rect.y + rect.h),
    ];
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for (x, y) in corners {
        let m = t.projective_matrix();
        let w = m[6] * x + m[7] * y + m[8];
        if !w.is_finite() || w.abs() < 1e-6 {
            return padded;
        }
        let px = (m[0] * x + m[1] * y + m[2]) / w;
        let py = (m[3] * x + m[4] * y + m[5]) / w;
        if !px.is_finite() || !py.is_finite() {
            return padded;
        }
        min_x = min_x.min(px);
        min_y = min_y.min(py);
        max_x = max_x.max(px);
        max_y = max_y.max(py);
    }
    Rect {
        x: min_x - pad,
        y: min_y - pad,
        w: (max_x - min_x).max(1.0) + pad * 2.0,
        h: (max_y - min_y).max(1.0) + pad * 2.0,
    }
}

/// Bundled parameters for one `SceneNode::Text` emission.
#[derive(Clone, Copy)]
struct TextPass<'a> {
    text: &'a str,
    rect: Rect,
    color: [u8; 4],
    font_family: &'a str,
    font_size: f32,
    spacing: f32,
    weight: FontWeight,
    style: FontStyle,
    decoration: TextDecoration,
}

/// Process-lifetime interner for font family names.
///
/// `SceneNode::Text::font_family` is `Option<&'static str>` while the
/// pipeline hands us an owned `String` per layer. Family names are small and
/// few (one per script style), so interning each distinct name once is
/// bounded. Repose resolves families against system fonts best-effort, so
/// passing the resolved name through (instead of `None`) is what selects
/// the right face; an empty name keeps the framework default.
fn intern_font_family(name: &str) -> Option<&'static str> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    if name.is_empty() {
        return None;
    }
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("font family interner lock");
    if let Some(hit) = guard.get(name) {
        return Some(*hit);
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    guard.insert(name.to_owned(), leaked);
    Some(leaked)
}

/// Outline colour plus width in pixels.
type OutlineSpec = ([u8; 4], f32);

/// `\frx`/`\fry`/`\frz` degrees plus `\org` rotation centre in screen pixels.
type RotationSpec = (f32, f32, f32, Option<(f32, f32)>);

fn text_node(pass: &TextPass<'_>, draw_style: DrawStyle) -> SceneNode {
    SceneNode::Text {
        rect: pass.rect,
        text: Arc::from(pass.text),
        color: Color::from_rgba(pass.color[0], pass.color[1], pass.color[2], pass.color[3]),
        size: Px(pass.font_size.max(1.0)),
        font_family: intern_font_family(pass.font_family),
        text_align: TextAlign::Unspecified,
        font_weight: pass.weight,
        font_style: pass.style,
        text_decoration: pass.decoration,
        letter_spacing: Px(pass.spacing),
        line_height: Px((pass.font_size * 1.2).max(1.0)),
        extra_style: TextExtraStyle {
            draw_style,
            ..TextExtraStyle::default()
        },
        url: None,
        font_variation_settings: None,
    }
}

/// Emit the stroke-under-fill outline for `pass`, if any.
///
/// `\be` (edge blur) wraps just this stroke in its own blur layer so the
/// fill stays sharp, mirroring the software reference where edge blur
/// applies to the outline only. Full `\blur` is handled by the caller via
/// the whole-run layer and must not be passed here.
fn emit_outline_pass(
    nodes: &mut Vec<SceneNode>,
    layers_ctx: &mut LayerCtx,
    pass: &TextPass<'_>,
    outline: Option<OutlineSpec>,
    edge_blur: Option<f32>,
    rot: Option<Transform>,
) {
    let Some((outline_color, outline_width)) = outline else {
        return;
    };
    let em = (outline_width / pass.font_size.max(1.0)).clamp(0.01, 0.5);
    let stroke = text_node(
        &TextPass {
            color: outline_color,
            decoration: TextDecoration::default(),
            ..*pass
        },
        DrawStyle::Stroke {
            width: em,
            cap: StrokeCap::Round,
            join: StrokeJoin::Round,
            miter: 4.0,
            path_effect: None,
        },
    );
    match edge_blur.filter(|r| *r > 0.0) {
        Some(radius) => {
            let id = layers_ctx.alloc();
            // Cover the transformed stroke plus blur bleed (see
            // `expand_layer_rect`); the shift keeps children layer-local.
            let layer_rect = expand_layer_rect(pass.rect, rot, radius + outline_width);
            nodes.push(SceneNode::BeginLayer {
                rect: layer_rect,
                layer_id: id,
                alpha: 1.0,
                blur_radius_x: Px(radius),
                blur_radius_y: Px(radius),
                rectangle_edge: true,
            });
            nodes.push(SceneNode::PushTransform {
                transform: Transform::translate(-layer_rect.x, -layer_rect.y),
            });
            nodes.push(stroke);
            nodes.push(SceneNode::PopTransform);
            nodes.push(SceneNode::EndLayer { layer_id: id });
        }
        None => nodes.push(stroke),
    }
}

/// libass perspective focal length in screen pixels.
///
/// libass rotates with `dist = 20000 * blur_scale` in outline units; those
/// are 26.6 fixed-point (64 units/px, size-independent), so the effective
/// focal is `20000 / 64 = 312.5`px. Validated differentially against libass
/// 0.17.5 (single glyphs at \frx±30/\frx60/\fry±20, `\frx45` ± `\org`,
/// 36–90px, 640x360 and 1280x720): this value minimizes squared edge error
/// (max residual ~1px, vs ~80px for the old shear hack). The fit is
/// size- and resolution-independent, matching the outline-units origin.
const LIBASS_FOCAL_PX: f32 = 312.5;

/// Build the libass 3D rotation as explicit homogeneous rows (pivot-relative).
///
/// Ports `calc_transform_matrix` (libass `ass_render.c`): shear rows, then
/// `\frz`, then `\frx`, then `\fry`, with libass's exact signs
/// (`sx=-sin(frx)`, `sy=+sin(fry)`, `sz=-sin(frz)`). ASS `\fscx` is folded
/// on the right (applied first, matching the affine path's scale order);
/// `\fax`/`\fay` ride the pre-rotation shear rows exactly like libass.
/// Rotation is about `(px, py)` (the `\org` pivot, or the run centre);
/// rows are constructed pivot-preserving (`M(pivot) = pivot`, `W = 1`).
/// Returns `(row_x, row_y, persp_row)` for
/// [`from_projective_rows`](repose_core::Transform::from_projective_rows).
fn libass_rotation_rows(
    frx_deg: f32,
    fry_deg: f32,
    frz_deg: f32,
    fax: f32,
    fay: f32,
    fscx: f32,
    px: f32,
    py: f32,
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (frx, fry, frz) = (
        frx_deg * PI / 180.0,
        fry_deg * PI / 180.0,
        frz_deg * PI / 180.0,
    );
    let (sx, cx) = (-frx.sin(), frx.cos());
    let (sy, cy) = (fry.sin(), fry.cos());
    let (sz, cz) = (-frz.sin(), frz.cos());
    // Shear rows (libass `x1`/`y1`, translation parts zero: pivot-relative).
    let x1 = (1.0f32, fax);
    let y1 = (fay, 1.0f32);
    // `\frz`.
    let x2 = (x1.0 * cz - y1.0 * sz, x1.1 * cz - y1.1 * sz);
    let y2 = (x1.0 * sz + y1.0 * cz, x1.1 * sz + y1.1 * cz);
    // `\frx`.
    let y3 = (y2.0 * cx, y2.1 * cx);
    let z3 = (y2.0 * sx, y2.1 * sx);
    // `\fry`.
    let x4 = (x2.0 * cy - z3.0 * sy, x2.1 * cy - z3.1 * sy);
    let z4 = (x2.0 * sy + z3.0 * cy, x2.1 * sy + z3.1 * cy);
    // ASS `\fscx` applies first (rightmost), like the affine path.
    let x4 = (x4.0 * fscx, x4.1 * fscx);
    // libass `offs` coupling: its rows are `row·dist + z·offs`, i.e. the
    // affine part carries `pivot·z / dist`. Without it the map under-shoots
    // libass by ~14px on rotated runs (the fitted form demands it; residual
    // ≤1px globally with it). `offs` is per-glyph in libass; the run pivot
    // is the run-level equivalent (exact for single-run lines). NOTE units:
    // the coupling uses the *scaled* z direction (over the focal), matching
    // the `W` row — raw z here blows the affine part up ~300x.
    let zx = z4.0 / LIBASS_FOCAL_PX;
    let zy = z4.1 / LIBASS_FOCAL_PX;
    let xc = (x4.0 + px * zx, x4.1 + px * zy);
    let yc = (y3.0 + py * zx, y3.1 + py * zy);
    let row_pivot = |p: f32, a: f32, b: f32| [a, b, p - a * px - b * py];
    (
        row_pivot(px, xc.0, xc.1),
        row_pivot(py, yc.0, yc.1),
        [zx, zy, 1.0 - (zx * px + zy * py)],
    )
}

/// Emit one fill pass of `pass`, honouring the outline effect as a
/// stroke-under-fill double emit.
fn emit_fill_pass(
    nodes: &mut Vec<SceneNode>,
    layers_ctx: &mut LayerCtx,
    pass: &TextPass<'_>,
    outline: Option<OutlineSpec>,
    edge_blur: Option<f32>,
    rot: Option<Transform>,
) {
    emit_outline_pass(nodes, layers_ctx, pass, outline, edge_blur, rot);
    nodes.push(text_node(pass, DrawStyle::Fill));
}

/// Emit a `Text` layer with all of its effects.
fn emit_text(out: &mut BuiltScene, layers_ctx: &mut LayerCtx, data: &TextData) {
    let mut weight = FontWeight::NORMAL;
    let mut style = FontStyle::Normal;
    let mut decoration = TextDecoration::default();
    let mut outline: Option<OutlineSpec> = None;
    let mut shadow: Option<([u8; 4], f32, f32)> = None;
    // Full-run `\blur`: wraps fill, outline and shadow in one blur layer,
    // like the software reference's blur temp.
    let mut blur: Option<f32> = None;
    // Edge-only `\be`: blurs just the outline stroke (see
    // [`emit_outline_pass`]), never merged into `blur`.
    let mut edge_blur: Option<f32> = None;
    // `(progress, karaoke style 0-3, unsung colour)`. Style 0 (`\k`) flips
    // the whole run; styles 1-3 sweep, like the software reference.
    let mut karaoke: Option<(f32, u8, [u8; 4])> = None;
    let mut rotation: Option<RotationSpec> = None;
    let mut scale: Option<(f32, f32)> = None;
    // Accumulated `\fax`/`\fay` shear. (`\frx`/`\fry` used to fold in here
    // as a sin-based skew; they now take the perspective path above, with
    // shear folded into the rotation rows libass-style instead.)
    let mut shear: Option<(f32, f32)> = None;
    let mut clip: Option<(f32, f32, f32, f32, bool)> = None;
    // Tessellated drawing clip (`\clip(m ...)`), emitted as a stencil
    // `PushVectorClip` around the run.
    let mut vclip: Option<(lyon_path::Path, bool)> = None;
    let mut opaque: Option<([u8; 4], f32)> = None;

    for effect in data.effects.iter() {
        match effect {
            TextEffect::Bold => weight = FontWeight::BOLD,
            TextEffect::Italic => style = FontStyle::Italic,
            TextEffect::Underline => decoration.underline = true,
            TextEffect::Strikethrough => decoration.strikethrough = true,
            TextEffect::Outline {
                color,
                width_x,
                width_y,
            } => {
                // Uniform stroke until anisotropic outlines land; both axes
                // stay preserved in the IR (see `TextEffect::Outline`).
                outline = Some((*color, width_x.max(*width_y)));
            }
            TextEffect::Shadow {
                color,
                x_offset,
                y_offset,
            } => shadow = Some((*color, *x_offset, *y_offset)),
            TextEffect::Blur { radius } => blur = Some(blur.unwrap_or(0.0).max(*radius)),
            TextEffect::EdgeBlur { radius } => {
                edge_blur = Some(edge_blur.unwrap_or(0.0).max(*radius));
            }
            TextEffect::Karaoke {
                progress,
                style,
                secondary,
            } => karaoke = Some((progress.clamp(0.0, 1.0), *style, *secondary)),
            TextEffect::Rotation { x, y, z, origin } => {
                rotation = Some((*x, *y, *z, *origin));
            }
            TextEffect::Shear { x, y } => {
                let s = shear.get_or_insert((0.0, 0.0));
                s.0 += *x;
                s.1 += *y;
            }
            // ASS scale tags are percentages (`\fscx150` → 150.0) while
            // Repose takes multipliers. Y is already baked into `font_size`
            // during shaping, so only X is applied here.
            TextEffect::Scale { x, .. } => scale = Some((*x / 100.0, 1.0)),
            TextEffect::Clip {
                x1,
                y1,
                x2,
                y2,
                inverse,
            } => clip = Some((*x1, *y1, *x2, *y2, *inverse)),
            TextEffect::VectorClip { path, inverse } => {
                vclip = Some((path.clone(), *inverse));
            }
            TextEffect::OpaqueBox { color, padding } => opaque = Some((*color, *padding)),
        }
    }

    let rect = text_rect(data);
    let base = TextPass {
        text: &data.text,
        rect,
        color: data.color,
        font_family: &data.font_family,
        font_size: data.font_size,
        spacing: data.spacing,
        weight,
        style,
        decoration,
    };
    let nodes = &mut out.scene.nodes;

    if let Some((x1, y1, x2, y2, inverse)) = clip {
        nodes.push(SceneNode::PushClip {
            rect: Rect {
                x: x1,
                y: y1,
                w: (x2 - x1).max(0.0),
                h: (y2 - y1).max(0.0),
            },
            radius: [Px::ZERO; 4],
            op: if inverse {
                ClipOp::Difference
            } else {
                ClipOp::Intersect
            },
        });
    }

    // Drawing clip (`\clip(m ...)`): tessellate to a stencil mask. A failed
    // tessellation drops the clip (fail-open) rather than the run — a missing
    // clip is closer to libass than missing text.
    if let Some((path, inverse)) = &vclip {
        if let Some(mesh) = tessellate_fill_mesh(path, [1.0, 1.0, 1.0, 1.0]) {
            nodes.push(SceneNode::PushVectorClip {
                mesh,
                op: if *inverse {
                    ClipOp::Difference
                } else {
                    ClipOp::Intersect
                },
            });
        } else {
            vclip = None;
        }
    }

    let has_perspective = rotation.is_some_and(|(x, y, _, _)| x != 0.0 || y != 0.0);
    let has_transform = rotation.is_some() || scale.is_some() || shear.is_some();
    // The pushed transform (if any), kept to expand blur-layer rects below:
    // rotated/projected content must not clip at the flat text rect.
    let pushed_transform: Option<Transform> = if has_perspective {
        // True perspective via the scene graph's projective row: libass's
        // exact 3D rotation (order, signs, shear coupling) about the `\org`
        // pivot (default: run centre), with the differentially-fit focal.
        // The subtree flattens into a layer and composites projectively.
        // Shear and `\fscx` are folded into the rows (libass order), so no
        // separate affine transform is emitted.
        let (x_deg, y_deg, z_deg, origin) = rotation.unwrap_or((0.0, 0.0, 0.0, None));
        let (fax, fay) = shear.unwrap_or((0.0, 0.0));
        let fscx = scale.map_or(100.0, |(x, _)| x * 100.0);
        let (px, py) = origin.unwrap_or((rect.x + rect.w * 0.5, rect.y + rect.h * 0.5));
        let (rx, ry, rp) =
            libass_rotation_rows(x_deg, y_deg, z_deg, fax, fay, fscx / 100.0, px, py);
        Some(Transform::from_projective_rows(rx, ry, rp))
    } else if has_transform {
        let (sx, sy) = scale.unwrap_or((1.0, 1.0));
        let (_, _, z_deg, origin) = rotation.unwrap_or((0.0, 0.0, 0.0, None));
        // ASS rotates counter-clockwise in degrees; repose takes radians.
        let rotate = -z_deg.to_radians();
        let (shear_x, shear_y) = shear.unwrap_or((0.0, 0.0));
        let (origin_x, origin_y) = origin.map_or((0.5, 0.5), |(ox, oy)| {
            // Normalized pivot, deliberately NOT clamped: a distant `\org`
            // (rotation lever) must keep its true position. Repose applies
            // the pivot in rect space, which is valid for any finite value.
            (
                (ox - rect.x) / rect.w.max(1.0),
                (oy - rect.y) / rect.h.max(1.0),
            )
        });
        Some(Transform {
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: sx,
            scale_y: sy,
            rotate,
            shear_x,
            shear_y,
            origin_x,
            origin_y,
            perspective: [0.0, 0.0, 1.0],
        })
    } else {
        None
    };
    if let Some(transform) = pushed_transform {
        nodes.push(SceneNode::PushTransform { transform });
    }

    // Full `\blur` wraps the whole run (fill, outline, shadow) in an
    // offscreen layer, like the reference's blur temp. `\be` is handled
    // per-outline inside `emit_fill_pass` and never reaches this layer.
    // Layers expect layer-local children (the Repose producer contract —
    // repose-ui pushes the same shift), so a `-rect` shift wraps the
    // content; rotation/clip nodes inside compose on top of it. The layer
    // rect covers the transformed run plus blur bleed.
    let layer_id = blur.filter(|r| *r > 0.0).map(|radius| {
        let id = layers_ctx.alloc();
        let layer_rect = expand_layer_rect(rect, pushed_transform, radius);
        nodes.push(SceneNode::BeginLayer {
            rect: layer_rect,
            layer_id: id,
            alpha: 1.0,
            blur_radius_x: Px(radius),
            blur_radius_y: Px(radius),
            rectangle_edge: true,
        });
        nodes.push(SceneNode::PushTransform {
            transform: Transform::translate(-layer_rect.x, -layer_rect.y),
        });
        id
    });

    if let Some((color, padding)) = opaque {
        nodes.push(SceneNode::Rect {
            rect: Rect {
                x: rect.x - padding,
                y: rect.y - padding,
                w: rect.w + padding * 2.0,
                h: rect.h + padding * 2.0,
            },
            brush: Brush::Solid(Color::from_rgba(color[0], color[1], color[2], color[3])),
            radius: [Px::ZERO; 4],
        });
    }

    if let Some((color, dx, dy)) = shadow {
        let shadow_pass = TextPass {
            rect: Rect {
                x: rect.x + dx,
                y: rect.y + dy,
                w: rect.w,
                h: rect.h,
            },
            color,
            decoration: TextDecoration::default(),
            ..base
        };
        emit_fill_pass(
            nodes,
            layers_ctx,
            &shadow_pass,
            outline,
            edge_blur,
            pushed_transform,
        );
    }

    if let Some((progress, style, secondary)) = karaoke {
        if style == 0 {
            // Basic `\k`: the whole run flips between sung and unsung at
            // progress > 0, exactly like the software reference — no sweep
            // window, which is reserved for styles 1-3 below.
            let sung = if progress > 0.0 {
                base
            } else {
                TextPass {
                    color: secondary,
                    ..base
                }
            };
            emit_fill_pass(
                nodes,
                layers_ctx,
                &sung,
                outline,
                edge_blur,
                pushed_transform,
            );
        } else {
            // Swept styles (`\K`, `\kf`, `\ko`): unsung base in the
            // secondary colour, then a clipped window of the sung colour
            // sweeping left to right.
            let unsung = TextPass {
                color: secondary,
                ..base
            };
            emit_fill_pass(
                nodes,
                layers_ctx,
                &unsung,
                outline,
                edge_blur,
                pushed_transform,
            );
            if progress > 0.0 {
                nodes.push(SceneNode::PushClip {
                    rect: Rect {
                        x: rect.x,
                        y: rect.y,
                        w: rect.w * progress,
                        h: rect.h,
                    },
                    radius: [Px::ZERO; 4],
                    op: ClipOp::Intersect,
                });
                emit_fill_pass(
                    nodes,
                    layers_ctx,
                    &base,
                    outline,
                    edge_blur,
                    pushed_transform,
                );
                nodes.push(SceneNode::PopClip);
            }
        }
    } else {
        emit_fill_pass(
            nodes,
            layers_ctx,
            &base,
            outline,
            edge_blur,
            pushed_transform,
        );
    }

    if let Some(id) = layer_id {
        nodes.push(SceneNode::PopTransform);
        nodes.push(SceneNode::EndLayer { layer_id: id });
    }
    if pushed_transform.is_some() {
        nodes.push(SceneNode::PopTransform);
    }
    if vclip.is_some() {
        nodes.push(SceneNode::PopVectorClip);
    }
    if clip.is_some() {
        nodes.push(SceneNode::PopClip);
    }
}

/// sRGB bytes → premultiplied-linear vertex colour.
fn premult_linear(color: [u8; 4]) -> [f32; 4] {
    let lin = Color::from_rgba(color[0], color[1], color[2], color[3]).to_linear();
    [lin[0] * lin[3], lin[1] * lin[3], lin[2] * lin[3], lin[3]]
}

/// Tessellate a lyon path into a solid `VectorMesh`.
fn tessellate_fill_mesh(path: &lyon_path::Path, color: [f32; 4]) -> Option<Arc<VectorMeshData>> {
    use lyon_tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex};

    let mut buffers: lyon_tessellation::VertexBuffers<[f32; 2], u32> =
        lyon_tessellation::VertexBuffers::new();
    FillTessellator::new()
        .tessellate(
            path,
            &FillOptions::tolerance(0.5),
            &mut BuffersBuilder::new(&mut buffers, |v: FillVertex| v.position().to_array()),
        )
        .ok()?;
    if buffers.indices.is_empty() {
        return None;
    }
    let vertices: Arc<[VectorVertex]> = buffers
        .vertices
        .iter()
        .map(|pos| VectorVertex {
            pos: *pos,
            color,
            uv: [0.0, 0.0],
        })
        .collect();
    Some(Arc::new(VectorMeshData {
        vertices,
        indices: buffers.indices.into(),
    }))
}

/// Tessellate a lyon path into `VectorMesh` nodes (fill + optional
/// stroke) for a vector drawing layer.
///
/// Returns `false` when there is no path or tessellation fails.
fn emit_vector(out: &mut BuiltScene, data: &VectorData) -> bool {
    use lyon_tessellation::{BuffersBuilder, StrokeOptions, StrokeTessellator};

    let Some(path) = &data.path else {
        return false;
    };
    // libass drawings are filled AND stroked (when a border is set), so emit
    // one mesh per pass instead of choosing stroke *instead of* fill. Each
    // pass carries its own colour: `data.color` for the fill, `stroke.color`
    // for the stroke (previously ignored).
    let mut emitted = 0;
    if let Some(mesh) = tessellate_fill_mesh(path, premult_linear(data.color)) {
        out.scene.nodes.push(vector_mesh_node(mesh));
        emitted += 1;
    } else {
        return false;
    }
    if data.stroke.is_some() {
        let width = data.stroke.as_ref().map_or(0.5, |s| s.width.max(0.5));
        let options = StrokeOptions::tolerance(0.5).with_line_width(width);
        let mut buffers: lyon_tessellation::VertexBuffers<[f32; 2], u32> =
            lyon_tessellation::VertexBuffers::new();
        let ok = StrokeTessellator::new()
            .tessellate(
                path,
                &options,
                &mut BuffersBuilder::new(&mut buffers, |v: lyon_tessellation::StrokeVertex| {
                    v.position().to_array()
                }),
            )
            .is_ok();
        if !ok || buffers.indices.is_empty() {
            return false;
        }
        let color = premult_linear(data.stroke.as_ref().map_or([0, 0, 0, 0], |s| s.color));
        let vertices: Arc<[VectorVertex]> = buffers
            .vertices
            .iter()
            .map(|pos| VectorVertex {
                pos: *pos,
                color,
                uv: [0.0, 0.0],
            })
            .collect();
        out.scene
            .nodes
            .push(vector_mesh_node(Arc::new(VectorMeshData {
                vertices,
                indices: buffers.indices.into(),
            })));
        emitted += 1;
    }
    emitted > 0
}

/// Wrap a tessellated mesh in a world-space `VectorMesh` scene node.
fn vector_mesh_node(mesh: Arc<VectorMeshData>) -> SceneNode {
    SceneNode::VectorMesh {
        mesh,
        // Repose 2x3 convention is `[m00, m01, m10, m11, tx, ty]`
        // (identity `[1, 0, 0, 1, 0, 0]`): the tessellated vertices are
        // already in world pixels, so no local transform applies.
        transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        paint: PaintDesc::Solid,
        clip: None,
        blend: BlendMode::Alpha,
    }
}

/// How much of the frame the reference software backend covered.
#[must_use]
pub fn covered_pixels(rgba: &[u8]) -> u64 {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[3] > 0)
        .count() as u64
}
