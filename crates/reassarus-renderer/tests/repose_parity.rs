//! Parity tests: Repose scene backend vs the tiny-skia software reference.
//!
//! Each test renders the same script through both paths:
//!
//! - reference: `reassarus-renderer`'s `SoftwarePipeline` + `SoftwareBackend`
//!   (tiny-skia) → RGBA; asserts the frame actually covers pixels, so the
//!   scene comparison below is anchored to a non-empty render.
//! - new: `backends::repose::layers_to_scene` → structural assertions on
//!   the emitted `SceneNode`s (the six parity areas: outlines/shadows/blur,
//!   clip/iclip, rotations/shear/scale, karaoke, vector drawings, animated
//!   transforms).
//!
//! Needs the `repose-backend` feature (plus `software-backend` for the
//! reference side) and rustc 1.98+. No GPU is needed: scene emission is
//! pure CPU.
#![cfg(all(feature = "repose-backend", feature = "software-backend"))]

use reassarus_core::parser::Script;
use reassarus_renderer::backends::repose::{covered_pixels, layers_to_scene};
use reassarus_renderer::backends::BackendType;
use reassarus_renderer::pipeline::{IntermediateLayer, Pipeline, SoftwarePipeline};
use reassarus_renderer::renderer::{EventSelector, RenderContext, Renderer};
use repose_core::SceneNode;

const HEAD: &str = "[Script Info]\nPlayResX: 1280\nPlayResY: 720\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,30,30,30,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

/// Run the software pipeline for one dialogue line and return the layers.
fn layers_at(time_cs: u32, dialogue_text: &str) -> Vec<IntermediateLayer> {
    let script_text =
        format!("{HEAD}Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,{dialogue_text}\n");
    // The borrow checker needs the owned string alive while layers borrow it.
    let script = Box::leak(Box::new(script_text));
    let script = Box::leak(Box::new(Script::parse(script).expect("parse")));
    let ctx = RenderContext::new(1280, 720);
    let mut selector = EventSelector::new();
    let active = selector
        .select_active(script, time_cs)
        .expect("select active");
    let mut pipeline = SoftwarePipeline::new();
    pipeline.prepare_script(script, None).expect("prepare");
    pipeline
        .process_events(&active.events, time_cs, &ctx)
        .expect("process")
}

/// Reference RGBA render; asserts the reference itself covers pixels.
fn reference_covers(dialogue_text: &str, time_cs: u32) -> u64 {
    let script_text =
        format!("{HEAD}Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,{dialogue_text}\n");
    let script = Script::parse(&script_text).expect("parse");
    let ctx = RenderContext::new(1280, 720);
    let mut renderer = Renderer::new(BackendType::Software, ctx).expect("renderer");
    let frame = renderer.render_frame(&script, time_cs).expect("render");
    covered_pixels(frame.data())
}

fn text_nodes(nodes: &[SceneNode]) -> Vec<&SceneNode> {
    nodes
        .iter()
        .filter(|n| matches!(n, SceneNode::Text { .. }))
        .collect()
}

#[test]
fn plain_fill_emits_single_text_node() {
    let n = reference_covers("Hello", 200);
    assert!(n > 0, "reference must cover pixels, got {n}");
    let layers = layers_at(200, "Hello");
    let built = layers_to_scene(&layers, 1280, 720);
    assert_eq!(built.skipped_raster, 0);
    let texts = text_nodes(&built.scene.nodes);
    assert_eq!(texts.len(), 1, "plain fill: one Text node");
}

#[test]
fn outline_emits_stroke_under_fill() {
    let dialogue = r"{\bord2\3c&H0000FF&}Hello";
    let n = reference_covers(dialogue, 200);
    assert!(n > 0, "reference must cover pixels, got {n}");
    let layers = layers_at(200, dialogue);
    let built = layers_to_scene(&layers, 1280, 720);
    let texts = text_nodes(&built.scene.nodes);
    assert_eq!(texts.len(), 2, "outline: stroke + fill Text nodes");
    let is_stroke = texts.iter().any(|t| match t {
        SceneNode::Text { style, .. } => {
            matches!(style.draw_style, repose_core::DrawStyle::Stroke { .. })
        }
        _ => false,
    });
    assert!(is_stroke, "one node must use DrawStyle::Stroke");
}

#[test]
fn shadow_emits_offset_pass() {
    let dialogue = r"{\shad2}Hello";
    let n = reference_covers(dialogue, 200);
    assert!(n > 0, "reference must cover pixels, got {n}");
    let layers = layers_at(200, dialogue);
    let built = layers_to_scene(&layers, 1280, 720);
    let texts = text_nodes(&built.scene.nodes);
    assert_eq!(texts.len(), 2, "shadow: offset pass + main Text nodes");
}

#[test]
fn blur_wraps_run_in_layer() {
    let dialogue = r"{\blur3}Hello";
    let n = reference_covers(dialogue, 200);
    assert!(n > 0, "reference must cover pixels, got {n}");
    let layers = layers_at(200, dialogue);
    let built = layers_to_scene(&layers, 1280, 720);
    let has_layer = built.scene.nodes.iter().any(|node| match node {
        SceneNode::BeginLayer { blur_radius_x, .. } => blur_radius_x.0 > 0.0,
        _ => false,
    });
    assert!(has_layer, "blur must emit a BeginLayer with radius");
    assert!(
        built
            .scene
            .nodes
            .iter()
            .any(|node| matches!(node, SceneNode::EndLayer { .. })),
        "blur layer must be closed"
    );
}

#[test]
fn clip_emits_intersect_and_iclip_difference() {
    let layers = layers_at(200, r"{\clip(10,10,200,100)}Hello");
    let built = layers_to_scene(&layers, 1280, 720);
    let clip = built.scene.nodes.iter().find_map(|node| match node {
        SceneNode::PushClip { rect, op, .. } => Some((rect, op)),
        _ => None,
    });
    let (rect, op) = clip.expect("clip must emit PushClip");
    assert!(matches!(op, repose_core::ClipOp::Intersect));
    assert_eq!((rect.x, rect.y, rect.w, rect.h), (10.0, 10.0, 190.0, 90.0));

    let layers = layers_at(200, r"{\iclip(10,10,200,100)}Hello");
    let built = layers_to_scene(&layers, 1280, 720);
    let op = built.scene.nodes.iter().find_map(|node| match node {
        SceneNode::PushClip { op, .. } => Some(op),
        _ => None,
    });
    assert!(
        matches!(op, Some(repose_core::ClipOp::Difference)),
        r"\iclip must emit Difference"
    );
}

#[test]
fn rotation_emits_transform() {
    let dialogue = r"{\frz45}Hello";
    let n = reference_covers(dialogue, 200);
    assert!(n > 0, "reference must cover pixels, got {n}");
    let layers = layers_at(200, dialogue);
    let built = layers_to_scene(&layers, 1280, 720);
    let rotate = built.scene.nodes.iter().find_map(|node| match node {
        SceneNode::PushTransform { transform } => Some(transform.rotate),
        _ => None,
    });
    let rotate = rotate.expect("rotation must emit PushTransform");
    assert!(
        (rotate.abs() - 45f32.to_radians()).abs() < 1e-4,
        "rotate must be ±45°, got {rotate}"
    );
}

#[test]
fn shear_emits_transform() {
    let layers = layers_at(200, r"{\fax2}Hello");
    let built = layers_to_scene(&layers, 1280, 720);
    let shear = built.scene.nodes.iter().find_map(|node| match node {
        SceneNode::PushTransform { transform } => Some((transform.shear_x, transform.shear_y)),
        _ => None,
    });
    let (shear_x, shear_y) = shear.expect("shear must emit PushTransform");
    assert!(
        (shear_x - 2.0).abs() < 1e-4 && shear_y.abs() < 1e-4,
        "\\fax2 must land in shear_x, got ({shear_x}, {shear_y})"
    );
}

#[test]
fn karaoke_sweep_emits_progress_clip() {
    // Swept style (`\K`): secondary base plus a clipped sung window.
    let dialogue = r"{\K100}Ka{\K100}ra";
    // Mid second syllable: first fully sung, second partially.
    let n = reference_covers(dialogue, 150);
    assert!(n > 0, "reference must cover pixels, got {n}");
    let layers = layers_at(150, dialogue);
    let built = layers_to_scene(&layers, 1280, 720);
    let clips: Vec<_> = built
        .scene
        .nodes
        .iter()
        .filter(|node| matches!(node, SceneNode::PushClip { .. }))
        .collect();
    assert!(
        !clips.is_empty(),
        "karaoke sweep must emit at least one progress PushClip"
    );
    assert!(
        text_nodes(&built.scene.nodes).len() >= 2,
        "karaoke needs sung + unsung Text nodes"
    );
}

#[test]
fn karaoke_basic_flips_without_sweep_clip() {
    // Basic `\k` flips the whole run at progress > 0 (like the software
    // reference) instead of sweeping: no karaoke PushClip may be emitted.
    let dialogue = r"{\k100}Ka{\k100}ra";
    let layers = layers_at(150, dialogue);
    let built = layers_to_scene(&layers, 1280, 720);
    assert_eq!(built.skipped_tess, 0);
    let clips = built
        .scene
        .nodes
        .iter()
        .filter(|node| matches!(node, SceneNode::PushClip { .. }))
        .count();
    assert_eq!(clips, 0, "basic \\k must not emit a sweep PushClip");
    assert!(
        text_nodes(&built.scene.nodes).len() >= 2,
        "basic \\k still needs sung + unsung Text nodes"
    );
}

#[test]
fn scale_percent_normalizes_to_multiplier() {
    // `\fscx150` is 150 PERCENT: the transform must carry 1.5x, and Y must
    // stay 1.0 because it is already baked into the font size at shaping.
    let layers = layers_at(200, r"{\fscx150}Hello");
    let built = layers_to_scene(&layers, 1280, 720);
    let scale = built.scene.nodes.iter().find_map(|node| match node {
        SceneNode::PushTransform { transform } => Some((transform.scale_x, transform.scale_y)),
        _ => None,
    });
    let (sx, sy) = scale.expect("scale must emit PushTransform");
    assert!(
        (sx - 1.5).abs() < 1e-4,
        "\\fscx150 must become 1.5x, got {sx}"
    );
    assert!(
        (sy - 1.0).abs() < 1e-4,
        "Y scale is pre-applied to the font size, got {sy}"
    );
}

#[test]
fn fay_alone_emits_shear() {
    // A lone `\fay` (no `\fax`) must still reach the transform; it used to
    // be dropped by the shear emission gate.
    let layers = layers_at(200, r"{\fay1}Hello");
    let built = layers_to_scene(&layers, 1280, 720);
    let shear = built.scene.nodes.iter().find_map(|node| match node {
        SceneNode::PushTransform { transform } => Some((transform.shear_x, transform.shear_y)),
        _ => None,
    });
    let (shear_x, shear_y) = shear.expect("lone \\fay must emit PushTransform");
    assert!(
        shear_x.abs() < 1e-4 && (shear_y - 1.0).abs() < 1e-4,
        "\\fay1 must land in shear_y, got ({shear_x}, {shear_y})"
    );
}

#[test]
fn text_carries_resolved_font_family() {
    // The pipeline resolves the style font; the adapter must forward it so
    // Repose selects the right face instead of its default.
    let layers = layers_at(200, "Hello");
    let built = layers_to_scene(&layers, 1280, 720);
    let family = built.scene.nodes.iter().find_map(|node| match node {
        SceneNode::Text { style, .. } => Some(style.font_family),
        _ => None,
    });
    assert_eq!(
        family,
        Some(Some("Arial")),
        "Text node must carry the resolved family"
    );
}

#[test]
fn edge_blur_wraps_outline_only() {
    // `\be` blurs just the outline stroke: the stroke lives inside a blur
    // layer while the fill stays outside it. (Full `\blur` wraps the run.)
    let dialogue = r"{\bord2\be2}Hello";
    let n = reference_covers(dialogue, 200);
    assert!(n > 0, "reference must cover pixels, got {n}");
    let layers = layers_at(200, dialogue);
    let built = layers_to_scene(&layers, 1280, 720);
    let nodes = &built.scene.nodes;
    let begin = nodes
        .iter()
        .position(|n| matches!(n, SceneNode::BeginLayer { .. }))
        .expect("\\be must open a blur layer");
    let end = nodes
        .iter()
        .position(|n| matches!(n, SceneNode::EndLayer { .. }))
        .expect("\\be layer must close");
    assert!(begin < end, "layer must open before it closes");
    let sharp_fill = nodes.iter().enumerate().any(|(i, n)| {
        matches!(
            n,
            SceneNode::Text { style, .. }
            if matches!(
                style.draw_style,
                repose_core::DrawStyle::Fill
            )
        ) && (i < begin || i > end)
    });
    assert!(
        sharp_fill,
        "the fill Text node must stay outside the edge-blur layer"
    );
}

#[test]
fn distant_org_keeps_true_pivot() {
    // A far-away `\org` (rotation lever) must not be clamped into the text
    // rectangle: Repose applies the normalized pivot in rect space.
    let layers = layers_at(200, r"{\org(10000,10000)\frz45}Hello");
    let built = layers_to_scene(&layers, 1280, 720);
    let origin = built.scene.nodes.iter().find_map(|node| match node {
        SceneNode::PushTransform { transform } => Some((transform.origin_x, transform.origin_y)),
        _ => None,
    });
    let (ox, oy) = origin.expect("rotation with \\org must emit PushTransform");
    assert!(
        ox > 1.0 && oy > 1.0,
        "distant \\org must stay outside [0,1], got ({ox}, {oy})"
    );
}

#[test]
fn vector_stroke_emits_fill_and_stroke_passes() {
    use reassarus_renderer::pipeline::{StrokeInfo, VectorData};
    // Drawings are filled AND stroked: one mesh per pass, each in its own
    // colour (the stroke used to replace the fill and ignore its colour).
    let mut builder = lyon_path::Path::builder();
    builder.begin(lyon_path::math::Point::new(0.0, 0.0));
    builder.line_to(lyon_path::math::Point::new(60.0, 0.0));
    builder.line_to(lyon_path::math::Point::new(60.0, 60.0));
    builder.line_to(lyon_path::math::Point::new(0.0, 60.0));
    builder.end(true);
    let layers = vec![IntermediateLayer::Vector(VectorData {
        path: Some(builder.build()),
        color: [255, 0, 0, 255],
        stroke: Some(StrokeInfo {
            color: [0, 0, 255, 255],
            width: 2.0,
        }),
        bounds: None,
    })];
    let built = layers_to_scene(&layers, 1280, 720);
    assert_eq!(built.skipped_tess, 0);
    let meshes: Vec<_> = built
        .scene
        .nodes
        .iter()
        .filter_map(|node| match node {
            SceneNode::VectorMesh { mesh, .. } => Some(mesh),
            _ => None,
        })
        .collect();
    assert_eq!(meshes.len(), 2, "fill + stroke must emit two meshes");
    for mesh in meshes {
        assert!(!mesh.indices.is_empty(), "each pass must have triangles");
    }
}

#[test]
fn vector_drawing_tessellates_to_mesh() {
    let dialogue = r"{\p1}m 0 0 l 60 0 l 60 60 l 0 60{\p0}";
    let n = reference_covers(dialogue, 200);
    assert!(n > 0, "reference must cover pixels, got {n}");
    let layers = layers_at(200, dialogue);
    let built = layers_to_scene(&layers, 1280, 720);
    assert_eq!(built.skipped_tess, 0);
    let meshes: Vec<_> = built
        .scene
        .nodes
        .iter()
        .filter(|node| matches!(node, SceneNode::VectorMesh { .. }))
        .collect();
    assert_eq!(meshes.len(), 1, "drawing must emit one VectorMesh");
    if let SceneNode::VectorMesh { mesh, .. } = meshes[0] {
        assert!(!mesh.indices.is_empty(), "mesh must have triangles");
        assert!(!mesh.vertices.is_empty(), "mesh must have vertices");
    }
}

#[test]
fn animated_move_stays_renderable() {
    let dialogue = r"{\move(10,10,400,300,0,1000)}Hello";
    // Mid-flight at 5s.
    let n = reference_covers(dialogue, 500);
    assert!(n > 0, "reference must cover pixels, got {n}");
    let layers = layers_at(500, dialogue);
    let built = layers_to_scene(&layers, 1280, 720);
    assert!(
        !built.scene.nodes.is_empty(),
        "animated transform must still emit a scene"
    );
    assert!(!text_nodes(&built.scene.nodes).is_empty());
}

#[test]
fn shadow_carries_full_ass_depth() {
    // No empirical 0.5 factor: libass 0.17.5 offsets the shadow by exactly
    // (depth, depth) — measured +6px for Shadow=6. PlayRes == frame size, so
    // the scale here is 1 and the IR must carry (6, 6).
    let layers = layers_at(200, r"{\shad6}X");
    let mut saw = false;
    for layer in &layers {
        if let IntermediateLayer::Text(data) = layer {
            for effect in data.effects.iter() {
                if let reassarus_renderer::pipeline::TextEffect::Shadow {
                    x_offset, y_offset, ..
                } = effect
                {
                    assert_eq!((*x_offset, *y_offset), (6.0, 6.0));
                    saw = true;
                }
            }
        }
    }
    assert!(saw, "shad6 must emit a Shadow effect");
}

#[test]
fn transparent_primary_culls_event_layers() {
    // libass emits nothing (glyph, outline AND shadow) once the primary is
    // (near-)fully transparent: `\1a&HFD&` renders, `\1a&HFE&` does not.
    assert!(
        layers_at(200, r"{\shad6\1a&HFD&}X")
            .iter()
            .any(|l| matches!(l, IntermediateLayer::Text(_))),
        "opacity 2/255 must still emit layers"
    );
    assert!(
        !layers_at(200, r"{\shad6\1a&HFF&}X")
            .iter()
            .any(|l| matches!(l, IntermediateLayer::Text(_))),
        "zero-opacity primary must cull the event's text layers"
    );
    assert!(
        !layers_at(200, r"{\shad6\1a&HFE&}X")
            .iter()
            .any(|l| matches!(l, IntermediateLayer::Text(_))),
        "opacity 1/255 must cull the event's text layers"
    );
}

#[test]
fn text_layers_carry_measured_bounds() {
    // The pipeline must populate shaping-measured bounds from the same cached
    // run it lays out with, and the Repose scene rect must use them instead
    // of the old glyph-count estimate.
    let layers = layers_at(200, r"{\fsp4}Hello");
    let data = layers
        .iter()
        .find_map(|l| match l {
            IntermediateLayer::Text(d) => Some(d),
            _ => None,
        })
        .expect("text layer");
    let m = data
        .measured
        .expect("pipeline must populate measured bounds");
    assert!(
        m.width > 0.0 && m.height > 0.0,
        "bounds must be positive: {m:?}"
    );
    assert!(
        m.baseline > 0.0 && m.baseline < m.height,
        "baseline must sit inside the box: {m:?}"
    );
    let expected = m.spaced_width(data.spacing, data.text.chars().count());
    assert_eq!(data.spacing, 4.0, "fsp4 must reach the layer");
    assert!(
        (expected - (m.width + 16.0)).abs() < 0.01,
        "5 glyphs at fsp4 add 4*4px: {expected} vs {}",
        m.width + 16.0
    );
    let built = layers_to_scene(&layers, 1280, 720);
    let node = text_nodes(&built.scene.nodes)
        .into_iter()
        .next()
        .expect("text node");
    if let SceneNode::Text { rect, .. } = node {
        assert!(
            (rect.w - expected).abs() < 0.01,
            "scene rect must use measured width: {} vs {expected}",
            rect.w
        );
        assert!(
            (rect.h - m.height).abs() < 0.01,
            "scene rect must use measured height: {} vs {}",
            rect.h,
            m.height
        );
    } else {
        panic!("expected a Text node");
    }
}

/// Find the first `PushTransform` in a scene.
fn first_push(nodes: &[SceneNode]) -> repose_core::Transform {
    nodes
        .iter()
        .find_map(|n| match n {
            SceneNode::PushTransform { transform } => Some(*transform),
            _ => None,
        })
        .expect("PushTransform")
}

#[test]
fn frx_emits_projective_transform() {
    // `\frx30` takes the perspective path: libass x-rotation direction in
    // the projective row (focal 312.5px), pivot at the run centre.
    let layers = layers_at(200, r"{\frx30}H");
    let built = layers_to_scene(&layers, 1280, 720);
    let t = first_push(&built.scene.nodes);
    assert!(t.has_perspective(), "frx must set a projective row");
    // libass `z = -y·sin(frx)`: negative y-slope, magnitude sin30/312.5.
    let expect_slope = -0.5f32 / 312.5;
    assert!(
        (t.perspective[0] - 0.0).abs() < 1e-6 && (t.perspective[1] - expect_slope).abs() < 1e-6,
        "perspective row {:?}, want [0, {expect_slope}, _]",
        t.perspective
    );
}

#[test]
fn frz_without_tilt_stays_affine() {
    // No `\frx`/`\fry`: the validated affine path is untouched.
    let layers = layers_at(200, r"{\frz30}H");
    let built = layers_to_scene(&layers, 1280, 720);
    let t = first_push(&built.scene.nodes);
    assert!(
        !t.has_perspective(),
        "pure frz must not take the perspective path"
    );
    assert!((t.rotate + 30f32.to_radians()).abs() < 1e-4);
}

#[test]
fn org_sets_projective_pivot() {
    // The `\org` pivot is a fixed point of the projective map.
    let layers = layers_at(200, r"{\frx30\org(200,150)}H");
    let built = layers_to_scene(&layers, 1280, 720);
    let t = first_push(&built.scene.nodes);
    assert!(t.has_perspective());
    let m = t.projective_matrix();
    let w = m[6] * 200.0 + m[7] * 150.0 + m[8];
    let x = (m[0] * 200.0 + m[1] * 150.0 + m[2]) / w;
    let y = (m[3] * 200.0 + m[4] * 150.0 + m[5]) / w;
    assert!(
        (x - 200.0).abs() < 0.01 && (y - 150.0).abs() < 0.01,
        "org must be a fixed point, mapped to ({x}, {y})"
    );
}
