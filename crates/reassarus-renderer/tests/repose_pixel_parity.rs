//! Pixel-level parity: Repose GPU backend vs the tiny-skia software reference.
//!
//! The two backends rasterize text with different stacks (Repose shapes through
//! its own text system; the reference uses the pipeline's rustybuzz/tiny-skia
//! path), so these assert *structural* similarity — both sides cover pixels,
//! ink mass is in the same ballpark, line centres agree — rather than
//! bit-identity. Exact-pixel A/B against libass lives in
//! `examples/libass_ffi_compare.rs`.
//!
//! Needs a WGPU adapter (real GPU or lavapipe). Without one the tests skip
//! loudly instead of failing: the Repose backend's documented contract is to
//! fail `composite_layers` with an adapter error, which we detect.

#![cfg(all(feature = "repose-backend", feature = "software-backend"))]

use reassarus_core::parser::Script;
use reassarus_renderer::backends::BackendType;
use reassarus_renderer::renderer::{RenderContext, Renderer};

const HEAD: &str = "[Script Info]\nPlayResX: 640\nPlayResY: 360\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,DejaVu Sans,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,2,5,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

/// Render one dialogue line through `backend`. Returns `None` when the Repose
/// backend has no WGPU adapter (skip), so adapter-less CI stays green.
fn render(backend: BackendType, dialogue: &str) -> Option<Vec<u8>> {
    let text = format!("{HEAD}Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,{dialogue}\n");
    let script = Script::parse(&text).expect("parse");
    let ctx = RenderContext::new(640, 360);
    let mut renderer = Renderer::new(backend, ctx).expect("renderer");
    match renderer.render_frame(&script, 200) {
        Ok(frame) => Some(frame.data().to_vec()),
        Err(e) => {
            let msg = format!("{e}");
            if backend == BackendType::Repose && msg.contains("WGPU adapter") {
                eprintln!("SKIP: no WGPU adapter for Repose pixel parity ({msg})");
                return None;
            }
            panic!("render failed: {e}");
        }
    }
}

/// (covered pixels, ink mass, centre x, centre y) of a premultiplied RGBA frame.
fn stats(rgba: &[u8]) -> (u64, u64, f64, f64) {
    let (mut n, mut mass, mut sx, mut sy) = (0u64, 0u64, 0u64, 0u64);
    for (i, px) in rgba.as_chunks::<4>().0.iter().enumerate() {
        if px[3] > 8 {
            n += 1;
            mass += u64::from(px[3]);
            sx += (i % 640) as u64;
            sy += (i / 640) as u64;
        }
    }
    (
        n,
        mass,
        sx as f64 / n.max(1) as f64,
        sy as f64 / n.max(1) as f64,
    )
}

/// Structural comparison shared by the tests below.
fn assert_structural_parity(dialogue: &str) {
    let Some(soft) = render(BackendType::Software, dialogue) else {
        unreachable!("software backend never skips");
    };
    let Some(gpu) = render(BackendType::Repose, dialogue) else {
        return; // skipped: no adapter
    };
    let (sn, smass, scx, scy) = stats(&soft);
    let (gn, gmass, gcx, gcy) = stats(&gpu);
    assert!(sn > 100, "reference must cover pixels, got {sn}");
    assert!(gn > 100, "repose must cover pixels, got {gn}");
    let mass_ratio = gmass as f64 / smass.max(1) as f64;
    assert!(
        (0.25..4.0).contains(&mass_ratio),
        "ink mass ballpark: software={smass} repose={gmass}"
    );
    assert!(
        (gcx - scx).abs() < 24.0 && (gcy - scy).abs() < 24.0,
        "line centres must agree: soft=({scx:.1},{scy:.1}) repose=({gcx:.1},{gcy:.1})"
    );
    let _ = sn;
    let _ = gn;
}

#[test]
fn plain_text_structural_parity() {
    assert_structural_parity("Hello");
}

#[test]
fn shadow_outline_structural_parity() {
    assert_structural_parity(r"{\bord3\shad4}Hello");
}

#[test]
fn vector_drawing_structural_parity() {
    assert_structural_parity(r"{\p1}m 0 0 l 100 0 l 100 100 l 0 100{\p0}");
}
