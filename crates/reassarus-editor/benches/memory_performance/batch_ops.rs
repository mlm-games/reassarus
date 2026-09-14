//! Benchmarks for batched style and tag operations on large documents.

use crate::common::generate_large_script;
use criterion::Criterion;
use reassarus_editor::commands::{
    BatchCommand, EditStyleCommand, EditorCommand, InsertTagCommand, RemoveTagCommand,
    ReplaceTagCommand,
};
use reassarus_editor::core::{EditorDocument, Position, Range};
use std::hint::black_box;

/// Benchmark batch operations on large documents
pub fn bench_batch_large_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_large_ops");

    let script = generate_large_script(5000, 50);

    // Batch style changes
    group.bench_function("batch_style_changes", |b| {
        b.iter_batched(
            || EditorDocument::from_content(&script).unwrap(),
            |mut doc| {
                let batch = BatchCommand::new("Batch style update".to_string())
                    .add_command(Box::new(
                        EditStyleCommand::new("Style1".to_string())
                            .set_size(24)
                            .set_bold(true),
                    ))
                    .add_command(Box::new(
                        EditStyleCommand::new("Style5".to_string()).set_font("Helvetica"),
                    ))
                    .add_command(Box::new(
                        EditStyleCommand::new("Style10".to_string()).set_color("&H00FF00FF"),
                    ));

                black_box(batch.execute(&mut doc).unwrap())
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Batch tag operations
    group.bench_function("batch_tag_ops", |b| {
        b.iter_batched(
            || EditorDocument::from_content(&script).unwrap(),
            |mut doc| {
                let batch = BatchCommand::new("Batch tag update".to_string())
                    .add_command(Box::new(ReplaceTagCommand::new(
                        Range::new(Position::new(0), Position::new(10000)),
                        "\\pos(960,540)".to_string(),
                        "\\pos(640,360)".to_string(),
                    )))
                    .add_command(Box::new(
                        RemoveTagCommand::new(Range::new(Position::new(0), Position::new(10000)))
                            .pattern("\\be1".to_string()),
                    ))
                    .add_command(Box::new(InsertTagCommand::new(
                        Position::new(1000),
                        "\\fade(255,0)".to_string(),
                    )));

                black_box(batch.execute(&mut doc).unwrap())
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}
