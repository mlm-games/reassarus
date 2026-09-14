//! Benchmarks for basic and comprehensive validation on large documents.

use crate::common::generate_large_script;
use criterion::{BenchmarkId, Criterion};
use reassarus_editor::core::EditorDocument;
use std::hint::black_box;

/// Benchmark validation on large documents
pub fn bench_large_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_validation");

    for size in [1000, 5000, 10000].iter() {
        let script = generate_large_script(*size, 50);

        group.bench_with_input(BenchmarkId::new("validate_basic", size), size, |b, _| {
            b.iter_batched(
                || EditorDocument::from_content(&script).unwrap(),
                |doc| {
                    doc.validate().unwrap();
                    black_box(())
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(
            BenchmarkId::new("validate_comprehensive", size),
            size,
            |b, _| {
                b.iter_batched(
                    || EditorDocument::from_content(&script).unwrap(),
                    |mut doc| black_box(doc.validate_comprehensive().unwrap()),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}
