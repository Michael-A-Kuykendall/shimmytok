use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;
use shimmytok::Tokenizer;
use std::path::Path;
use std::time::Duration;

// Reuse the in-memory GGUF fixtures from the test suite so the batch-backend
// benchmarks run with NO model files (works locally and in CI).
#[path = "../tests/common/mod.rs"]
mod common;

fn get_model_path() -> String {
    std::env::var("GGUF_MODEL_PATH").unwrap_or_else(|_| {
        dirs::home_dir()
            .map(|h| h.join(".cache/models/gguf/gpt2.Q4_K_M.gguf"))
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_else(|| "model.gguf".to_string())
    })
}

fn bench_encode(c: &mut Criterion) {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping encode benchmarks: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load tokenizer");

    let mut group = c.benchmark_group("encode");

    for size in &[10, 100, 1000] {
        let text = "Hello world ".repeat(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| tokenizer.encode(black_box(&text), false));
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping decode benchmarks: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load tokenizer");

    let tokens: Vec<u32> = (0..1000)
        .map(|i| i % tokenizer.vocab_size() as u32)
        .collect();

    c.bench_function("decode_1000_tokens", |b| {
        b.iter(|| tokenizer.decode(black_box(&tokens), false));
    });
}

fn bench_load(c: &mut Criterion) {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping load benchmarks: model not found at {model_path}");
        return;
    }

    c.bench_function("load_tokenizer", |b| {
        b.iter(|| Tokenizer::from_gguf_file(black_box(&model_path)));
    });
}

fn bench_encode_batch(c: &mut Criterion) {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping batch benchmarks: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load tokenizer");

    let mut group = c.benchmark_group("encode_batch");

    for batch_size in &[1, 10, 100] {
        let texts: Vec<String> = (0..*batch_size)
            .map(|i| format!("This is test string number {i} with some content"))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| tokenizer.encode_batch(black_box(&text_refs), false));
            },
        );
    }
    group.finish();
}

fn bench_multi_pattern_models(c: &mut Criterion) {
    // Benchmark multi-pattern models (DeepSeek, StarCoder)
    let models = vec![
        ("deepseek-llm", "deepseek-llm-7b-chat-q4_k_m.gguf"),
        ("starcoder", "starcoder2-3b-q4_k_m.gguf"),
    ];

    for (name, filename) in models {
        let model_path = dirs::home_dir()
            .map(|h| h.join(".cache/models/gguf").join(filename))
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_else(|| filename.to_string());

        if !Path::new(&model_path).exists() {
            eprintln!("Skipping {name} benchmark: model not found");
            continue;
        }

        let tokenizer = match Tokenizer::from_gguf_file(&model_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to load {name}: {e}");
                continue;
            }
        };

        let text = "Hello world! This is a test with numbers 123 and 中文.";
        c.bench_function(&format!("encode_{name}"), |b| {
            b.iter(|| tokenizer.encode(black_box(text), false));
        });
    }
}

fn bench_sentencepiece_models(c: &mut Criterion) {
    // Benchmark SentencePiece models (Llama-3, Mistral)
    let models = vec![
        ("llama3", "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"),
        ("mistral", "mistral-7b-instruct-v0.2-q4_k_m.gguf"),
    ];

    for (name, filename) in models {
        let model_path = dirs::home_dir()
            .map(|h| h.join(".cache/models/gguf").join(filename))
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_else(|| filename.to_string());

        if !Path::new(&model_path).exists() {
            eprintln!("Skipping {name} benchmark: model not found");
            continue;
        }

        let tokenizer = match Tokenizer::from_gguf_file(&model_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to load {name}: {e}");
                continue;
            }
        };

        let text = "The quick brown fox jumps over the lazy dog.";
        c.bench_function(&format!("encode_{name}"), |b| {
            b.iter(|| tokenizer.encode(black_box(text), false));
        });
    }
}

/// Portable, model-free comparison of the sequential vs parallel batch
/// backends across batch sizes and payload sizes.
///
/// This is the data source for the `PARALLEL_BATCH_THRESHOLD` chosen in
/// `src/lib.rs`. It measures the two backends head-to-head so the crossover
/// batch size (where parallel begins to win) can be read directly. Because it
/// uses an in-memory GGUF fixture, it needs no model on disk.
fn bench_batch_backends(c: &mut Criterion) {
    let tok = Tokenizer::from_bytes(&common::bpe_gpt2_fixture()).expect("fixture tokenizer");

    let mut group = c.benchmark_group("batch_backends");
    // Threshold-finding does not need high precision; keep total runtime sane.
    group.sample_size(30);
    group.measurement_time(Duration::from_millis(800));
    group.warm_up_time(Duration::from_millis(300));

    // Two payload sizes: a short token-like input and a longer document-like
    // input, so the per-item cost (and thus the crossover) can be compared.
    let short = "abcacbab".to_string();
    let long = "abcacbab ".repeat(128); // ~1 KB of BPE work per item

    for (payload_name, payload) in [("short", &short), ("long", &long)] {
        for &batch in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
            let texts: Vec<&str> = vec![payload.as_str(); batch];

            group.bench_with_input(
                BenchmarkId::new(format!("sequential/{payload_name}"), batch),
                &batch,
                |b, _| {
                    b.iter(|| {
                        let r: Result<Vec<_>, _> = texts
                            .iter()
                            .map(|t| tok.encode(black_box(t), false))
                            .collect();
                        black_box(r)
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("parallel/{payload_name}"), batch),
                &batch,
                |b, _| {
                    b.iter(|| {
                        let r: Result<Vec<_>, _> = texts
                            .par_iter()
                            .map(|t| tok.encode(black_box(t), false))
                            .collect();
                        black_box(r)
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_load,
    bench_encode_batch,
    bench_multi_pattern_models,
    bench_sentencepiece_models,
    bench_batch_backends
);
criterion_main!(benches);
