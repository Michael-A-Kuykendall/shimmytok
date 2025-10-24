use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use shimmytok::Tokenizer;
use std::path::Path;

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
        eprintln!(
            "Skipping encode benchmarks: model not found at {model_path}"
        );
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
        eprintln!(
            "Skipping decode benchmarks: model not found at {model_path}"
        );
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
        eprintln!(
            "Skipping load benchmarks: model not found at {model_path}"
        );
        return;
    }

    c.bench_function("load_tokenizer", |b| {
        b.iter(|| Tokenizer::from_gguf_file(black_box(&model_path)));
    });
}

fn bench_encode_batch(c: &mut Criterion) {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!(
            "Skipping batch benchmarks: model not found at {model_path}"
        );
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

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_load,
    bench_encode_batch
);
criterion_main!(benches);
