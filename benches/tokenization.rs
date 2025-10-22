use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use shimmytok::Tokenizer;
use std::path::Path;

fn get_model_path() -> String {
    std::env::var("GGUF_MODEL_PATH")
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .map(|h| h.join(".cache/models/gguf/gpt2.Q4_K_M.gguf"))
                .and_then(|p| p.to_str().map(String::from))
                .unwrap_or_else(|| "model.gguf".to_string())
        })
}

fn bench_encode(c: &mut Criterion) {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping encode benchmarks: model not found at {}", model_path);
        return;
    }
    
    let tokenizer = Tokenizer::from_gguf_file(&model_path)
        .expect("Failed to load tokenizer");
    
    let mut group = c.benchmark_group("encode");
    
    for size in [10, 100, 1000].iter() {
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
        eprintln!("Skipping decode benchmarks: model not found at {}", model_path);
        return;
    }
    
    let tokenizer = Tokenizer::from_gguf_file(&model_path)
        .expect("Failed to load tokenizer");
    
    let tokens: Vec<u32> = (0..1000).map(|i| i % tokenizer.vocab_size() as u32).collect();
    
    c.bench_function("decode_1000_tokens", |b| {
        b.iter(|| tokenizer.decode(black_box(&tokens), false));
    });
}

fn bench_load(c: &mut Criterion) {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping load benchmarks: model not found at {}", model_path);
        return;
    }
    
    c.bench_function("load_tokenizer", |b| {
        b.iter(|| Tokenizer::from_gguf_file(black_box(&model_path)));
    });
}

criterion_group!(benches, bench_encode, bench_decode, bench_load);
criterion_main!(benches);
