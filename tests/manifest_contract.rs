use shimmytok::manifest::{read_manifest, read_metadata, ParserLimits};
use std::io::Cursor;

fn string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}
fn fixture(alignment: u32, dims: &[u64], quant: u32, offsets: &[u64]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((offsets.len() as u64).to_le_bytes());
    out.extend(1u64.to_le_bytes());
    string(&mut out, "general.alignment");
    out.extend(4u32.to_le_bytes());
    out.extend(alignment.to_le_bytes());
    for (i, offset) in offsets.iter().enumerate() {
        string(&mut out, &format!("weight{i}"));
        out.extend((dims.len() as u32).to_le_bytes());
        for dim in dims {
            out.extend(dim.to_le_bytes());
        }
        out.extend(quant.to_le_bytes());
        out.extend(offset.to_le_bytes());
    }
    let a = if alignment.is_power_of_two() {
        alignment as usize
    } else {
        32
    };
    let start = (out.len() + a - 1) & !(a - 1);
    out.resize(start + 1024, 0);
    out
}
fn parse(bytes: &[u8]) -> Result<shimmytok::manifest::ModelManifest, shimmytok::Error> {
    read_manifest(
        Cursor::new(bytes),
        bytes.len() as u64,
        ParserLimits::default(),
    )
}
fn rejects(bytes: &[u8], code: &str) {
    let err = parse(bytes).unwrap_err().to_string();
    assert!(err.contains(code), "{err} expected {code}");
}
#[test]
fn declared_alignment_controls_absolute_offset() {
    let bytes = fixture(64, &[8, 2], 0, &[64]);
    let m = parse(&bytes).unwrap();
    assert_eq!(m.header().alignment, 64);
    assert_eq!(m.data_start() % 64, 0);
    let t = m.tensor("weight0").unwrap();
    assert_eq!(t.absolute_offset(), m.data_start() + 64);
    assert_eq!(t.byte_len(), 64);
    assert_eq!(t.row_bytes(), 32);
}
#[test]
fn invalid_alignment_and_offsets_reject() {
    rejects(&fixture(0, &[8], 0, &[0]), "alignment");
    rejects(&fixture(48, &[8], 0, &[0]), "alignment");
    rejects(&fixture(64, &[8], 0, &[32]), "misaligned_tensor");
}
#[test]
fn shapes_quantization_and_ranges_reject() {
    rejects(&fixture(32, &[0], 0, &[0]), "zero_tensor_dimension");
    rejects(&fixture(32, &[u64::MAX, 2], 0, &[0]), "overflow");
    rejects(
        &fixture(32, &[8], 999, &[0]),
        "unsupported_quantization_layout",
    );
    rejects(&fixture(32, &[255], 12, &[0]), "partial_quantization_block");
    rejects(&fixture(32, &[8], 0, &[1024]), "tensor_out_of_bounds");
    rejects(&fixture(32, &[16], 0, &[0, 32]), "overlapping_tensors");
}
#[test]
fn truncation_and_limits_reject_without_panicking() {
    let bytes = fixture(32, &[8], 0, &[0]);
    for len in 0..80 {
        assert!(parse(&bytes[..len]).is_err(), "prefix {len}");
    }
    let limits = ParserLimits {
        tensors: 0,
        ..Default::default()
    };
    assert!(read_manifest(Cursor::new(&bytes), bytes.len() as u64, limits).is_err());
}
#[test]
fn metadata_reader_does_not_require_tokenizer_or_tensor_payload() {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(1u64.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    let meta = read_metadata(Cursor::new(bytes), ParserLimits::default()).unwrap();
    assert_eq!(meta.tensor_count, 1);
    assert_eq!(meta.alignment, 32);
    assert!(!meta.alignment_declared);
}
#[test]
fn duplicate_metadata_and_invalid_bool_reject() {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(2u64.to_le_bytes());
    for _ in 0..2 {
        string(&mut bytes, "x");
        bytes.extend(4u32.to_le_bytes());
        bytes.extend(1u32.to_le_bytes());
    }
    assert!(read_metadata(Cursor::new(bytes), ParserLimits::default())
        .unwrap_err()
        .to_string()
        .contains("duplicate"));
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(1u64.to_le_bytes());
    string(&mut bytes, "x");
    bytes.extend(7u32.to_le_bytes());
    bytes.push(2);
    assert!(read_metadata(Cursor::new(bytes), ParserLimits::default()).is_err());
}
#[test]
#[ignore = "requires SHIMMY_CERT_MODEL_DIR fixtures"]
fn exact_real_models_share_metadata_and_tensor_manifest() {
    let root = std::env::var("SHIMMY_CERT_MODEL_DIR").expect("fixture directory");
    for file in [
        "Qwen3-0.6B-Q4_K_M.gguf",
        "llama3.2-3b-instruct-dde5aa3f.gguf",
    ] {
        let path = std::path::Path::new(&root).join(file);
        let manifest = shimmytok::manifest::load_manifest(&path).unwrap();
        let tokenizer = shimmytok::gguf::load_metadata(&path).unwrap();
        assert_eq!(manifest.header().values, *tokenizer.metadata());
        assert!(!manifest.tensors().is_empty());
        assert!(manifest.tensor("token_embd.weight").is_some());
    }
}
