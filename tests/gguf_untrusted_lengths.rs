//! Regression for PR #3's untrusted lengths, against the newer shared manifest.
//! No older parser code is imported: https://github.com/Michael-A-Kuykendall/shimmytok/pull/3
use shimmytok::gguf::{load_metadata_from_reader, GGUFValue};
use shimmytok::manifest::{read_metadata, ParserLimits};
use std::io::{Cursor, Read};

fn header(entries: u64) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(entries.to_le_bytes());
    out
}
fn string(out: &mut Vec<u8>, value: &str) {
    out.extend((value.len() as u64).to_le_bytes());
    out.extend(value.as_bytes());
}
fn array(element: u32, count: u64) -> Vec<u8> {
    let mut out = header(1);
    string(&mut out, "tokenizer.ggml.tokens");
    out.extend(9u32.to_le_bytes());
    out.extend(element.to_le_bytes());
    out.extend(count.to_le_bytes());
    out
}
struct NoPayloadRead(Cursor<Vec<u8>>);
impl Read for NoPayloadRead {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        assert!(
            self.0.position() < self.0.get_ref().len() as u64,
            "parser attempted payload read before rejecting untrusted allocation"
        );
        self.0.read(out)
    }
}
#[test]
fn five_billion_elements_reject_before_payload_for_every_known_type() {
    for element in 0..=12 {
        let error =
            load_metadata_from_reader(NoPayloadRead(Cursor::new(array(element, 5_000_000_000))))
                .err()
                .expect("malformed metadata accepted")
                .to_string();
        assert!(error.contains("count_limit"), "type {element}: {error}");
        assert!(error.contains("5000000000"), "wrong count: {error}");
    }
}
#[test]
fn allocation_budget_is_enforced_below_element_limit_for_every_known_type() {
    let limits = ParserLimits::default();
    let count = limits.allocation_bytes / std::mem::size_of::<GGUFValue>() as u64 + 1;
    assert!(
        count <= limits.array_elements,
        "fixture must reach budget guard"
    );
    for element in 0..=12 {
        let error = load_metadata_from_reader(NoPayloadRead(Cursor::new(array(element, count))))
            .err()
            .expect("malformed metadata accepted")
            .to_string();
        assert!(
            error.contains("allocation_limit"),
            "type {element}: {error}"
        );
    }
}
#[test]
fn maximal_counts_and_invalid_element_types_fail_without_payload() {
    let error = load_metadata_from_reader(NoPayloadRead(Cursor::new(array(8, u64::MAX))))
        .err()
        .expect("malformed metadata accepted")
        .to_string();
    assert!(error.contains("count_limit"), "{error}");
    let error = load_metadata_from_reader(NoPayloadRead(Cursor::new(array(13, 5_000_000_000))))
        .err()
        .expect("malformed metadata accepted")
        .to_string();
    assert!(error.contains("unsupported_metadata_type"), "{error}");
}
#[test]
fn multiplication_overflow_is_checked_when_custom_limits_allow_the_count() {
    let limits = ParserLimits {
        array_elements: u64::MAX,
        ..ParserLimits::default()
    };
    let error = read_metadata(NoPayloadRead(Cursor::new(array(8, u64::MAX))), limits)
        .expect_err("malformed metadata accepted")
        .to_string();
    assert!(error.contains("overflow"), "{error}");
}
#[test]
fn truncated_headers_counts_and_payloads_return_errors() {
    let full = array(8, 1);
    for len in [0, 3, 7, 15, 23, full.len() - 1, full.len()] {
        assert!(load_metadata_from_reader(Cursor::new(full[..len].to_vec())).is_err());
    }
    let mut short_string = full;
    short_string.extend(4u64.to_le_bytes());
    short_string.extend(b"ab");
    assert!(load_metadata_from_reader(Cursor::new(short_string)).is_err());
}
#[test]
fn large_bounded_tokenizer_metadata_is_preserved() {
    let count = 131_072;
    let mut data = array(8, count);
    for i in 0..count {
        string(&mut data, &format!("token-{i}"));
    }
    let metadata = load_metadata_from_reader(Cursor::new(data)).unwrap();
    assert_eq!(metadata.tokens.len(), count as usize);
    assert_eq!(metadata.tokens.first().unwrap(), "token-0");
    assert_eq!(metadata.tokens.last().unwrap(), "token-131071");
    assert_eq!(metadata.model_type, "llama");
}
