use shimmytok::{EncodeOptions, Error, GGUFValue, SpecialTokenOverride, Tokenizer};

fn push_key(buf: &mut Vec<u8>, key: &str) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
}

fn push_string_value(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&8u32.to_le_bytes());
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn push_u8_value(buf: &mut Vec<u8>, value: u8) {
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.push(value);
}

fn push_u8_array_value(buf: &mut Vec<u8>, values: &[u8]) {
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    buf.extend_from_slice(values);
}

fn fixture() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&4u64.to_le_bytes());

    push_key(&mut buf, "tokenizer.ggml.tokens");
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());
    buf.extend_from_slice(&5u64.to_le_bytes());
    for token in ["<unk>", "hello", "world", "<s>", "</s>"] {
        buf.extend_from_slice(&(token.len() as u64).to_le_bytes());
        buf.extend_from_slice(token.as_bytes());
    }

    push_key(&mut buf, "test.unknown_u8");
    push_u8_value(&mut buf, 7);
    push_key(&mut buf, "test.unknown_string");
    push_string_value(&mut buf, "preserved");
    push_key(&mut buf, "test.unknown_array");
    push_u8_array_value(&mut buf, &[1, 2, 3]);

    buf
}

fn tokenizer() -> Tokenizer {
    Tokenizer::from_bytes(&fixture()).expect("fixture should load")
}

fn parse_options(add_special_tokens: bool) -> EncodeOptions {
    EncodeOptions::with_parse_special(add_special_tokens, true)
}

#[test]
fn external_marker_is_inserted_at_exact_position() {
    let tokenizer = tokenizer();
    let options = parse_options(false);
    let result = tokenizer
        .encode_with_external_special_tokens(
            "<image>",
            &options,
            &[SpecialTokenOverride::new("<image>", 2)],
        )
        .unwrap();

    assert_eq!(result, vec![2]);
}

#[test]
fn multiple_external_markers_preserve_input_order() {
    let tokenizer = tokenizer();
    let options = parse_options(false);
    let result = tokenizer
        .encode_with_external_special_tokens(
            "<image><audio>",
            &options,
            &[
                SpecialTokenOverride::new("<image>", 2),
                SpecialTokenOverride::new("<audio>", 3),
            ],
        )
        .unwrap();

    assert_eq!(result, vec![2, 3]);
}

#[test]
fn bos_is_added_once_with_external_markers() {
    let tokenizer = tokenizer();
    let options = parse_options(true);
    let result = tokenizer
        .encode_with_external_special_tokens(
            "<image>",
            &options,
            &[SpecialTokenOverride::new("<image>", 2)],
        )
        .unwrap();

    assert_eq!(result, vec![1, 2]);
}

#[test]
fn external_markers_are_ignored_without_opt_in_parsing() {
    let tokenizer = tokenizer();
    let options = EncodeOptions::with_special_tokens(false);
    let normal = tokenizer.encode("<image>", false).unwrap();
    let with_override = tokenizer
        .encode_with_external_special_tokens(
            "<image>",
            &options,
            &[SpecialTokenOverride::new("<image>", 2)],
        )
        .unwrap();

    assert_eq!(with_override, normal);
}

#[test]
fn invalid_external_ids_and_collisions_are_rejected() {
    let tokenizer = tokenizer();
    let options = parse_options(false);

    let unknown = tokenizer.encode_with_external_special_tokens(
        "<image>",
        &options,
        &[SpecialTokenOverride::new("<image>", 99)],
    );
    assert!(matches!(unknown, Err(Error::InvalidSpecialToken(_))));

    let collision = tokenizer.encode_with_external_special_tokens(
        "<unk>",
        &options,
        &[SpecialTokenOverride::new("<unk>", 2)],
    );
    assert!(matches!(collision, Err(Error::InvalidSpecialToken(_))));
}

#[test]
fn duplicate_external_markers_are_rejected() {
    let tokenizer = tokenizer();
    let options = parse_options(false);
    let result = tokenizer.encode_with_external_special_tokens(
        "<image>",
        &options,
        &[
            SpecialTokenOverride::new("<image>", 2),
            SpecialTokenOverride::new("<image>", 3),
        ],
    );

    assert!(matches!(result, Err(Error::InvalidSpecialToken(_))));
}

#[test]
fn unknown_metadata_values_are_preserved() {
    let tokenizer = tokenizer();
    let metadata = tokenizer.metadata();

    assert_eq!(metadata.get("test.unknown_u8"), Some(&GGUFValue::U8(7)));
    assert_eq!(
        metadata.get("test.unknown_string"),
        Some(&GGUFValue::String("preserved".to_string()))
    );
    assert_eq!(
        metadata.get("test.unknown_array"),
        Some(&GGUFValue::Array(vec![
            GGUFValue::U8(1),
            GGUFValue::U8(2),
            GGUFValue::U8(3),
        ]))
    );
}
