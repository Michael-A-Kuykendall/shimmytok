//! GGUF file format reader

use crate::{Error, TokenType};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

pub struct GGUFMetadata {
    pub tokens: Vec<String>,
    pub scores: Option<Vec<f32>>,
    pub token_types: Option<Vec<TokenType>>,
    pub model_type: String,
    pub pre_type: Option<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub add_space_prefix: Option<bool>,
    pub merges: Option<Vec<(String, String)>>,
}

pub fn load_metadata<P: AsRef<Path>>(path: P) -> Result<GGUFMetadata, Error> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Track total string allocation to prevent OOM (Issue R2#3)
    let mut total_string_bytes: usize = 0;

    // Read magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;

    if &magic != b"GGUF" {
        return Err(Error::InvalidMetadata("Not a GGUF file".into()));
    }

    // Read version
    let version = read_u32(&mut reader)?;
    // Support both version 2 and 3 (GPT-2 uses v3)
    if !(2..=3).contains(&version) {
        return Err(Error::InvalidMetadata(format!(
            "Unsupported GGUF version: {} (only versions 2-3 are supported)",
            version
        )));
    }

    // Read counts
    let _tensor_count = read_u64(&mut reader)?;
    let metadata_count = read_u64(&mut reader)?;

    // Read metadata key-value pairs
    let mut kv_pairs = HashMap::new();
    for _ in 0..metadata_count {
        let key = read_string(&mut reader, &mut total_string_bytes)?;
        let value = read_value(&mut reader, &mut total_string_bytes)?;
        kv_pairs.insert(key, value);
    }

    // Extract tokenizer metadata
    let tokens = match kv_pairs.get("tokenizer.ggml.tokens") {
        Some(Value::StringArray(arr)) => arr.clone(),
        _ => {
            return Err(Error::InvalidMetadata(
                "Missing tokenizer.ggml.tokens".into(),
            ))
        }
    };

    let scores = match kv_pairs.get("tokenizer.ggml.scores") {
        Some(Value::F32Array(arr)) => Some(arr.clone()),
        _ => None,
    };

    let token_types = match kv_pairs.get("tokenizer.ggml.token_type") {
        Some(Value::I32Array(arr)) => Some(arr.iter().map(|&t| TokenType::from(t)).collect()),
        _ => None,
    };

    let model_type = match kv_pairs.get("tokenizer.ggml.model") {
        Some(Value::String(s)) => s.clone(),
        _ => "llama".to_string(), // Default
    };

    let pre_type = match kv_pairs.get("tokenizer.ggml.pre") {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    };

    // Special tokens
    let bos_token_id = match kv_pairs.get("tokenizer.ggml.bos_token_id") {
        Some(Value::U32(v)) => Some(*v),
        _ => None,
    };

    let eos_token_id = match kv_pairs.get("tokenizer.ggml.eos_token_id") {
        Some(Value::U32(v)) => Some(*v),
        _ => None,
    };

    let unk_token_id = match kv_pairs.get("tokenizer.ggml.unknown_token_id") {
        Some(Value::U32(v)) => Some(*v),
        _ => None,
    };

    let pad_token_id = match kv_pairs.get("tokenizer.ggml.padding_token_id") {
        Some(Value::U32(v)) => Some(*v),
        _ => None,
    };

    // Flags
    let add_bos_token = match kv_pairs.get("tokenizer.ggml.add_bos_token") {
        Some(Value::Bool(v)) => Some(*v),
        _ => None,
    };

    let add_eos_token = match kv_pairs.get("tokenizer.ggml.add_eos_token") {
        Some(Value::Bool(v)) => Some(*v),
        _ => None,
    };

    let add_space_prefix = match kv_pairs.get("tokenizer.ggml.add_space_prefix") {
        Some(Value::Bool(v)) => Some(*v),
        _ => None,
    };

    // BPE merges
    let merges = match kv_pairs.get("tokenizer.ggml.merges") {
        Some(Value::StringArray(arr)) => {
            let mut result = Vec::new();
            for merge_str in arr {
                let parts: Vec<&str> = merge_str.split(' ').collect();
                if parts.len() == 2 {
                    result.push((parts[0].to_string(), parts[1].to_string()));
                }
            }
            Some(result)
        }
        _ => None,
    };

    Ok(GGUFMetadata {
        tokens,
        scores,
        token_types,
        model_type,
        pre_type,
        bos_token_id,
        eos_token_id,
        unk_token_id,
        pad_token_id,
        add_bos_token,
        add_eos_token,
        add_space_prefix,
        merges,
    })
}

#[derive(Debug)]
#[allow(dead_code)]
enum Value {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    StringArray(Vec<String>),
    I32Array(Vec<i32>),
    F32Array(Vec<f32>),
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, Error> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, Error> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32, Error> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32, Error> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_string<R: Read>(reader: &mut R, total_bytes: &mut usize) -> Result<String, Error> {
    const MAX_STRING_SIZE: usize = 1024 * 1024; // 1MB max per string
    const MAX_TOTAL_STRING_DATA: usize = 100 * 1024 * 1024; // 100MB total
    let len_u64 = read_u64(reader)?;
    
    // Prevent truncation on 32-bit systems (Issue R4#12)
    if len_u64 > usize::MAX as u64 {
        return Err(Error::InvalidMetadata(format!(
            "String length {} exceeds platform limit",
            len_u64
        )));
    }
    let len = len_u64 as usize;
    
    if len > MAX_STRING_SIZE {
        return Err(Error::InvalidMetadata(format!(
            "String too large: {} bytes (max: {})",
            len, MAX_STRING_SIZE
        )));
    }

    // Check total allocation with overflow protection (Issue R3#2)
    *total_bytes = total_bytes.checked_add(len)
        .ok_or_else(|| Error::InvalidMetadata(
            "Total string data overflow".to_string()
        ))?;
    if *total_bytes > MAX_TOTAL_STRING_DATA {
        return Err(Error::InvalidMetadata(format!(
            "Total string data too large: {} bytes (max: {})",
            *total_bytes, MAX_TOTAL_STRING_DATA
        )));
    }
    
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| Error::InvalidMetadata(format!("Invalid UTF-8: {}", e)))
}

fn read_value<R: Read>(reader: &mut R, total_bytes: &mut usize) -> Result<Value, Error> {
    let type_id = read_u32(reader)?;

    match type_id {
        4 => Ok(Value::U32(read_u32(reader)?)),
        5 => Ok(Value::I32(read_i32(reader)?)),
        6 => Ok(Value::F32(read_f32(reader)?)),
        7 => {
            let mut byte = [0u8; 1];
            reader.read_exact(&mut byte)?;
            Ok(Value::Bool(byte[0] != 0))
        }
        8 => Ok(Value::String(read_string(reader, total_bytes)?)),
        9 => {
            // Array
            let array_type = read_u32(reader)?;
            let array_len = read_u64(reader)? as usize;

            match array_type {
                5 => {
                    // I32 array
                    let mut arr = Vec::with_capacity(array_len);
                    for _ in 0..array_len {
                        arr.push(read_i32(reader)?);
                    }
                    Ok(Value::I32Array(arr))
                }
                6 => {
                    // F32 array
                    let mut arr = Vec::with_capacity(array_len);
                    for _ in 0..array_len {
                        arr.push(read_f32(reader)?);
                    }
                    Ok(Value::F32Array(arr))
                }
                8 => {
                    // String array
                    let mut arr = Vec::with_capacity(array_len);
                    for _ in 0..array_len {
                        arr.push(read_string(reader, total_bytes)?);
                    }
                    Ok(Value::StringArray(arr))
                }
                _ => Err(Error::InvalidMetadata(format!(
                    "Unsupported array type: {}",
                    array_type
                ))),
            }
        }
        _ => Err(Error::InvalidMetadata(format!(
            "Unsupported value type: {}",
            type_id
        ))),
    }
}
