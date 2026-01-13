//! Debug GGUF metadata reader

use std::fs::File;
use std::io::{BufReader, Read};

fn read_u32<R: Read>(reader: &mut R) -> u32 {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).unwrap();
    u32::from_le_bytes(buf)
}

fn read_u64<R: Read>(reader: &mut R) -> u64 {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).unwrap();
    u64::from_le_bytes(buf)
}

fn read_string<R: Read>(reader: &mut R) -> String {
    let len = read_u64(reader) as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).unwrap();
    String::from_utf8_lossy(&buf).to_string()
}

fn main() {
    let path = std::env::args().nth(1).expect("path");
    let file = File::open(&path).unwrap();
    let mut reader = BufReader::new(file);

    // Magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).unwrap();
    println!("Magic: {:?}", std::str::from_utf8(&magic));

    // Version
    let version = read_u32(&mut reader);
    println!("Version: {}", version);

    // Counts
    let _tensor_count = read_u64(&mut reader);
    let kv_count = read_u64(&mut reader);
    println!("KV pairs: {}", kv_count);

    // Read KV pairs
    for _ in 0..kv_count.min(100) {
        let key = read_string(&mut reader);
        let type_id = read_u32(&mut reader);

        // Skip value based on type
        let value_str = match type_id {
            4 => format!("{}", read_u32(&mut reader)),
            7 => {
                let mut byte = [0u8; 1];
                reader.read_exact(&mut byte).unwrap();
                format!("{}", byte[0] != 0)
            }
            8 => read_string(&mut reader),
            9 => {
                let arr_type = read_u32(&mut reader);
                let arr_len = read_u64(&mut reader);
                // Skip array contents
                match arr_type {
                    5 => {
                        for _ in 0..arr_len {
                            read_u32(&mut reader);
                        }
                    } // I32
                    6 => {
                        for _ in 0..arr_len {
                            read_u32(&mut reader);
                        }
                    } // F32
                    8 => {
                        for _ in 0..arr_len {
                            read_string(&mut reader);
                        }
                    } // String
                    _ => {}
                }
                format!("Array({}, len={})", arr_type, arr_len)
            }
            _ => format!("type={}", type_id),
        };

        if key.contains("pre") || key.contains("model") {
            println!("  {}: {}", key, value_str);
        }
    }
}
