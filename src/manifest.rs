//! One bounded GGUF wire interpretation shared by tokenizer and engine adapters.
//! Structural parsing does not imply that an architecture/operator is executable.
use crate::gguf::GGUFValue;
use crate::Error;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct ParserLimits {
    pub metadata_entries: u64,
    pub tensors: u64,
    pub string_bytes: u64,
    pub array_elements: u64,
    pub allocation_bytes: u64,
    pub nesting_depth: u32,
}
impl Default for ParserLimits {
    fn default() -> Self {
        Self {
            metadata_entries: 1_000_000,
            tensors: 1_000_000,
            string_bytes: 1_048_576,
            array_elements: 10_000_000,
            allocation_bytes: 256 * 1_048_576,
            nesting_depth: 8,
        }
    }
}
#[derive(Clone, Debug)]
pub struct ParsedMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub values: HashMap<String, GGUFValue>,
    pub array_element_types: HashMap<String, u32>,
    pub alignment: u32,
    pub alignment_declared: bool,
    pub metadata_end: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantLayout {
    pub elements_per_block: u64,
    pub bytes_per_block: u64,
}
/// Known storage layouts, independent of engine/kernel execution support.
pub fn quant_layout(kind: u32) -> Option<QuantLayout> {
    let (elements_per_block, bytes_per_block) = match kind {
        0 => (1, 4),
        1 | 25 | 30 => (1, 2),
        24 => (1, 1),
        26 => (1, 4),
        27 | 28 => (1, 8),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 40),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        _ => return None,
    };
    Some(QuantLayout {
        elements_per_block,
        bytes_per_block,
    })
}
#[derive(Clone, Debug)]
pub struct TensorDescriptor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    relative_offset: u64,
    absolute_offset: u64,
    byte_len: u64,
    row_bytes: u64,
    layout: QuantLayout,
}
impl TensorDescriptor {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }
    pub fn ggml_type(&self) -> u32 {
        self.ggml_type
    }
    pub fn relative_offset(&self) -> u64 {
        self.relative_offset
    }
    pub fn absolute_offset(&self) -> u64 {
        self.absolute_offset
    }
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
    pub fn row_bytes(&self) -> u64 {
        self.row_bytes
    }
    pub fn layout(&self) -> QuantLayout {
        self.layout
    }
}
#[derive(Clone, Debug)]
pub struct ModelManifest {
    header: ParsedMetadata,
    tensors: Vec<TensorDescriptor>,
    data_start: u64,
    file_len: u64,
}
impl ModelManifest {
    pub fn header(&self) -> &ParsedMetadata {
        &self.header
    }
    pub fn metadata(&self) -> &HashMap<String, GGUFValue> {
        &self.header.values
    }
    pub fn tensors(&self) -> &[TensorDescriptor] {
        &self.tensors
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
    pub fn data_start(&self) -> u64 {
        self.data_start
    }
    pub fn file_len(&self) -> u64 {
        self.file_len
    }
}
fn invalid(code: &str, detail: impl std::fmt::Display) -> Error {
    Error::InvalidMetadata(format!("{code}: {detail}"))
}
struct Decoder<R> {
    reader: R,
    position: u64,
    file_len: Option<u64>,
    allocated: u64,
    limits: ParserLimits,
}
impl<R: Read> Decoder<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let end = self
            .position
            .checked_add(N as u64)
            .ok_or_else(|| invalid("overflow", "reader position"))?;
        if self.file_len.is_some_and(|len| end > len) {
            return Err(invalid("truncated_gguf", "scalar crosses file end"));
        }
        let mut bytes = [0u8; N];
        self.reader.read_exact(&mut bytes)?;
        self.position = end;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn charge(&mut self, bytes: u64) -> Result<(), Error> {
        self.allocated = self
            .allocated
            .checked_add(bytes)
            .ok_or_else(|| invalid("overflow", "allocation budget"))?;
        if self.allocated > self.limits.allocation_bytes {
            return Err(invalid("allocation_limit", self.allocated));
        }
        Ok(())
    }
    fn count(&mut self, n: u64, max: u64, unit: u64) -> Result<usize, Error> {
        if n > max {
            return Err(invalid("count_limit", n));
        }
        self.charge(
            n.checked_mul(unit)
                .ok_or_else(|| invalid("overflow", "allocation count"))?,
        )?;
        usize::try_from(n).map_err(|_| invalid("platform_limit", n))
    }
    fn string(&mut self) -> Result<String, Error> {
        let n = self.u64()?;
        let len = self.count(n, self.limits.string_bytes, 1)?;
        let end = self
            .position
            .checked_add(n)
            .ok_or_else(|| invalid("overflow", "string extent"))?;
        if self.file_len.is_some_and(|size| end > size) {
            return Err(invalid("truncated_gguf", "string crosses file end"));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| invalid("allocation_failure", len))?;
        bytes.resize(len, 0);
        self.reader.read_exact(&mut bytes)?;
        self.position = end;
        String::from_utf8(bytes).map_err(|_| invalid("invalid_utf8", "metadata/tensor string"))
    }
    fn value(&mut self, kind: u32, depth: u32) -> Result<(GGUFValue, Option<u32>), Error> {
        if depth > self.limits.nesting_depth {
            return Err(invalid("nesting_limit", depth));
        }
        let value = match kind {
            0 => GGUFValue::U8(self.bytes::<1>()?[0]),
            1 => GGUFValue::I8(i8::from_le_bytes(self.bytes()?)),
            2 => GGUFValue::U16(u16::from_le_bytes(self.bytes()?)),
            3 => GGUFValue::I16(i16::from_le_bytes(self.bytes()?)),
            4 => GGUFValue::U32(self.u32()?),
            5 => GGUFValue::I32(i32::from_le_bytes(self.bytes()?)),
            6 => GGUFValue::F32(f32::from_le_bytes(self.bytes()?)),
            7 => match self.bytes::<1>()?[0] {
                0 => GGUFValue::Bool(false),
                1 => GGUFValue::Bool(true),
                n => return Err(invalid("invalid_boolean", n)),
            },
            8 => GGUFValue::String(self.string()?),
            9 => {
                let element = self.u32()?;
                let n = self.u64()?;
                if element > 12 {
                    return Err(invalid("unsupported_metadata_type", element));
                }
                let len = self.count(
                    n,
                    self.limits.array_elements,
                    std::mem::size_of::<GGUFValue>() as u64,
                )?;
                let mut values = Vec::new();
                values
                    .try_reserve_exact(len.min(4096))
                    .map_err(|_| invalid("allocation_failure", len))?;
                for _ in 0..len {
                    if values.len() == values.capacity() {
                        values
                            .try_reserve(1)
                            .map_err(|_| invalid("allocation_failure", "array growth"))?;
                    }
                    values.push(self.value(element, depth + 1)?.0);
                }
                return Ok((GGUFValue::Array(values), Some(element)));
            }
            10 => GGUFValue::U64(self.u64()?),
            11 => GGUFValue::I64(i64::from_le_bytes(self.bytes()?)),
            12 => GGUFValue::F64(f64::from_le_bytes(self.bytes()?)),
            n => return Err(invalid("unsupported_metadata_type", n)),
        };
        Ok((value, None))
    }
    fn parse_header(&mut self) -> Result<ParsedMetadata, Error> {
        if &self.bytes::<4>()? != b"GGUF" {
            return Err(invalid("invalid_magic", "expected little-endian GGUF"));
        }
        let version = self.u32()?;
        if !(2..=3).contains(&version) {
            return Err(invalid("unsupported_version", version));
        }
        let tensor_count = self.u64()?;
        if tensor_count > self.limits.tensors {
            return Err(invalid("tensor_count_limit", tensor_count));
        }
        let n = self.u64()?;
        if self
            .file_len
            .is_some_and(|len| n > len.saturating_sub(self.position) / 14)
        {
            return Err(invalid(
                "truncated_gguf",
                "metadata count exceeds remaining bytes",
            ));
        }
        let count = self.count(n, self.limits.metadata_entries, 128)?;
        let mut values = HashMap::new();
        let mut array_element_types = HashMap::new();
        values
            .try_reserve(count.min(1024))
            .map_err(|_| invalid("allocation_failure", count))?;
        for _ in 0..count {
            let key = self.string()?;
            if key.is_empty() || values.contains_key(&key) {
                return Err(invalid("invalid_or_duplicate_key", key));
            }
            let kind = self.u32()?;
            let (value, array_type) = self.value(kind, 0)?;
            if let Some(element) = array_type {
                self.charge(key.len() as u64 + 64)?;
                array_element_types.insert(key.clone(), element);
            }
            values
                .try_reserve(1)
                .map_err(|_| invalid("allocation_failure", "metadata entry"))?;
            values.insert(key, value);
        }
        let alignment_declared = values.contains_key("general.alignment");
        let alignment = match values.get("general.alignment") {
            None => 32,
            Some(GGUFValue::U32(n)) if n.is_power_of_two() => *n,
            _ => {
                return Err(invalid(
                    "invalid_alignment",
                    "expected nonzero power-of-two UINT32",
                ))
            }
        };
        Ok(ParsedMetadata {
            version,
            tensor_count,
            values,
            array_element_types,
            alignment,
            alignment_declared,
            metadata_end: self.position,
        })
    }
}
/// The legacy tokenizer's metadata-only Read API shares this exact lexer.
pub fn read_metadata<R: Read>(reader: R, limits: ParserLimits) -> Result<ParsedMetadata, Error> {
    Decoder {
        reader,
        position: 0,
        file_len: None,
        allocated: 0,
        limits,
    }
    .parse_header()
}
pub fn load_manifest(path: impl AsRef<Path>) -> Result<ModelManifest, Error> {
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    read_manifest(file, len, ParserLimits::default())
}
/// Validate the full declared tensor table against an exact file length.
/// This does not read or allocate weight payloads.
pub fn read_manifest<R: Read>(
    reader: R,
    file_len: u64,
    limits: ParserLimits,
) -> Result<ModelManifest, Error> {
    let mut decoder = Decoder {
        reader,
        position: 0,
        file_len: Some(file_len),
        allocated: 0,
        limits,
    };
    let header = decoder.parse_header()?;
    if header.tensor_count > file_len.saturating_sub(decoder.position) / 33 {
        return Err(invalid(
            "truncated_gguf",
            "tensor count exceeds remaining bytes",
        ));
    }
    let count = decoder.count(header.tensor_count, limits.tensors, 192)?;
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(count)
        .map_err(|_| invalid("allocation_failure", count))?;
    let mut names = HashSet::new();
    for _ in 0..count {
        let name = decoder.string()?;
        if name.is_empty() || name.len() > 63 {
            return Err(invalid("invalid_tensor_name", name));
        }
        decoder.charge(name.len() as u64 + 64)?;
        if !names.insert(name.clone()) {
            return Err(invalid("duplicate_tensor", name));
        }
        let nd = decoder.u32()?;
        if !(1..=4).contains(&nd) {
            return Err(invalid("invalid_tensor_rank", nd));
        }
        let mut dimensions = Vec::with_capacity(nd as usize);
        let mut elements = 1u64;
        for _ in 0..nd {
            let dimension = decoder.u64()?;
            if dimension == 0 {
                return Err(invalid("zero_tensor_dimension", &name));
            }
            elements = elements
                .checked_mul(dimension)
                .ok_or_else(|| invalid("overflow", &name))?;
            dimensions.push(dimension);
        }
        let ggml_type = decoder.u32()?;
        let layout = quant_layout(ggml_type)
            .ok_or_else(|| invalid("unsupported_quantization_layout", ggml_type))?;
        if dimensions[0].checked_rem(layout.elements_per_block) != Some(0) {
            return Err(invalid("partial_quantization_block", &name));
        }
        let row_bytes = (dimensions[0] / layout.elements_per_block)
            .checked_mul(layout.bytes_per_block)
            .ok_or_else(|| invalid("overflow", &name))?;
        let byte_len = (elements / layout.elements_per_block)
            .checked_mul(layout.bytes_per_block)
            .ok_or_else(|| invalid("overflow", &name))?;
        let relative_offset = decoder.u64()?;
        if relative_offset.checked_rem(u64::from(header.alignment)) != Some(0) {
            return Err(invalid("misaligned_tensor", &name));
        }
        tensors.push(TensorDescriptor {
            name,
            dimensions,
            ggml_type,
            relative_offset,
            absolute_offset: 0,
            byte_len,
            row_bytes,
            layout,
        });
    }
    let align = u64::from(header.alignment);
    let data_start = decoder
        .position
        .checked_add(align - 1)
        .ok_or_else(|| invalid("overflow", "data alignment"))?
        & !(align - 1);
    if count > 0 && data_start > file_len {
        return Err(invalid("truncated_gguf", "missing aligned data section"));
    }
    for tensor in &mut tensors {
        tensor.absolute_offset = data_start
            .checked_add(tensor.relative_offset)
            .ok_or_else(|| invalid("overflow", &tensor.name))?;
        let end = tensor
            .absolute_offset
            .checked_add(tensor.byte_len)
            .ok_or_else(|| invalid("overflow", &tensor.name))?;
        if end > file_len {
            return Err(invalid("tensor_out_of_bounds", &tensor.name));
        }
    }
    decoder.charge(
        header
            .tensor_count
            .checked_mul(24)
            .ok_or_else(|| invalid("overflow", "range index"))?,
    )?;
    let mut ranges: Vec<_> = tensors
        .iter()
        .map(|t| {
            (
                t.absolute_offset,
                t.absolute_offset + t.byte_len,
                t.name.as_str(),
            )
        })
        .collect();
    ranges.sort_unstable_by_key(|r| r.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(invalid(
                "overlapping_tensors",
                format!("{} / {}", pair[0].2, pair[1].2),
            ));
        }
    }
    Ok(ModelManifest {
        header,
        tensors,
        data_start,
        file_len,
    })
}
