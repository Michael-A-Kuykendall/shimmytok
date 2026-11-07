# GGUF parser security — ShimmyTok 0.8.5

The public `load_metadata_from_reader` path and shared manifest parser enforce
`ParserLimits` before allocating or reading length-directed metadata payloads.
Airframe and ShimmyTok consume this checked interpretation instead of competing
GGUF layouts.

## Default limits

| Resource | Default limit |
|---|---:|
| Metadata entries | 1,000,000 |
| Tensors | 1,000,000 |
| One string | 1 MiB |
| Array elements | 10,000,000 |
| Aggregate allocation budget | 256 MiB |
| Metadata nesting | 8 |

The full shared manifest path also validates declared alignment, dimensions,
quantization layouts, offsets and file bounds. The tokenizer metadata-only
reader is not a complete tensor/execution-graph validation entry point. Checked arithmetic and fallible reservation reject
overflow/invalid lengths with errors. Limits are defensive bounds, not a proof
that every accepted GGUF or tokenizer operation is secure.

## Permanent regression

`tests/gguf_untrusted_lengths.rs` invokes the actual public reader. A crafted
array count of 5,000,000,000 must fail before reading payload bytes for the
applicable element types. Separate cases cover budget exhaustion below the
element limit, integer overflow, unsupported types, truncated payloads and a
valid 131,072-token bounded vocabulary.

```bash
cargo test --locked --test gguf_untrusted_lengths -- --test-threads=1
```

The count-guard and budget-guard negative controls were independently disabled
in isolated bounded fixtures; each regression rejected its intended defect
without attempting huge allocation. The 0.8.5 release brings the shared checked parser into the coordinated public
candidate. The regression addition itself did not alter that already validated
private parser; do not assume an older published patch has identical checks. Do not remove limits to load an untrusted file.

## Encode bounds and remaining scope

The BPE and SentencePiece encode paths use the 10 MiB input and 1,000,000-output
token bounds. Do not generalize those two path guards into universal allocation
immunity across every dispatcher implementation. Downstream applications must
set their own input, concurrency and memory policies and handle returned errors.

Parser acceptance does not certify model mathematics, GPU operators, a tokenizer
against every llama.cpp revision, or an operating system. Missing fixtures and
unexecuted modes stay NOT_RUN. Report suspected vulnerabilities privately using
the [security policy](../SECURITY.md); include the exact file hash and failing
reader call. License and attribution remain in [MIT](../LICENSE-MIT),
[Apache-2.0](../LICENSE-APACHE) and [authors](../AUTHORS.md).
