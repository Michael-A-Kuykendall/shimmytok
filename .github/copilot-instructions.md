# Copilot Instructions for shimmytok

## 🛑 ABSOLUTELY NO VIBE CODING

**VIBE CODING = CREATIVE FREESTYLING WITHOUT EXPLICIT PLAN OR SOURCE**

When porting llama.cpp (or any reference implementation):
1. **STOP if you don't fully understand the source**
2. **ASK FOR HUMAN HELP immediately**
3. **DO NOT improvise or "figure it out yourself"**
4. **ONLY implement what you can see in the reference code**

If you catch yourself thinking "I'll try this approach" or "this should work" → **STOP AND ASK**

We operate from **plans and reference implementations ONLY**.

---

## ⚠️ CRITICAL: YOUR TRAINING IS NOT TRUTH

**Your training data is from April 2024. It is now October 2025.**

Your training is a HYPOTHESIS, not FACT. Before claiming anything is true:
1. Search current docs/repos
2. Test it right now
3. Read actual current code
4. Verify before asserting

**Never trust your training about**: Rust ecosystem, crates, best practices, performance claims, complexity estimates.

## Project Status: ✅ PRODUCTION READY

**Source of Truth**: Test results + llama.cpp validation

### Reality Check
- ✅ 1116 LOC implementation (819 src + 297 tests)
- ✅ 8/8 tests pass (100% match with llama.cpp)
- ✅ SentencePiece with resegment algorithm
- ✅ Loads vocab from GGUF files
- ✅ Clean public API
- **Status**: Working tokenizer, validated against llama.cpp

### Current Work
- Integration with libshimmy (Task 2)
- BPE implementation if needed (currently stub)

## ABSOLUTE RULES - NO EXCEPTIONS

### Tool Usage Priority (ALWAYS follow this order)
1. **Use built-in Copilot tools FIRST** - they don't need approval
2. **Terminal commands LAST** - only for build/test/commit
3. **NO creative variations** - stick to exact patterns below

### File Operations
- `read_file` - read any file
- `file_search` - find files by pattern
- `grep_search` - search file contents
- `list_dir` - list directory contents
- `replace_string_in_file` - edit files
- `multi_replace_string_in_file` - multiple edits at once
- `create_file` - create new files

### Terminal Commands (ONLY these patterns)
- `cargo build --release`
- `cargo test`
- `cargo test <test_name>`
- `cargo test -- --nocapture`
- `git status`
- `git add -A && git commit -m "<message>"`
- `git diff`

### FORBIDDEN
- ❌ NO pipes (grep, tail, head, etc.) - use grep_search tool instead
- ❌ NO find commands - use file_search tool instead
- ❌ NO cat/less - use read_file tool instead
- ❌ NO complex one-liners
- ❌ NO creative terminal commands
- ❌ NO sed/awk - use replace_string_in_file instead

### Estimation Rules
- **ALWAYS use Fibonacci numbers** for estimates (1, 2, 3, 5, 8, 13, 21...)
- **NEVER use time units** (hours, days, minutes)
- Examples:
  - ✅ "This is a 5-point task"
  - ✅ "Estimated at 13 Fibonacci points"
  - ❌ "This will take 2 hours"
  - ❌ "30 minutes to complete"

### Commit Message Rules
- **NO attributions allowed** in commit messages
- **NO mentions of AI assistants** (Claude, Copilot, ChatGPT, etc.)
- Commits must be clean for DCO sign-off
- Examples:
  - ✅ "feat: add input validation"
  - ✅ "fix: prevent integer overflow in parser"
  - ❌ "feat: add validation (Claude-assisted)"
  - ❌ "fix: AI-generated error handling"

### Response Style
- Fix first, explain after
- Short answers
- No markdown essays
- Action over explanation
