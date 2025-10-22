# Security Policy

## Supported Versions

We actively support the following versions of shimmytok with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of shimmytok seriously. If you discover a security vulnerability, please follow these guidelines:

### 🔒 Private Disclosure Process

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security issues privately using one of these methods:

1. **GitHub Security Advisories (Preferred)**
   - Go to the [Security tab](https://github.com/Michael-A-Kuykendall/shimmytok/security) of this repository
   - Click "Report a vulnerability"
   - Fill out the advisory form with details

2. **Direct Email**
   - Send details to: michaelallenkuykendall@gmail.com
   - Include "SECURITY: shimmytok" in the subject line

### 📋 What to Include

Please provide the following information in your report:

- **Description**: Clear description of the vulnerability
- **Impact**: What could an attacker accomplish?
- **Reproduction**: Step-by-step instructions to reproduce the issue
- **Environment**:
  - shimmytok version
  - Operating system
  - Rust version
  - GGUF model used (if applicable)
- **Proof of Concept**: Code, GGUF files, or logs demonstrating the issue
- **Suggested Fix**: If you have ideas for remediation

### 🕒 Response Timeline

We aim to respond to security reports according to the following timeline:

- **Initial Response**: Within 48 hours of report
- **Triage**: Within 7 days - confirm/deny vulnerability
- **Resolution**: Within 30 days for critical issues, 90 days for others
- **Disclosure**: Public disclosure after fix is released and users have time to update

### 🛡️ Vulnerability Severity Guidelines

We use the following criteria to classify vulnerabilities:

#### Critical
- Remote code execution via malicious GGUF file
- Memory corruption leading to arbitrary code execution
- Unsafe access patterns causing undefined behavior

#### High
- Denial of service via crafted GGUF input
- Memory exhaustion attacks
- Integer overflow in size calculations
- Unsafe pointer dereference

#### Medium
- Information disclosure (e.g., out-of-bounds reads)
- Panic in safe Rust code
- Resource leaks

#### Low
- Issues requiring local access
- Minor information leaks
- Performance degradation attacks

### 🎁 Recognition

We believe in recognizing security researchers who help keep shimmytok secure:

- **Hall of Fame**: Public recognition in our security acknowledgments
- **CVE Assignment**: For qualifying vulnerabilities
- **Early Access**: Beta access to new features
- **Acknowledgment**: Credit in release notes

*Note: We currently do not offer monetary bug bounties, but we deeply appreciate responsible disclosure.*

### 🚨 Emergency Contact

For critical vulnerabilities that are being actively exploited:

- **Email**: michaelallenkuykendall@gmail.com
- **Subject**: "URGENT SECURITY: shimmytok - [Brief Description]"
- **Response**: Within 12 hours

## Security Best Practices

### For Users

1. **Keep Updated**: Always use the latest supported version
2. **Model Security**:
   - Only use GGUF models from trusted sources
   - Verify checksums of downloaded models
   - Be cautious with user-uploaded models
3. **Input Validation**:
   - Validate text input length before tokenization
   - Use the provided MAX_INPUT_SIZE constant
   - Handle tokenization errors gracefully

### For Developers

1. **Dependencies**:
   - Regularly audit and update dependencies
   - Use `cargo audit` to check for known vulnerabilities
   - Minimize dependency count (currently only thiserror + regex)
2. **Input Validation**:
   - Validate GGUF file structure before parsing
   - Check array bounds and sizes
   - Reject malformed metadata
3. **Memory Safety**:
   - Avoid unsafe code unless absolutely necessary
   - Document any unsafe blocks with safety proofs
   - Test edge cases (empty input, max size, Unicode)

## Security Features

shimmytok includes several built-in security features:

- **Memory Safety**: Built with Rust for memory-safe execution
- **Input Size Limits**: MAX_INPUT_SIZE prevents memory exhaustion
- **Token Count Limits**: MAX_OUTPUT_TOKENS prevents unbounded allocation
- **GGUF Validation**: Strict parsing with error handling
- **No Unsafe Code**: Pure safe Rust implementation
- **Minimal Dependencies**: Only 2 direct dependencies (thiserror, regex)

## Scope

This security policy covers:

- **In Scope**:
  - shimmytok library code
  - GGUF parsing and validation
  - Tokenization algorithms (SentencePiece, BPE)
  - Memory management and allocation
  - Error handling

- **Out of Scope**:
  - Third-party GGUF models (user responsibility)
  - Issues in Rust standard library
  - Platform-specific OS issues
  - Applications using shimmytok (downstream responsibility)

## Known Limitations

- **GGUF Format**: Only supports v2 and v3. Older/newer versions may not parse correctly.
- **Input Size**: Limited to 10MB (MAX_INPUT_SIZE). Larger inputs are rejected.
- **Token Count**: Limited to 1M tokens (MAX_OUTPUT_TOKENS). Prevents unbounded allocation.
- **Malicious Models**: While we validate structure, we cannot detect all malicious payloads in GGUF files.

## Legal

- We will not pursue legal action against security researchers who follow this policy
- We ask that researchers not access, modify, or delete data without explicit permission
- Please do not perform testing on production systems without prior authorization

## Contact

For non-security related issues, please use:
- GitHub Issues: https://github.com/Michael-A-Kuykendall/shimmytok/issues
- GitHub Discussions: https://github.com/Michael-A-Kuykendall/shimmytok/discussions

---

*This security policy is effective as of October 22, 2025 and may be updated periodically.*
