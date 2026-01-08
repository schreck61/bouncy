# Contributing to Bouncy

Thank you for your interest in contributing to Bouncy! This document provides guidelines and best practices for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Guidelines](#development-guidelines)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive environment for everyone.

## Developer Certificate of Origin (DCO)

This project uses the Developer Certificate of Origin (DCO) to ensure that contributors have the right to submit their contributions. By submitting a contribution, you agree to the following:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### How to Sign Off

You must sign off on every commit to certify your agreement with the DCO. Add a `Signed-off-by` line to your commit messages:

```
This is my commit message

Signed-off-by: Your Name <your.email@example.com>
```

You can do this automatically by using the `-s` flag when committing:

```bash
git commit -s -m "Your commit message"
```

**All commits must include a DCO sign-off.** Pull requests with unsigned commits will not be accepted.

## Getting Started

### Prerequisites

- Rust 1.70 or later
- A GPU with Vulkan, Metal, or DX12 support
- Git

### Setting Up Your Development Environment

1. Fork the repository on GitHub

2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/bouncy.git
   cd bouncy
   ```

3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/schreck61/bouncy.git
   ```

4. Build the project:
   ```bash
   cargo build
   ```

5. Run the tests (when available):
   ```bash
   cargo test
   ```

## How to Contribute

### Reporting Bugs

Before submitting a bug report:

1. Check existing issues to avoid duplicates
2. Collect relevant information:
   - Operating system and version
   - GPU model and driver version
   - Rust version (`rustc --version`)
   - Steps to reproduce the issue
   - Expected vs. actual behavior

Create a new issue with a clear title and detailed description.

### Suggesting Features

Feature suggestions are welcome! Please open an issue with:

- A clear description of the feature
- The problem it solves or value it adds
- Any implementation ideas you have
- Whether you're willing to implement it yourself

### Submitting Code

1. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the [Development Guidelines](#development-guidelines)

3. Commit your changes with DCO sign-off:
   ```bash
   git commit -s -m "Add feature: description"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a Pull Request against the `main` branch

## Development Guidelines

### Code Quality

- Write clear, self-documenting code
- Add doc comments for public functions and complex logic
- Keep functions focused and reasonably sized
- Handle errors appropriately using `Result` and `expect()` with descriptive messages

### Testing

- Test your changes manually across different scenarios
- Verify performance isn't negatively impacted (check FPS output)
- Test both `--spawn-at-collision` and default modes if relevant

### Safety

- Avoid `unsafe` code unless absolutely necessary
- Do not introduce security vulnerabilities
- Be mindful of resource usage (memory, GPU)

### Compatibility

- Ensure changes work on macOS (primary development platform)
- Consider cross-platform implications for Windows and Linux
- Maintain compatibility with the minimum supported Rust version

## Pull Request Process

1. **Title**: Use a clear, descriptive title summarizing the change

2. **Description**: Include:
   - What the change does
   - Why it's needed
   - How it was tested
   - Any breaking changes or migration notes

3. **DCO**: Ensure all commits are signed off

4. **Review**: Address any feedback from reviewers promptly

5. **Merge**: Once approved, your PR will be merged by a maintainer

### PR Checklist

- [ ] Code compiles without warnings (`cargo build`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] Code passes lints (`cargo clippy`)
- [ ] All commits are signed off (DCO)
- [ ] Documentation is updated if needed
- [ ] Changes are tested manually

## Style Guide

### Rust Style

- Follow standard Rust conventions and idioms
- Use `cargo fmt` to format code
- Use `cargo clippy` to catch common issues
- Prefer explicit types when they aid readability

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Fix bug" not "Fixes bug")
- Keep the first line under 72 characters
- Reference issues when applicable (e.g., "Fixes #123")

Example:
```
Add spatial grid for collision detection

Implement a simple spatial hashing grid to improve collision
detection performance for large particle counts. This reduces
complexity from O(n²) to O(n) for uniformly distributed particles.

Fixes #42

Signed-off-by: Your Name <your.email@example.com>
```

### Documentation

- Use `///` for doc comments on public items
- Keep comments current with code changes
- Explain *why*, not just *what*

## Questions?

If you have questions about contributing, feel free to open an issue for discussion.

Thank you for contributing to Bouncy!
