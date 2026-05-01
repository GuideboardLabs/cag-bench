# Contributing to CAG Bench

Thanks for considering contributing.

By contributing to this project, you agree that your contributions will be licensed under the same license as the project:

```text
AGPL-3.0-or-later
```

## Attribution

Please add yourself to `AUTHORS.md` when making a meaningful contribution.

## Contribution guidelines

- Keep benchmark changes auditable.
- Prefer transparent metrics over opaque scoring.
- Keep raw outputs inspectable.
- Do not add app-specific routing, product-specific guardrails, or hidden task hints.
- If adding a new metric, document how it is computed.
- If changing task data, explain why the change improves benchmark validity.

## Pull request checklist

Before opening a pull request, please verify:

- Dry run completes.
- Existing CLI behavior still works or the README explains the change.
- New graphs or metrics are documented.
- Raw JSONL output remains inspectable.
- License and attribution notices are preserved.
