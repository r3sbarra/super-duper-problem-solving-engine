# Contributing to Super-Duper-Problem-Solving-Engine

Thank you for your interest in contributing to the **Super-Duper-Problem-Solving-Engine**!

This project provides an autonomous neurosymbolic reasoning and scientific problem-solving engine unifying classical inquiry paradigms with modern continuous-vector reasoning architectures.

---

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/r3sbarra/super-duper-problem-solving-engine.git
   cd super-duper-problem-solving-engine
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package in editable mode with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

---

## Testing & Verification

Before submitting any code changes, ensure the entire test suite passes:

```bash
pytest -v tests/
```

To run tests with code coverage:
```bash
pytest --cov=super_solver --cov-report=term-missing tests/
```

To check code style and linting:
```bash
ruff check src/ tests/
```

---

## Architectural Principles

When extending or modifying the engine:
1. **Zero-Daemon Local Operations:** The core reasoning substrate and embedding projections should remain fast, local, and deterministic without mandatory external server daemons.
2. **Type Safety:** Use type annotations and Pydantic models for structured state definitions.
3. **Formal Refutation:** Ground exploratory trajectories with refutation safeguards (DPLL, SMT, or boundary checks).
4. **Clean Abstractions:** Keep cognitive frameworks modular and loosely coupled.

---

## Submitting Pull Requests

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-enhancement
   ```
2. Write tests covering new functionality in `tests/`.
3. Verify that all existing and new tests pass.
4. Open a pull request against `main` with a clear description of the problem solved and the architectural decisions made.

---

## Code of Conduct

Please maintain a collaborative, respectful, and evidence-driven environment. We follow the standard Contributor Covenant guidelines.
