# Changelog

## [1.0.0] - 2026-02-10

### Changed
- Restructured into institutional-grade monorepo
- Python code moved to installable `quantlaxmi/` package
- Rust workspace integrated at `rust/`
- All imports updated to `quantlaxmi.*` namespace

### Added
- `pyproject.toml` with full dependency specification
- `Makefile` for common development tasks
- `.pre-commit-config.yaml` for code quality hooks
- `docker-compose.yml` for containerized development
- `.env.example` template (secrets removed from tracking)
- Documentation hub (`docs/`)

### Security
- `.env` removed from git tracking
- `.gitignore` hardened
