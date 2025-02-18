build:
  @uv run maturin build -r

dev:
  @uv run maturin develop --uv

test: build
  # @uv run pytest tests/test_append.py --size 100_000
  # @uv run pytest tests/test_append.py --size 1_000_000
  # @uv run pytest tests/test_append.py --size 10_000_000
  # @uv run pytest tests/test_append.py --size 100_000_000

  @uv run pytest tests/test_cdf.py --size 100_000
  @uv run pytest tests/test_cdf.py --size 1_000_000
  @uv run pytest tests/test_cdf.py --size 10_000_000
  @uv run pytest tests/test_cdf.py --size 100_000_000
