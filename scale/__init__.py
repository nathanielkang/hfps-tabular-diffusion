"""
scale/ - Large-scale extension for TabOversample-HFPS.

Adds, without changing the released 27-column DDPM core:
  - dictionary-driven variable schema (arbitrary column count)
  - Parquet streaming / chunked reading with optional training subsample
  - memory-safe batched generation and decoding
  - a public generate(n_syn, seed, numeric_vars, categorical_vars, ...) entry point
  - a gate-stepwise benchmark harness for the n_syn ladder

The generative principle is unchanged: encode -> DDPM -> decode.
"""