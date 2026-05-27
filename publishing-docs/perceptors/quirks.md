# Perceptor Quirks

- Pass the perceptor **class**, not an instance, to `Perceptor(...)`.
- Some sims return list/array observations; normalize to dict inside `compute(...)` when needed.
- Stateful perceptors must manage reset behavior explicitly.
