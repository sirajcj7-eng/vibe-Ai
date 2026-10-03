## Model family direction (EVY)
- EVY 50M is a general conversational prototype. Its job is to
  prove we can build, train, evaluate, save, load and run a
  language model from random initialization.
- No routing, mixture-of-experts, agents or tool ecosystems at 50M.
- Specialization (Code, Math, Physics, Research, General) begins
  around 100M. Whether variants use separate weights, a shared
  base, or adapters is a research question for Phase 8-9.
- Domain routing and retrieval (EVY Research) come after the base
  models are validated.
- No automatic scaling: a model is scaled only after the previous
  one is evaluated and its problems are understood.
- Every run records: parameter count, config, tokenizer version,
  dataset version, training tokens, context length, training
  config, compute, duration, validation loss, eval results,
  inference speed, memory, qualitative examples, failure modes.