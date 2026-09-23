# Global CLIP model selector

Approved through the grilling interview on 2026-09-22.

The settings screen selects one global retrieval model: OpenAI CLIP ViT-B/32,
ViT-B/16 (default), or ViT-L/14. Models are downloaded and validated during
deployment preparation; applying a model never downloads weights. Unprepared
models cannot be applied. A registry describes checkpoint identity, immutable
revision, embedding dimension, processor, and runtime adapter, leaving a seam
for future trained model packages. Custom package registration UI is deferred.

Switching temporarily stops person analysis and search. Inform the operator
about the analysis gap before applying, and show durable progress and outcome.
Re-embed all retained, non-deleted person crops. Skip missing or corrupt images
and report counts and reasons. Other failures (inference, database, disk) fail
the transition and restore the old model and data. Preserve old vectors until
new data is completely prepared; activation must be atomic. Interrupted jobs
must recover safely after process restart. Different model spaces must never
be compared, including equal-dimensional models and image-similarity search.

Only one model is active in GPU memory; old weights remain available on disk
for rollback. Runtime loading must support 512- and 768-dimensional vectors.
Keep existing authentication and existing retention dates and deletion rules.
Do not record video or add deferred ingestion/replay. Do not add multilingual
search or model benchmarking product features.

The observed development GPU is RTX 5070 Ti, 16303 MiB total memory. Validate
each CLIP model together with the detector on the actual GPU where feasible.
Report verification limitations honestly. Do not mutate existing deployment
data or stop running services merely to test the feature; use isolated checks.
