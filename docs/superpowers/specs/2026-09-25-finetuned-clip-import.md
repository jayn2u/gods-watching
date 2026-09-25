# Fine-tuned CLIP package import

## Goal and scope

Train a complete image/text CLIP ViT-B/16 pair outside Gods Watching with CUHK-PEDES. Import only the finished package and its evaluation record into this product. CUHK-PEDES images, captions, identities, and gallery are not product data. The product continues to search its retained person appearances from RTSP camera crops with English descriptions and to offer Find similar.

Training scripts, training compute, browser checkpoint upload, Korean search, cross-camera identity, and person ReID are outside this feature.

## Package contract

- Import is an offline server CLI operation. It reads a local directory, never downloads at runtime, and creates an immutable package under the shared model asset cache. Registration and activation are separate operations.
- The first supported custom architecture is a Hugging Face `CLIPModel` ViT-B/16 with its matching `CLIPProcessor`, tokenizer, image preprocessing configuration, both trained encoders, 512-dimensional image/text embeddings, and `safetensors` weights. Custom Python modules and `trust_remote_code` are forbidden.
- The package carries a unique model ID outside the built-in catalog, immutable revision derived from content, display name, base checkpoint identity, file manifest with SHA-256 and sizes, and CUHK-PEDES evaluation provenance. Reimporting identical bytes is idempotent; reusing an ID for different bytes fails. No in-place overwrite.
- Files are copied and verified into a temporary directory, then atomically published. Import failure leaves no selectable partial package. API, worker, CLI, and Triton must resolve the same catalog after restart. Triton retains read-only access to the asset cache.
- The default and immediately previous active packages remain available for rollback. An active, transition-referenced, default, or previous package cannot be removed.

## Quality and operational gates

- CUHK-PEDES uses a fixed identity-disjoint train/validation/test protocol in the external training workflow. The submitted report binds dataset split, source checkpoint ID and revision, candidate `model.safetensors` SHA-256, evaluation code revision, metric definition, and baseline/candidate scores. The deployment-owned policy pins the source revision and exact metric definition. The file manifest binds report bytes and weights into the immutable package revision. Candidate must improve over baseline on the same held-out test protocol.
- Product evaluation uses the existing planned `qa/retrieval-cases.json` evidence contract: real crops from at least two scenes, at least 40 appearances/distractors, 20 English text queries, and 20 held-out image queries with complete relevance sets fixed before results are viewed. Baseline and candidate run against the same cases. Both text and image macro Recall@5 must be at least 0.8 and improve over the baseline. CUHK-PEDES benchmark results cannot substitute for this product gate.
- Registration checks structure, file hashes, image/text dimensions, finite unit-normalized outputs, GPU coexistence with the detector, and the model's pinned processor. The candidate is listed but cannot be applied until both quality records and GPU proof pass. Missing proof fails closed.
- Apply remains a manual global switch. Search and person analysis pause while retained crops are re-embedded. The system counts retained crops and measures target-model throughput on the deployment GPU before queueing. Full-path rehearsal proof must match current retained appearance IDs, crop keys and bytes, and the installed transition source digest. Corpus or runtime changes require a fresh rehearsal; unknown bindings fail closed. It rejects apply if there is no measured rate or estimated transition time exceeds 15 minutes. It shows the estimate, affected crop count, and expected missing/corrupt crops before confirmation.
- Existing staged vectors, model identity checks, atomic activation, durable progress, missing-crop skip reporting, and rollback behavior are retained. Model spaces are never mixed, including equal-dimensional spaces. Applying the previous model uses the same preflight and switch path.
- The new model must also meet the existing four-camera performance gate: average accepted detector rate of at least 4.8 fps per camera and first-searchable latency p95 at most 5 seconds on the target GPU, with the existing 15-minute load procedure.

## Product presentation

The Cameras model selector displays imported model name, immutable revision, preparation and quality status, and an actionable reason when application is unavailable. The confirmation shows the preflight estimate and makes the analysis/search pause explicit. Transition progress and skipped crop reasons remain durable after refresh. The UI does not accept model files or CUHK-PEDES data.

## Delivery order

1. Define and validate the offline package and immutable catalog.
2. Make all processes load that catalog and verify GPU serving.
3. Produce CUHK and real-crop quality evidence and gate activation.
4. Add measured 15-minute preflight to the current transition path.
5. Present model eligibility and preflight in the selector; run end-to-end recovery and performance checks.
