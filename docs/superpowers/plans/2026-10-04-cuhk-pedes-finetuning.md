# CUHK-PEDES Fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 웹에서 설정하고 관찰할 수 있는 내장 CUHK-PEDES CLIP 전체 파인튜닝과 수동 후보 적용을 제공한다.

**Architecture:** API는 인증·검증·PostgreSQL 작업 상태를 담당한다. 별도 GPU 학습 supervisor가 메모리 admission과 단일 실행 소유권을 관리하고, 자식 프로세스가 학습·평가·export를 수행한다. 기존 immutable importer와 preparation/quality/preflight를 재사용하여 후보를 수동 적용한다.

**Tech Stack:** 기존 Python 3.12, FastAPI, SQLAlchemy/PostgreSQL, React/TypeScript; 학습 전용 환경에 기존 inference 환경과 같은 torch 2.7.1 CUDA 12.8, transformers 4.56.1, safetensors 0.6.2.

**Spec:** [승인된 설계](../specs/2026-10-04-cuhk-pedes-finetuning-design.md)

## Global Constraints

- 작업 브랜치: codex/cuhk-pedes-finetuning. 기존 사용자 수정과 다른 worktree를 보존한다.
- 제품 코드 및 테스트 작성: GPT Luna max. 기본 실행 제안은 하나의 Luna 구현 agent가 순서대로 작업하고 단계별 검토를 받는 방식이다.
- ViT-B/16 image/text 전체 학습, dataset read-only, 기존 API/worker model mount read-only.
- 추론 유지, 학습 한 건, 시작 직전 memory 재검사, 자동 대기·자동 batch 변경·자동 재시도 없음.
- 초기값과 범위는 승인된 spec 표를 그대로 적용한다. 유효 batch와 contrastive negative pool의 차이를 설명한다.
- 데이터셋 및 설정은 실행에 결속한다. epoch atomic checkpoint, interrupted 수동 resume, OOM 시 프로세스 종료.
- 학습 완료는 적용 승인이 아니다. 기존 GPU/제품 품질/전환 gate를 우회하지 않는다.
- git primary author 설정 유지. commit 전 실제 참여 agent의 공식 또는 설정된 co-author identity를 확인한다. 확인되지 않은 이메일을 만들어 쓰지 않는다.
- focused test RED→GREEN과 실제 GPU/browser 증거를 남긴다. 전체 장기 학습은 smoke 성공으로 대체하지 않는다.

## Review Focus

- 추정 직후 GPU 여유가 감소한 경우: worker의 최종 판정에서 거절하고 실행 작업을 남기지 않는다. Task 3/4.
- annotation 경로 traversal·symlink·split identity 중복: 검증 실패 후 학습 불가. Task 2.
- worker 재시작 시 기존 child가 남은 경우: child 종료 확인 전 새 실행 허용 금지. Task 4/5.
- checkpoint 저장 중 취소·디스크 부족: partial을 resume 대상으로 사용하지 않고 이전 완전한 checkpoint 유지. Task 5.
- 후보 게시 후 API catalog가 오래된 경우: process restart 없이 새 후보 조회, 적용 gate 계속 유지. Task 6/7.

## File map

새 `server/src/gods_watching/training/` 모듈은 settings/contracts와 dataset, memory, repository, supervisor, engine, checkpoints, evaluation, publishing의 경계를 갖는다. CPU API import가 torch를 요구하지 않도록 GPU imports는 engine/memory 실행 경계에 둔다. `training/models.py`의 ORM은 기존 storage Base에 등록하고 alembic/env.py의 metadata discovery에 포함한다.

새 학습 이미지/lock은 `training/pyproject.toml`, `training/uv.lock`, `deploy/Dockerfile.training`에 둔다. UI는 `web/src/features/training/`에 둔다. 아래 경로는 저장소 루트 기준이다.

### Task 1: 설정 계약과 durable 작업 저장

**Files:** Create `server/src/gods_watching/contracts/training.py`, `training/__init__.py`, `training/models.py`, `training/repository.py`; modify `server/alembic/env.py`; create `server/alembic/versions/0008_training_jobs.py`, `server/tests/test_training_contracts.py`, `server/tests/integration/test_training_jobs.py`.

**Interfaces:** `TrainingConfig`는 spec의 필드와 엄격한 범위, extra 금지, finite numeric 검증을 제공한다. `TrainingJob`은 UUID, request_id unique, immutable config/dataset/source fingerprint, phase, epoch/step, owner_generation, child identity, cancellation, attempts, best metric, checkpoint/candidate 및 bounded error를 저장한다. `TrainingRepository.create(session, request_id, config, dataset)`, `transition(session, job_id, expected_phase, next_phase)`는 transaction/CAS를 사용한다. 별도 singleton GPU 실행 slot이 active job 한 건을 보장한다.

- [ ] RED: `test_training_config_defaults_and_boundaries`에서 batch=1, NaN lr, unknown field 거절; `test_duplicate_request_is_idempotent`, `test_concurrent_active_job_is_rejected`, `test_terminal_phase_cannot_regress` 작성.
- [ ] `.venv/bin/pytest server/tests/test_training_contracts.py server/tests/integration/test_training_jobs.py -q`로 의도한 실패 확인. integration은 기존 DB fixture 규칙을 따른다.
- [ ] 위 계약·migration·repository를 구현하고 immutable input을 UPDATE로 변경할 수 없게 한다.
- [ ] 같은 tests GREEN 및 isolated DB upgrade 확인 후 검토·commit.

### Task 2: CUHK 등록과 identity-safe sampler

**Files:** Create `training/dataset.py`, `training/sampler.py`, `training/settings.py`, `server/tests/test_training_dataset.py`, `server/tests/test_training_sampler.py`.

**Interfaces:** `validate_cuhk(root: Path) -> DatasetManifest`는 파일 hash 및 split statistics와 안전하게 resolve된 sample 목록을 생성한다. `DatasetManifest`는 dataset_id, fingerprint, protocol, split counts, image/caption/identity metadata를 갖는다. `IdentityBatchSampler(samples, batch_size, seed, epoch)`는 distinct identity인 batch를 생성하고 caption을 seeded 선택한다. 전체 train image를 epoch별 shuffled buckets로 소모하고 구성 불가능한 remainder 수는 기록한다. accumulation은 negative pool을 합치지 않는다.

- [ ] RED: `test_dataset_rejects_escape_symlink_missing_image`, `test_identity_overlap_rejected`, `test_fingerprint_changes_with_image_bytes`, `test_actual_caption_count_is_reported`, `test_sampler_has_distinct_identity_and_reproducible_epochs`; 두 샘플의 person id가 같으면 같은 micro batch에 배치하지 않음을 assert.
- [ ] `.venv/bin/pytest server/tests/test_training_dataset.py server/tests/test_training_sampler.py -q` 실패 확인.
- [ ] 등록 경로는 환경설정 allowlisted root로 제한; imgs/reid_raw.json 전체 integrity 검사, finite counts, 최소 batch 조건과 캐시 fingerprint invalidation 구현. 증강 파일은 무시.
- [ ] tests GREEN, 실제 `/mnt/data/lab_datasets/CUHK-PEDES`의 split/image/caption 검사 결과 저장 후 검토·commit.

### Task 3: GPU calibration과 memory admission

**Files:** Create `training/memory.py`, `training/calibration.py`, `server/tests/test_training_memory.py`, `qa/training/calibrate_memory.py`.

**Interfaces:** `GpuSnapshot`는 uuid/total/free/observed_at을 갖는다. `MemoryEstimate`는 training_peak/reserve/required/profile identity를 갖는다. `estimate_memory(config, profile, gpu) -> MemoryEstimate`, `assess_admission(estimate, gpu) -> AdmissionResult`. profile key에는 GPU, source, torch/CUDA, processor, precision, checkpointing, batch가 포함된다. 지원 범위 밖의 profile은 fail closed.

- [ ] RED: `test_reserve_is_max_2gib_or_15percent`, `test_unknown_profile_rejected`, `test_stale_or_wrong_gpu_profile_rejected`, `test_free_bytes_change_reverses_admission`, `test_optimizer_state_included`.
- [ ] `.venv/bin/pytest server/tests/test_training_memory.py -q` 실패 확인.
- [ ] 실제 backward/AdamW optimizer step을 포함한 isolated calibration child를 구현. 추론 사용량을 관측하고 현재 여유 안에서만 profile 측정하며 OOM으로 범위를 탐색하지 않는다. 정확히 측정하거나 검증한 보수적 상한이 있는 설정만 허용한다.
- [ ] tests GREEN. RTX 5070 Ti에서 FP16/FP32와 checkpointing 대표 조합 peak 측정; 측정하지 못한 조합은 unsupported로 남겨 검토·commit.

### Task 4: 인증 API와 supervisor 실행 소유권

**Files:** Create `training/service.py`, `training/supervisor.py`, `training/app.py`, `api/training_routes.py`, `server/tests/test_training_routes.py`, `server/tests/test_training_supervisor.py`; modify `api/dependencies.py`, `api/application.py`, `api/production.py`.

**Interfaces:** `/api/training/datasets`, `/config`, `/preflight`, `/jobs` GET/POST; `/jobs/{id}` GET; `/cancel`, `/resume` POST; `/metrics`, `/logs` cursor GET. API service는 CPU-only. Supervisor가 저장된 request에 응답하여 slot 아래 GPU memory를 재검사하고 admission을 승인한 경우에만 job을 생성/child 실행한다. request refusal도 idempotent response로 저장하되 실행 job/history에는 넣지 않는다. worker 미응답은 503이며 숨은 자동 실행이 없도록 request 만료를 강제한다.

- [ ] RED: unauthenticated=401, invalid config=422, memory refusal=409 with required/free/reserve, worker unavailable=503, concurrent submit rejection, repeated request returns same result; orphan child가 살아 있으면 slot 재사용 거절.
- [ ] `.venv/bin/pytest server/tests/test_training_routes.py server/tests/test_training_supervisor.py -q` 실패 확인.
- [ ] DB request handshake, bounded timeout, lease/generation fencing과 child PID/start-time 확인을 구현. arbitrary command/path 입력 금지, log cursor와 응답 크기 제한.
- [ ] tests GREEN, 실제 DB 동시 요청/취소 경쟁 통과 후 검토·commit.

### Task 5: 학습 engine·checkpoint·취소와 resume

**Files:** Create `training/engine.py`, `training/checkpoints.py`, `training/metrics.py`, `training/runner.py`, `training/pyproject.toml`, `training/uv.lock`, `server/tests/test_training_checkpoints.py`, `server/tests/test_training_engine.py`.

**Interfaces:** `run_training(job_snapshot, paths, reporter, cancellation) -> TrainingResult`; `save_checkpoint_atomic(state, path)`, `load_checkpoint_verified(path, snapshot)`. CUDA engine imports는 전용 환경에서만 로드한다. recipe AdamW betas=(0.9,0.999), eps=1e-8, bidirectional CLIP loss, optimizer-step warmup/cosine, FP16 GradScaler/FP32, checkpointing, clipping.

- [ ] RED: tiny model weights change, accumulation partial group 정상 normalization/step, warmup optimizer-step 기준, nonfinite loss 실패, atomic save interrupted 시 이전 checkpoint 유지, cancellation/OOM cleanup, RNG/optimizer/scaler/sampler resume equivalence.
- [ ] CPU-compatible tests를 `.venv/bin/pytest server/tests/test_training_checkpoints.py -q`로, torch tests는 전용 training 환경으로 실행해 실패 확인.
- [ ] 학습/validation loop, epoch fsync+atomic rename, cancellation, finite checks, best checkpoint, patience를 구현. transient progress와 durable epoch history 구분. disk space는 checkpoint size 상한과 temporary copy 공간까지 검사.
- [ ] tests GREEN 및 실제 GPU tiny-data train/restart/resume에서 가중치 변경·continuity 확인 후 검토·commit.

### Task 6: 최종 평가와 immutable 후보 게시

**Files:** Create `training/evaluation.py`, `training/publishing.py`, `server/tests/test_training_evaluation.py`, `server/tests/test_training_publishing.py`, `qa/training/evaluate_product.py`; modify `model_selection/registry.py`, `model_selection/service.py`의 catalog refresh 경계.

**Interfaces:** `evaluate_retrieval(image_embeddings, text_embeddings, image_ids, text_ids) -> RetrievalScores`는 text query macro R@1/5/10, same-person relevance, deterministic tie breaking을 사용한다. `publish_candidate(job, best_checkpoint, report, assets_root) -> CandidateIdentity`는 save_pretrained/export 및 existing importer를 호출한다. product evaluator는 기존 quality policy/evidence 계약을 소비·생성한다.

- [ ] RED: multi-relevant gallery metric과 exact ties, validation best와 test 분리, baseline cache binding mismatch, incomplete export refusal, retry publication idempotency, catalog refresh 새 candidate 등장, CUHK success만으로 apply 차단.
- [ ] `.venv/bin/pytest server/tests/test_training_evaluation.py server/tests/test_training_publishing.py server/tests/test_model_quality_gate.py -q` 실패 확인.
- [ ] 최종 test report 및 weights hash/provenance를 existing importer schema에 정확히 맞춰 생성. API/worker registry refresh는 active identity를 보존하며 새 후보를 반영한다. baseline/candidate product evidence는 같은 사례와 code fingerprint로 평가한다.
- [ ] tests GREEN, 실제 exported CLIP strict load/import, product cases 부재 시 명확한 차단 확인 후 검토·commit.

### Task 7: 학습 UI와 동기화된 입력

**Files:** Create `web/src/features/training/TrainingScreen.tsx`, `TrainingForm.tsx`, `NumericParameter.tsx`, `TrainingHistory.tsx`, `TrainingDetail.tsx`, `MetricChart.tsx`, `training.css`, `trainingModel.ts`, `trainingModel.test.ts`; modify `web/src/app/client.ts`, `clientTypes.ts`, `App.tsx`, `layout/AppShell.tsx`; create `web/e2e/training.spec.ts`.

**Interfaces:** typed training client은 Task 4 API와 동일 response names를 쓴다. `NumericParameter`는 controlled value/onChange, min/max/step/logScale을 받아 slider/number 공통 값을 관리한다. request generations와 abort를 사용해 늦은 응답이 최신 설정을 덮지 못하게 한다.

- [ ] RED: slider/number bidirectional sync, lr logarithmic mapping, invalid/incomplete numeric text, config immutability; browser에서 부족 팝업 required/free/reserve, duplicate click, stale preflight response, history refresh, cancel/resume, candidate link와 apply blocked 상태 확인.
- [ ] `web/node_modules/.bin/vitest run` 및 기존 harness 방식의 training spec으로 실패 확인.
- [ ] basic/advanced form, dataset validation 상태, live estimate, config summary, history/detail, SVG loss/Recall chart, bounded logs와 최종 평가 UI 구현. popup focus/keyboard/aria와 기존 375/768/1280/1440 responsive 폭 확인.
- [ ] tests GREEN, `web/node_modules/.bin/tsc -b --pretty false`, `web/node_modules/.bin/biome check .`(cwd web), Vite build 및 screenshot 검토 후 commit.

### Task 8: 컨테이너 배포와 실제 통합 검증

**Files:** Create `deploy/Dockerfile.training`, `qa/training/run_smoke.py`; modify `compose.yaml`, `.env.example`(현재 파일 존재 여부 확인 후), `server/src/gods_watching/lifecycle.py`의 설정 준비 경계, `README.md`, `docs/architecture.md`, `docs/remaining-work.md`, 기존 external import spec.

**Interfaces:** supervisor training image에 GPU allocation; dataset read-only, runs writable volume, assets writable publishing mount를 명시한다. API와 inference worker에는 runs 직접 접근 대신 DB API를 사용한다. training root 환경설정은 운영자가 설정; dataset 미설정은 service crash가 아니라 unavailable 기능 상태로 표시한다. 기존 host-network 방식과 포트 노출 범위를 유지한다.

- [ ] RED: compose contract tests에서 inference model mounts read-only, dataset read-only, training runs persistence, absence of docker socket/API GPU dependency assert. dataset 없을 때 학습 unavailable 및 기존 서비스 정상 확인.
- [ ] contract tests 실패 확인 후 Dockerfile/compose/settings/docs 구현.
- [ ] `docker compose config --quiet`, Dockerfile build/check, migration upgrade, standard image source/runtime 일치 확인.
- [ ] 실제 CUHK subset으로 웹 설정→admission→train→validation→cancel/resume→test→export/import→candidate 조회 실행. 실제 camera inference/search 유지, memory 부족 거절 및 처리 후 GPU 메모리 해제 관측. runtime dataset validation은 전체 원본, smoke 학습만 subset을 사용하고 production UI에 임의 subset 선택 기능을 추가하지 않는다.
- [ ] 실패가 있으면 원인 수정 후 영향 있는 checks만 재실행. 전체 장기 학습 미실행과 실제 제품 gate 미통과는 보고서에 구분하고 검토·commit.

### Task 9: 최종 검토와 전달

**Files:** 전체 diff 및 `output/training/` 실행 증거(민감 정보 제외).

- [ ] 필요한 focused tests, server regression suite, frontend build/lint/typecheck 및 browser tests를 실행하고 기존 실패와 신규 실패를 구분한다.
- [ ] spec coverage를 Q1~Q12와 대조하여 메모리 profile unsupported 처리, resume, manual apply gate, 실제 UI 증거를 확인한다.
- [ ] 독립 reviewer에게 branch diff와 exact source/evidence identity를 제공하고 실제 사용자 영향이 있는 findings를 수정한다.
- [ ] `git diff --check`, status, branch, configured author/coauthors 확인. 사용자 승인 없는 push/PR/merge를 실행 범위에 추가하지 않는다.
- [ ] 전달 시 변경·실제 검증·미실행 장기 학습/제품 품질 gate·지원되는 calibration 범위를 명시한다.

## Self-review

Spec의 dataset/recipe/UI/memory/admission/jobs/checkpoints/evaluation/export/application/deployment/validation을 Task 1~9에 매핑했다. Review Focus 다섯 조건은 해당 test 단계에 포함했다. CPU API와 CUDA worker dependencies를 분리하며 프로파일 없는 설정의 실행 차단과 candidate 등록 후 catalog refresh를 명시했다. sampler 및 고정 AdamW recipe는 구현 계획의 구체 결정이며 이 계획 검토에 포함된다.
