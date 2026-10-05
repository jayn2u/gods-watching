# CUHK-PEDES CLIP 파인튜닝 전달 및 검증 보고서

이 문서는 승인된 설계의 Q1–Q12 범위를 최종 구현·실행 증거와 대조한다. 구현은 학습 후보 생성까지 검증했다. 전체 장기 학습과 제품 crop 품질 gate는 별도이며, 최종 server 회귀 전체는 녹색이 아니다.

## Q1–Q12 범위와 판정

| 항목 | 승인된 요구사항 | 구현 및 검증 |
|---|---|---|
| Q1 | 제품의 기존 CLIP 선택 경로에 연결되는 내장 ViT-B/16 파인튜닝 | 인증된 Training 화면, API, 별도 GPU supervisor·runner를 추가했다. API 프로세스는 CUDA를 직접 사용하지 않는다. |
| Q2 | CUHK-PEDES만 등록 경로에서 사용하고 업로드/임의 호스트 경로 입력을 막음 | 운영자가 설정한 dataset root를 API와 trainer에 read-only mount한다. UI 요청은 경로를 보내지 않는다. 학습 이력·checkpoint·publisher 경계는 독립 볼륨으로 유지한다. |
| Q3 | 원본 데이터, split, identity, image, caption을 학습 전 검증 | 원본 전체가 `e62741f90efc9edde37fde3f3ef5c3ed7a774bdd3d7a7fc7c71d6475ed7fd333`으로 검증됐다: 40,206 images, 80,440 captions, 13,003 identities. train/val/test identity 분리와 원본 split protocol을 유지한다. |
| Q4 | 숫자 하이퍼파라미터를 slider나 숫자 입력으로 설정 | Training 화면이 controlled slider와 number input을 동기화한다. 브라우저에서 learning-rate와 weight-decay pointer drag 및 값 동기화를 확인했다. 값 범위와 incomplete draft는 API도 검증한다. |
| Q5 | 승인된 고정 학습 recipe 및 재현 가능한 설정 snapshot | 양방향 CLIP contrastive loss, identity-distinct sampler, AdamW, optimizer-step warmup/cosine, 고정 processor와 결정성 정책을 사용한다. 설정은 작업에 immutable하게 저장한다. |
| Q6 | optimizer·activation·reserve를 포함한 실제 메모리 profile | RTX 5070 Ti에서 batch 16, checkpointing on, FP16/FP32를 각각 측정했다. 이 두 조합만 profile 지원 대상으로 열고 profile이 없는 조합은 fail-closed로 거절한다. 실제 peak/reserve 값은 아래 runtime 항목에 있다. |
| Q7 | GPU 상태를 다시 확인해 부족하면 popup으로 설명하고 작업을 만들지 않음 | source `eef24`에서 admission 당시 free 13,271,826,432 bytes / required 5,908,476,724 bytes로 통과한 뒤 owned pressure를 유지해 free가 4,546,297,856 bytes가 된 상태에서 submit 409 `insufficient_free_memory`를 확인했다. popup에 required/free/reserve가 나왔다. job ID 목록은 전후 동일하고 runner child는 양쪽 다 0이었다. 네 영상이 계속 진행했고 camera2 detection 및 text/similar 각각 30개 결과를 확인한 뒤 pressure를 해제해 free 12,657 MiB로 회복했다. 이후 source-only ready/publication fixes에 맞춰 source7e0 profile을 재보정했으며 pressure run은 반복하지 않았다. |
| Q8 | 한 번에 한 학습 child, 저장된 상태와 제한된 history API | DB slot/CAS와 idempotent request를 사용한다. API에는 인증된 read-only runs mount가 있고 epoch history/log는 run JSONL에 저장한다. inference worker에는 runs mount가 없다. |
| Q9 | 취소와 재시작 뒤 완전한 epoch checkpoint에서 같은 작업 재개 | source7e0 smoke job `a0d0d097-b53b-45ba-9c25-5f566adb4e7c`에서 durable E1 checkpoint와 nonterminal optimizer work를 확인한 뒤 guarded trainer restart를 실행했다. UI resume는 같은 UUID를 owner generation 2로 재개해 E5/step77까지 완료했다. 새 job을 만들지 않았다. |
| Q10 | validation으로 best 선택, test는 최종 평가에만 사용 | native subset run은 validation best epoch 3을 선택했다. held-out test Recall@1/5/10은 baseline `.40625/.8125/1.0`, candidate `.78125/1.0/1.0`이었다. report는 checkpoint, source, dataset, weights hash를 묶는다. 이 작은 smoke split 점수는 제품 품질 증거가 아니다. |
| Q11 | immutable 후보 등록은 자동 적용과 분리 | canonical package `5f81251fa57715078179e23855a67bbf0328a3c4a97f4f6373ab12fe863c5265`에 후보를 등록했다. 같은 weight를 가진 다른 작업과 충돌하지 않도록 report hash에 `training_job_id`를 포함하되 같은 작업 재시도는 안정적으로 유지한다. |
| Q12 | 실제 browser/runtime 증거와 기존 제품 품질 gate 보존 | 실제 UI에서 4개 1280×720 영상의 재생을 확인했다. 과부하 원본 조건에서 안전한 co-existence 범위는 QA 카메라 2의 inference/search 한 대와 4개 영상 재생이다. 후보는 catalog에 보이지만 `prepare` 및 제품 crop evidence가 없어 Apply는 비활성이고 현재 ViT-B/16이 유지된다. |

## 최종 runtime 증거

최종 API/trainer source fingerprint는 `7e0f9f77c1bcd59c81078a6a43921125717d6bd7e42c1d7f1f74d2a396a7bc2e`다. API image SHA-256은 `553ac043035078aa81ce3ab5c7985ba4b19c94b1af5094399c65a03d827f5563e`, training image는 `c2e7125373ae7979ee6b4cc247905cc8d83bc01c6e5b9dc767f827a2d5f4b9bdc`다. Inference worker는 reviewed retention fix가 포함된 source `0b51` image `514816bf59790d83dfaf20448a329933df29b9ce203b9cc8cc1ced303668b74e`를 계속 사용했고 설치된 `retention/service.py` SHA-256은 `28671953126dad7334df4d4d4cb3e4b233c0b8e855a75784fcaad6ca2a43bcea`다. dataset scan 완료 후 authenticated API가 `valid=true`, original fingerprint e627, supervisor source7e0/dataset e627 `ready`를 반환했다.

source7e0 calibration은 GPU `GPU-17913b0a-8144-5f39-7062-15265e5dca33`에서 수행됐다. FP16 profile `0cf9b6aa5ba00e3f10035d0b9808b46193907804630b53779f4751d4d382f5c4`는 training peak 3,344,236,544 bytes, reserve 2,564,240,180 bytes, required 5,908,476,724 bytes다. FP32 profile `de6cd8dbd7e95c466c3ec49594b206f4f5c3714668ce19fb3697aeded8dba7a4`는 peak 3,419,734,016 bytes, 같은 reserve, required 5,983,974,196 bytes다. Calibration file SHA-256: `45c54fb4f9d3df6e7cfa0ac152397961b77c9fbee8b9eaf31ca56655346683b5`.

실제 5-epoch smoke는 native train 1,024 / val 16 / test 16 images subset에서 실행됐다. config는 FP16, batch 16, checkpointing on, gradient accumulation 4, learning rate `1e-5`, weight decay `0.01`, seed 42다. optimizer는 E5/step77까지 완료됐고 child가 종료됐으며 GPU는 3,182 MiB used / 12,657 MiB free로 돌아왔다. candidate model ID는 `local/cuhk-pedes-a0d0d097b53b45ba9c255f566adb4e7c`, weights SHA-256은 `0b87c4959e7381d79b0a590032c7d8db9eac1046aef5a2eb70142015abc096e3`다. 이 398-tensor state는 성공한 uninterrupted reference와 bitwise identical이며, job-bound report 때문에 package SHA는 구별된다. Native subset은 원본 dataset validation과 별도로 준비했으며 제품 UI에서 subset을 선택할 수 없다.

학습 중 browser는 같은 작업의 active optimizer progress, 4개 advancing video, camera2 한 대의 active analysis와 job 시작 이후 timestamp를 가진 camera2 text/Find Similar 결과를 관찰했다. 원본 4-camera × 15fps 조건에서 queue가 계속 쌓이는 용량 제한이 관찰돼 co-existence 증거는 inference를 camera2 한 대로 제한했다. camera1/3/4의 QA detector toggle은 원래 enabled/0.5 설정으로 복원했고 camera2는 시험 내내 유지했다. API/trainer의 original dataset mount도 복원되어 full validation 상태를 확인했다. Inference worker는 final runtime 동안 재시작되지 않았고 retention source0b51/image와 file SHA가 유지됐다. 전·중·후 증거와 actual pressure popup은 `output/training/task9-final-source7e0-resume-r1/`, `output/training/task9-final-source7e0-camera-after-r1/`, `output/training/task9-gpu-shortage-final3/`에 있다.

핵심 증거 파일 SHA-256:

| 증거 | SHA-256 |
|---|---|
| source7e0 FP16/FP32 profile file | `45c54fb4f9d3df6e7cfa0ac152397961b77c9fbee8b9eaf31ca56655346683b5` |
| actual shortage pressure recovery | `df3843374bb3cefcd56c35ca6e11ae0f015e0ebffc637a8e19aa467aedf65e53` |
| source7e0 active job + camera/search snapshot | `37e8bc898e8d9e8bc4110cd8a8016cf507eb9434284fe3132d39080914c0e8bf` |
| guarded restart/source/profile/slot snapshot | `a8053c4b46283cbdc8b40984336f116634167a8ac203190b010f140b35456c6f` |
| resumed-job dataset/supervisor readiness snapshot (not terminal-state evidence) | `aac51b73e33f334d1af8c1a2dc7aa52699b3396a7a5b43cc4eac744f46a668d9` |
| candidate UI gate snapshot | `68ef926daded4a54b7eac23ab944e7ee65cf5fd54ee95421f349cbcefe4fec7e` |
| candidate-phase UI smoke summary only (not terminal-state evidence) | `5cc6c49053b6dba7c4ae4228dd4a49536f8a1802cea726a4ddf69942787df94f` |
| post-training camera/search snapshot | `0932df77b99c1b1a294d438c0aa679b9442fb23c795e5d9ec1d0816771322f84` |
| authenticated API terminal job readback | `3b7af7a84d63834f83f62031118799aa6685383cc062e66d185a94f8e1ae7402` |
| PostgreSQL transaction READ ONLY terminal job/control/slot/checkpoint readback | `20ed58b823033ad4ae995457999bfc76108a736f9bb3c7177261a01e0509d0bd` |
| joined API/DB/canonical-candidate binding receipt | `bebb0f947688b716ee53d76de30076468dbae171acbe7c340f7128335317f166` |
| canonical `cuhk-report.json` | `a0bb886044ab0581208633815b7b886660d7f8b55698428d6325f4edabbc4220` |
| canonical package manifest | `5ac8863b08ff6ea12b56c8b40087cf48d38218c5199904c7db0c45315fda2beb` |

Final state comes from the authenticated API `GET /api/training/jobs/a0d0d097-b53b-45ba-9c25-5f566adb4e7c` and a live PostgreSQL `READ ONLY` transaction observed at `2026-10-05T06:58:20.721793Z`. Both identify the same succeeded UUID at owner generation 2, E5/step77, attempts 2, source7e0 and native dataset ef15. The database records one row for this job, an accepted resume request `9d130f5f-e923-481b-914a-8bae6ce86278` whose parent and job IDs both equal that UUID, a null active slot and no child PID/start time. Resume was created at `05:35:43.848764Z` and resolved at `05:35:44.719748Z`; job start was `05:32:15.238357Z`, engine completion `05:37:31.154291Z`, and terminal finish `05:37:50.302682Z`. The preflight bound profile `0cf9…` on GPU `GPU-17913b0a-8144-5f39-7062-15265e5dca33`. Database checkpoint `/runs/jobs/a0d0d097-b53b-45ba-9c25-5f566adb4e7c/last.pt` exists (2,394,529,167 bytes; mtime `05:37:08.367934Z`). The canonical report binds `training_job_id` to the same UUID; its weights SHA matches both the terminal evaluation and `manifest.json`'s `model.safetensors` record. Full API, database, manifest and report readbacks are preserved under `output/training/task9-final-source7e0-resume-r1/`.

후보 품질 적용은 의도적으로 차단됐다. 제품 기준은 실제 camera crop, 고정 relevance set, 두 장면 이상, 40개 이상 appearance, text 20개 및 image 20개 query가 필요하다. 이 자료는 아직 없으며 CUHK subset score로 대신하지 않는다. 준비되지 않은 model asset은 사용 불가하고 active model은 built-in B/16이다.

## 최종 회귀와 남은 검증 한계

- Server full suite를 현재 source7e0에서 한 번 실행했다: 688 passed, 16 skipped, 5 failed / 227.87s. 세 실패(`test_status` 두 건과 `.omo` Task10 observation 누락)는 이전 baseline에서도 재현됐다. 나머지 두 Docker signal-cleanup 실패는 full-run에서만 발생했다. cleanup receipt는 Docker container 생성 대기를 1초 뒤 종료했지만 늦게 생성된 test-owned container가 남았음을 보인다. 해당 verifier 파일은 base 대비 변경이 없다. 정확히 두 테스트를 base archive에서 2/2, 현재 source에서 2/2 통과했다. 따라서 전체 회귀를 녹색이라고 보고하지 않는다.
- Web unit: repository script `pnpm test` 8 files / 67 passed. TypeScript `tsc -b`, Vite production build, 새 runtime E2E 파일 Biome, Playwright `--list` 모두 통과했다. 실제 GPU/browser workflow는 위 경로의 증거로 검증했으며 마지막에 broad browser run은 반복하지 않았다.
- Python Ruff check는 training/QA/test scope에서 통과했다. Task8/9 review 변경 9개 파일의 format check와 BasedPyright 0/0/0 통과했다. 더 넓은 format sweep은 12 files를 표시했다. 9개는 base39a 대비 변경이 없고, 나머지 3개 pending-delta 파일의 formatter diff도 base39a 파일을 format한 diff와 동일하다. 추가된 변경 줄에는 새로운 format 위반이 없다. 따라서 source fingerprint와 profile을 보존하기 위한 재포맷 변경은 필요하지 않았다.
- 전체 CUHK long-run과 제품 품질/수동 Apply gate는 검증하지 않았다. 실제 지원 profile은 FP16/FP32, batch16, checkpointing on뿐이다. 다른 batch/precision/checkpoint 조합은 별도 source-bound calibration 전까지 사용할 수 없다.
