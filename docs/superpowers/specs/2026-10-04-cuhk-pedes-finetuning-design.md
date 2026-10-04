# 내장 CUHK-PEDES CLIP 파인튜닝

상태: 사용자 승인 완료. Q1~Q12와 구현 계획이 승인되었으며, 전용 브랜치에서 구현·검증을 진행 중이다.

## 목적과 성공 기준

사용자는 웹 UI에서 CUHK-PEDES를 이용한 CLIP ViT-B/16 이미지·텍스트 인코더 전체 파인튜닝을 실행한다. 숫자 하이퍼파라미터는 동기화된 슬라이더와 숫자 입력으로 조절한다. 학습 이력과 평가를 확인하고, 완료된 모델 후보를 기존 모델 선택 경로에서 수동 적용한다.

카메라 분석과 검색은 학습 동안 유지한다. GPU 메모리가 부족하면 학습 시작을 거절하고 필요한 메모리와 가용 메모리를 팝업으로 보여준다. 학습 성공과 모델 적용 가능 상태를 구분한다.

## 현재 코드와 변경 경계

현재 모델 registry, 로컬 package importer, prepared GPU proof, quality evidence 검사, 전환 preflight, durable 재임베딩과 rollback을 재사용한다. 현재 import 명세의 외부 학습 전제는 외부 import와 내장 학습을 모두 지원하도록 갱신한다.

API는 인증된 요청, 설정 검증, 상태 조회를 담당한다. 별도 학습 서비스가 GPU 실행과 checkpoint를 소유하며 API 프로세스나 Triton request handler 안에서 학습하지 않는다. PostgreSQL에 학습 작업과 최종 상태를 저장하고, 실행 파일·checkpoint·metric history는 별도 writable run storage에 보관한다. 기존 API와 추론의 모델 asset mount는 read-only를 유지한다.

## 데이터셋 등록

서버 설정으로 지정한 CUHK-PEDES 디렉터리를 학습 서비스에 read-only로 mount한다. 웹에서는 등록된 데이터셋의 검증 상태를 확인하고 선택한다. 웹 요청이 임의의 호스트 경로를 지정하거나 파일을 업로드하지 않는다.

기본 입력은 imgs/와 reid_raw.json이다. 시작 전에 annotation schema, split, image 경로와 파일 무결성, caption, identity 간 split 분리, 데이터 fingerprint를 검증한다. fingerprint와 split별 image/caption/identity 개수를 작업에 기록한다. 추가 증강 annotation은 기본 입력에 포함하지 않는다. 첫 버전은 기본 annotation만 지원하고 향후 별도 옵션으로 확장한다.

조사된 로컬 annotation의 caption 수는 readme 기재 수와 다르므로 실제 파일을 검증하고 관측된 개수를 표시한다. 일괄적으로 readme 숫자에 맞추거나 데이터를 자동 수정하지 않는다. 학습 데이터와 제품 crop은 저장·평가 경계를 분리한다.

## 학습 recipe와 UI 설정 제안

기본 recipe는 정규화한 이미지·텍스트 embedding의 양방향 대조학습 loss, AdamW, linear warmup 후 cosine learning-rate decay다. 같은 identity의 다른 샘플을 단순 false negative로 취급하지 않도록 batch의 identity 구성을 제어하고, 구체적인 sampler 계약을 구현 계획에 명시한다. 초기 전처리는 기존 ViT-B/16 processor 계약을 유지하며 임의 입력 해상도 변경은 첫 버전에 포함하지 않는다.

| 항목 | 초기값 | 허용 범위 제안 |
|---|---:|---|
| epochs | 30 | 정수 1~100 |
| learning rate | 0.00001 | 0.0000001~0.001, 로그 스케일 슬라이더 |
| micro batch size | 16 | 정수 2~128 |
| weight decay | 0.01 | 0~0.2 |
| gradient accumulation | 4 | 정수 1~32 |
| warmup 비율 | 0.05 | 0~0.3 |
| seed | 42 | 정수 0~2147483647 |
| early stopping patience | 5 | off 또는 정수 1~20 |
| gradient clipping norm | 1.0 | 0.1~10 |
| 혼합 정밀도 | FP16 | FP16 또는 FP32 |
| gradient checkpointing | on | on/off |

기본 화면에는 epochs, learning rate, batch size, weight decay를 둔다. 나머지는 고급 설정에 둔다. boolean은 토글, enum은 선택 컨트롤이다. 슬라이더와 숫자 입력은 하나의 값을 공유하며 키보드 조작과 label을 제공한다. 서버도 동일한 범위를 검증한다. 실행 전 설정 요약을 보여주고, 실행한 설정은 immutable snapshot으로 저장한다.

유효 batch는 micro batch × accumulation으로 표시한다. gradient accumulation은 optimizer 갱신 단위를 늘리지만 각 micro batch의 대조학습 negative pool을 늘리지 않는다는 설명을 제공한다. 최소 한 학습 batch를 구성할 수 없는 데이터는 실행 전 거절한다. AdamW betas/epsilon 및 processor는 첫 버전의 고정 recipe에 기록한다.

## 메모리 추정과 시작 판정

메모리 추정은 모델 parameter, gradient, optimizer state, precision별 추가 복사본, activation, CUDA workspace 및 allocator 여유를 포함한다. batch, sequence/image shape, precision, checkpointing에 따른 activation 변화를 반영한다. optimizer state가 생성되는 실제 training step의 peak memory로 RTX 5070 Ti profile을 보정한다. 단순 inference memory를 학습 추정치로 사용하지 않는다.

UI는 선택된 설정의 추정 사용량, 여유분, 현재 GPU 가용 공간과 판정 시각을 표시한다. 서버는 시작·재개 요청 직전에 GPU 상태를 다시 읽고 작업 소유권을 확보한 상태에서 판정한다. profile이 없거나 GPU 상태를 읽지 못하면 추정 불가로 거절한다. 초기 추론 여유분 제안은 2 GiB와 GPU 총 메모리의 15% 중 큰 값이며 실제 추론 부하 측정 결과에 따라 올린다. 이 여유분은 사용자 임의 축소 항목으로 노출하지 않는다.

거절된 요청은 실행 작업이나 자동 대기 작업으로 만들지 않는다. 팝업은 거절 이유, 추정 학습 메모리, 추론 여유분, 가용 메모리 및 설정 변경 안내를 보여준다. 다른 작업 시작과 경쟁해 동시에 실행되는 것을 DB 소유권/단일 실행 제약으로 막는다.

시작 판정은 이후 메모리를 예약하거나 성공을 보장하지 않는다. 실행 중 OOM은 학습 실패로 기록하고 학습 프로세스를 종료하여 메모리를 해제한다. 최근 완전히 저장된 checkpoint와 오류 이력을 보존한다. batch size를 자동 변경하거나 자동 재시도하지 않는다. 더 작은 batch를 선택한 재실행은 새로운 설정을 가진 새 작업이다.

## 작업 수명과 복구

학습 한 건만 실행한다. 동시에 실행 중인 작업이 있으면 추가 시작을 팝업으로 거절한다. 작업 상태는 starting, training, evaluating, publishing, succeeded, cancelling, cancelled, failed, interrupted로 구분한다. 시작 거절은 작업 상태가 아니라 요청 결과다.

취소 요청은 cooperative cancellation으로 처리하고, 안전한 경계에서 중단하며 현재 부분 파일을 완성된 checkpoint로 취급하지 않는다. epoch 완료마다 atomic checkpoint를 저장한다. checkpoint에는 model, optimizer, scheduler, AMP scaler, RNG 상태, sampler 진행 기준, best validation metric 및 설정/dataset/source identity를 포함한다.

서비스 재시작 후 실행 소유권이 사라진 작업은 interrupted로 표시한다. 자동 재실행하지 않는다. 사용자는 마지막 완전한 checkpoint에서 재개할 수 있으며 서버는 동일 설정·dataset·source 및 메모리를 재검사한다. 설정을 변경하면 기존 작업 재개가 아니라 새 작업이다. resume attempt 이력은 남긴다. 이전 프로세스가 실제로 종료되었는지 확인한 후 소유권을 넘긴다.

## 평가와 후보 생성

CUHK-PEDES 기존 train/val/test 분할을 유지한다. train으로 학습하고 매 epoch validation text-to-image Recall@1로 best checkpoint를 선택한다. early stopping도 같은 metric을 사용한다. test는 선택 완료 후 최종 평가에만 사용한다.

평가 gallery는 해당 split의 image이며 relevance는 동일 person identity다. text query마다 적합한 image가 top K 안에 있는지를 기준으로 macro Recall@1/5/10을 계산한다. baseline과 candidate는 동일 split, 입력과 metric 코드에서 비교하며 source revision, weights hash, dataset fingerprint, evaluator revision을 보고서에 기록한다. baseline은 같은 fingerprint와 evaluator identity로 묶인 결과만 재사용한다.

학습 결과는 표준 CLIPModel/CLIPProcessor와 safetensors로 내보내고, CUHK 보고서 및 package.json을 생성한다. 기존 importer를 통해 immutable candidate를 게시한다. export/import 실패는 작업 실패로 표시하며 완성된 checkpoint는 보존한다. 재개 시 완료된 평가·게시 단계의 identity를 확인해 중복 후보를 만들지 않는다.

학습 완료 후 자동 적용하지 않는다. candidate는 기존 registry와 모델 선택기에 표시한다. GPU preparation, CUHK 평가, 별도 제품 crop 품질 evidence, 전체 전환 preflight가 통과한 경우에만 수동 적용한다. CUHK 평가 결과가 제품 Recall@5 증거를 대신하지 않는다. 품질 evidence 생성 경로가 없는 기존 부분은 구현 범위에서 명시적으로 보완하여 검토 가능한 평가 결과를 만들고, 제품 fixture가 제공되지 않으면 적용 차단 이유를 표시한다.

## 웹/API 동작

인증된 학습 화면에 데이터셋 상태, 설정, 메모리 사전 검사, 시작 버튼, 이력과 상세 페이지를 둔다. 시작 버튼의 활성 여부는 UI 예상 상태이며 서버 판정이 최종 권위다. 중복 클릭은 한 번의 시작 요청으로 묶고 request identity로 중복 실행을 막는다.

상세 화면은 immutable 설정, epoch/step, loss와 validation Recall 그래프, GPU memory telemetry, 경과 시간, 로그, 최종 평가 및 후보 연결을 보여준다. 불명확한 예상 완료 시간을 확정값으로 표시하지 않는다. 화면 새로고침과 연결 복구 후 저장된 작업 상태를 다시 조회한다. 로그 응답 크기를 제한하고 경로·credential·원본 caption 등의 불필요한 정보를 노출하지 않는다.

API 계약은 등록 데이터셋 조회/검증 상태, 학습 설정과 메모리 preflight, 작업 생성/목록/상세, metric/log pagination, 취소, 재개를 포함한다. existing operator 인증을 적용하고 셸 명령이나 호스트 경로를 입력으로 받지 않는다. 실행별 저장 공간을 격리하고 사전 디스크 용량 검사·명확한 오류 처리를 제공한다. 첫 버전의 이력/checkpoint 삭제와 자동 보존 기간은 도입하지 않는다.

## 검증과 전달

설정 경계, dataset path/split/identity 검증, 메모리 거절·재확인·동시 요청, 취소·OOM·재시작·resume, checkpoint 원자성, 평가 metric, candidate export/import와 기존 적용 gate를 검증한다. 브라우저에서는 슬라이더/숫자 동기화, 키보드 접근성, 부족 팝업, 이력·새로고침·취소·재개 및 후보 연결을 확인한다.

실제 GPU에서 소규모 실제 데이터 학습·검증·export/import를 실행하여 가중치 변경과 재개를 확인하고, 기존 추론과 함께 실행할 때의 peak memory 및 검색·분석 동작을 측정한다. 전체 CUHK-PEDES 장기 학습이나 제품 품질 기준을 통과하지 않은 결과는 별도로 명시한다. 구현은 사용자 지침에 따라 GPT Luna max가 담당한다.

## 관련 문서

- [도메인 용어](../../../CONTEXT.md)
- [추론을 유지하는 학습 판정](../../adr/0001-training-admission-with-live-inference.md)
- [기존 모델 import 명세](2026-09-25-finetuned-clip-import.md)
