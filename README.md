# Gods Watching

GPU 기반 RTSP 인물 검색 서버입니다. 브라우저에서 실시간 카메라를 보고, 감지된 인물 crop을 영어 설명이나 기존 결과로 검색할 수 있습니다. 원본 영상은 저장하지 않습니다.

## 요구 사항

- Linux x86_64
- NVIDIA GPU와 동작하는 NVIDIA Container Toolkit
- Docker Engine 29 이상과 Docker Compose v2
- 잠긴 Python 환경과 `./gods-watching` launcher를 실행할 [uv](https://docs.astral.sh/uv/)
- 처음 빌드할 때 약 600 MB의 고정 CLIP 모델과 YOLO 가중치를 받을 수 있는 네트워크

기본 구성은 로컬 확인용입니다. `prepare`는 `.env`가 없을 때 운영자·PostgreSQL·내부 미디어 비밀번호와 Fernet 키를 무작위로 생성하고 파일 권한을 `0600`으로 설정합니다. 기존 값은 덮어쓰지 않으며, 이전 준비 파일에 내부 미디어 비밀번호만 없으면 두 값을 추가합니다. 이미 저장된 카메라가 있는 상태에서 `GW_CAMERA_CIPHER_KEY`를 바꾸면 기존 RTSP 자격 증명을 복호화할 수 없습니다.

## 켜기

```bash
./gods-watching doctor   # Docker, Compose, NVIDIA GPU, Compose 구성 점검
./gods-watching prepare  # 최초 자격 증명 생성 및 고정 이미지 빌드
./gods-watching up       # docker compose up -d
./gods-watching status   # docker compose ps
./gods-watching down     # docker compose down
```

첫 실행은 애플리케이션과 Triton 이미지를 빌드하고 모델을 내려받기 때문에 시간이 걸립니다. 모든 서비스가 준비되면 <http://localhost:8080>으로 접속합니다.

운영자 로그인 정보는 `.env`의 `GW_OPERATOR_USERNAME`과 `GW_OPERATOR_PASSWORD`가 단일 기준입니다. API는 시작할 때마다 `.env`의 비밀번호를 데이터베이스 credential에 반영하므로, 두 값을 `admin`으로 두면 `admin`/`admin`으로 로그인합니다. 아이디는 앞뒤 공백을 제거하고 대소문자를 구분하지 않으며, 비밀번호는 4~128자만 허용합니다. `.env` 값을 바꾸면 다음 시작 때 기존 세션이 모두 해지되고, `credentials set`으로 바꾼 비밀번호도 `.env` 값으로 되돌아갑니다. `./gods-watching doctor`는 파일 권한, 필수 값, placeholder 사용 여부를 확인하지만 비밀 값은 출력하지 않습니다.

`admin`/`admin`은 로컬 확인용 기본값입니다. LAN에 공개할 때는 `.env`의 두 값을 먼저 교체하십시오.

## 공개 포트 바꾸기

`8080/tcp`가 이미 사용 중이면 `.env`의 `GW_PUBLIC_PORT`를 바꿉니다. TLS overlay의 HTTPS 포트는 `GW_PUBLIC_TLS_PORT`입니다.

```dotenv
GW_PUBLIC_PORT=9080
GW_PUBLIC_ORIGIN=http://localhost:9080
```

`GW_PUBLIC_ORIGIN`의 포트를 함께 바꾸지 않으면 브라우저의 로그인 요청이 same-origin 검사에서 403으로 거부됩니다. `./gods-watching doctor`가 두 값의 불일치와 사용할 수 없는 포트 번호를 미리 잡아냅니다. 변경 후에는 `./gods-watching up`으로 gateway를 다시 만들어야 새 포트가 적용됩니다.

## LAN TLS

LAN에 공개할 때 `.env`의 `GW_PUBLIC_HOST`를 서버의 고정 IP 또는 DNS 이름으로 바꾸고 다음 줄을 추가합니다.

```dotenv
COMPOSE_FILE=compose.yaml:compose.tls.yaml
GW_PUBLIC_ORIGIN=https://192.0.2.10:8443
```

TLS overlay는 Caddy의 내부 CA와 인증서를 named volume에 유지하고 `GW_PUBLIC_PORT`를 `GW_PUBLIC_TLS_PORT` HTTPS로 redirect합니다. 시작 후 CA 인증서를 내보내 각 운영자 브라우저 또는 OS trust store에 한 번 설치합니다.

```bash
mkdir -p runtime
docker compose cp gateway:/data/caddy/pki/authorities/local/root.crt runtime/gods-watching-ca.crt
```

API는 TLS overlay에서 secure session cookie를 강제합니다. Caddy만 신뢰 gateway 주소에서 API에 접근할 수 있으며, Caddy의 기본 reverse-proxy 정책이 외부 `X-Forwarded-For` 값을 폐기하고 직접 연결 주소로 다시 작성합니다.

상태는 다음 명령으로 확인합니다.

```bash
docker compose ps
docker compose logs --tail=100 api worker triton
```

로그인한 운영자는 `GET /api/status`에서 worker heartbeat, 카메라별 decode/detector rate와 frame age, embedding queue, 검색 가능 latency, Triton, crop/DB 저장 압력을 JSON으로 확인할 수 있습니다. 응답의 `state`가 `ready`가 아니거나 source가 stale/offline이면 로그와 해당 카메라의 RTSP 접근성을 먼저 확인하십시오. 저장 압력으로 publication이 멈춘 경우 retention 또는 filesystem 여유 공간을 복구하면 worker가 bounded queue를 다시 처리합니다. 현재 브라우저 UI에는 별도 상태 화면이 없습니다.

## 검증

설치된 검증은 실행마다 격리된 namespace와 evidence 디렉터리를 만들고 종료 시 자신이 만든 resource만 정리합니다.

```bash
mkdir -p runtime/evidence
./gods-watching verify --scenario search --evidence runtime/evidence/search
./gods-watching verify --scenario retention --evidence runtime/evidence/retention
```

각 명령은 terminal JSON에 `outcome`, `exit_code`, source hash, Git/diff identity, check와 artifact 경로를 기록합니다. 지원되는 이름이라도 아직 구현되지 않은 scenario는 성공으로 대체하지 않고 `scenario_unavailable`로 종료합니다. 특히 `full`은 실제 retrieval-quality, 15분 load, fault matrix, production-browser gate가 끝나기 전까지 fail-closed 상태입니다. 현재 완료/미완료 목록은 `docs/remaining-work.md`를 확인하십시오.

운영 전 최소 점검은 `./gods-watching doctor`, `./gods-watching status`, 브라우저 login, 카메라 test/create, live frame 증가, text search와 Find similar, 재시작 후 설정/검색 결과 유지입니다. 원본 영상 파일이 crop volume에 생기지 않았는지도 확인하십시오.

## 끄기

```bash
docker compose down
```

이 명령은 컨테이너와 네트워크만 제거합니다. PostgreSQL 데이터와 인물 crop은 named volume에 유지되어 다음 실행에서 복구됩니다. 저장 데이터까지 삭제하려는 경우에만 명시적으로 `docker compose down --volumes`를 사용하십시오.

## 문제 해결

- `doctor`의 `credentials=false`: `.env`가 존재하고 mode `0600`인지, 필수 비밀번호와 Fernet 키가 placeholder가 아닌지 확인합니다. 비밀 값 자체는 명령 출력에 나타나지 않습니다.
- `doctor`의 `gpu=false`: host의 `nvidia-smi`와 NVIDIA Container Toolkit을 복구한 뒤 다시 실행합니다. CPU fallback으로 준비 완료를 주장하지 않습니다.
- login 실패: `.env`의 비밀번호는 새 데이터베이스 최초 생성 때만 초기 credential로 쓰입니다. 기존 volume에서 `.env` 값만 바꿔도 저장된 password가 바뀌지 않습니다.
- TLS 경고: `runtime/gods-watching-ca.crt`를 접속하는 OS/browser trust store에 설치하고 URL host가 `GW_PUBLIC_HOST`와 일치하는지 확인합니다.
- camera offline: H.264 RTSP/TCP URL과 credential을 camera test로 다시 확인합니다. 응답과 로그에는 credential-bearing URL이 redaction됩니다.
- search text만 503: Triton text encoder 상태를 확인합니다. 이미 저장된 vector 기반 browse/Find similar은 inference outage 중에도 계속 동작해야 합니다.

## 서비스 구성

- `gateway`: 브라우저가 접속하는 Caddy HTTP/선택적 LAN TLS 경계
- `api`: 인증, 카메라 관리, 실시간 WHEP 프록시, 검색 API와 React 앱
- `worker`: RTSP decode, GPU 검출·embedding, crop 보존과 retention
- `triton`: GPU 0에서 실행되는 YOLO11s와 CLIP 추론 서버
- `media-gateway`: MediaMTX RTSP/WebRTC 경계
- `postgres`: pgvector가 설치된 PostgreSQL 17
- `migrate`: 시작할 때 Alembic migration을 적용하고 종료하는 one-shot 서비스

기본 공개 포트는 HTTP `8080/tcp`(`GW_PUBLIC_PORT`)와 WebRTC media `8189/udp`입니다. TLS overlay에서는 HTTPS `8443/tcp`(`GW_PUBLIC_TLS_PORT`)도 공개하며 HTTP 포트는 redirect 전용입니다. PostgreSQL, Triton, MediaMTX 제어·WHEP 포트는 Compose 네트워크 내부에만 존재합니다.
