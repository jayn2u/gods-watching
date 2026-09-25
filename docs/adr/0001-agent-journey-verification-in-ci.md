---
status: accepted
---

# Agent Journey 검증을 개발 머신의 self-hosted runner에서 Claude 구독 CLI로 실행한다

PR에 `agent-verify` 라벨이 붙으면, 에이전트가 모든 Journey를 실제 GPU 스택에서 수행하고 Bug Report를 GitHub Issue로 남긴다. 실행은 개발 머신(RTX 5070 Ti) 한 대를 self-hosted runner로 쓰고, 에이전트는 `jwchoi` 계정의 Claude 구독 로그인으로 `claude -p --model opus`를 subprocess로 호출한다. 이렇게 정한 이유는 전체 스택에 NVIDIA GPU와 GPU UUID에 묶인 모델 cache가 필요하고, 사용할 수 있는 GPU 머신이 이 한 대뿐이며, 추가 API 과금 없이 기존 구독으로 운영하기 위해서다.

## Considered Options

- **Claude Agent SDK**: 공식 문서가 SDK에는 claude.ai 구독 로그인 대신 API 키를 쓰라고 안내하므로 기각했다. 구독 인증이 공식 지원되는 경로는 Claude Code CLI와 `claude-code-action`이다.
- **GitHub-hosted runner, GPU larger runner**: GPU가 없거나 유료 플랜이 필요하다. 또 모델 cache를 실행마다 다시 준비해야 하므로 기각했다.
- **별도 runner 머신과 전용 OS 사용자**: 검토했지만, 머신 한 대와 기존 로그인 파일을 그대로 쓰는 단순함을 택했다.
- **야간 실행, develop push 트리거**: 결과를 변경 단위에 묶고 개발 흐름을 방해하지 않도록 PR 라벨 opt-in만 남겼다.

## Consequences

- 실행이 시작되면 개발 스택을 내렸다가, 끝나면 원래 상태로 되돌린다. 16 GB VRAM을 두 스택이 나눠 쓰면 GPU 메모리 부족이 제품 버그처럼 보이는 오탐이 생기기 때문이다. 같은 이유로 라벨은 일회성이다.
- runner가 개인 계정으로 돌기 때문에 에이전트 도구 제한이 유일한 보안 경계다. 에이전트는 브라우저, 로그 읽기 wrapper, GET 전용 API wrapper만 쓸 수 있고 bash와 docker는 쓸 수 없다.
- CI가 개인 구독 사용량을 개발 작업과 나눠 쓴다. 한도가 소진되면 남은 Journey를 `inconclusive`로 끝낸다.
- Cross-check 판정자는 교체할 수 있게 둔다. 지금은 실행 기록을 모르는 별도 Claude 세션이 판정하고, 나중에 Codex를 도입하면 판정자를 Codex로 바꿔 같은 모델끼리 편향을 공유하는 문제를 줄인다.
