# Gods Watching

GPU 기반 RTSP 인물 검색 서버와, 그 서버가 정상 동작하는지 확인하는 검증 체계의 공용 언어.

## Verification

**Scenario**:
코드로 작성된 결정적 검증 단위. 정해진 check를 판정하고 evidence를 남긴다. `verify --scenario`로 실행한다.
_Avoid_: 테스트 시나리오, agent scenario

**Journey**:
자연어로 기술한 사용자 관점의 검증 과제. 에이전트가 실제 앱을 조작하며 수행하고, 정상 여부를 판단한다. Scenario와 별개의 개념이다.
_Avoid_: 시나리오, Agent Test Case, test case

**Journey Run**:
Journey 하나를 에이전트가 한 번 수행한 기록. 조작 과정, evidence, Verdict를 포함한다.

**Verdict**:
Journey Run의 판정 결과. `pass`, `bug`, `inconclusive` 중 하나다. 판정 근거가 부족하거나 에이전트가 중단된 경우는 `inconclusive`이며 버그가 아니다.
_Avoid_: result, status, 성공/실패

**Expected Outcome**:
Journey에 명시된, 관찰 가능한 정상 동작 조건. `bug` Verdict는 Expected Outcome 위반으로만 내려진다. 5xx 응답과 처리되지 않은 브라우저 예외가 없어야 한다는 조건은 모든 Journey에 공통으로 포함된다.
_Avoid_: 기대값, assertion

**Observation**:
Expected Outcome 위반은 아니지만 에이전트가 이상하다고 본 현상. Verdict에 영향을 주지 않고 Bug Report가 되지 않으며, Journey Run 기록에만 남는다.
_Avoid_: warning, minor bug

**Cross-check**:
Journey Run의 조작 과정을 모르는 독립된 판정자가 `bug` Verdict를 evidence와 Expected Outcome만으로 재검토하는 절차. 두 판정이 모두 `bug`일 때만 Bug Report가 된다.
_Avoid_: 재검증, double check

**Bug Report**:
Cross-check를 통과한 `bug` Verdict를 사람이 읽을 수 있게 정리한 기록. 재현 절차, 기대 동작, 실제 동작, evidence를 담는다.
_Avoid_: issue, 에러 리포트
