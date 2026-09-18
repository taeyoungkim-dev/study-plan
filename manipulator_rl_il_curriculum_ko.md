# 로봇 매니퓰레이터 강화학습·모방학습 전문가를 위한 커리큘럼

> 대상: Computer Science 학부 4학년. 목표: 로봇팔의 동작을 이해하고, RL/IL 정책을 구현·평가하며, 논문 재현과 독립적인 연구를 수행하는 역량.
>
> 최초 작성일: 2026-09-17 · 완성·개정일: 2026-09-18. 사용자 배경: CS 학부 4학년, PyTorch 사용 경험은 있으나 기초부터 학습 필요. 보유 장비: **SO-ARM-101 leader·follower 세트와 TurtleBot3 한 대**. 주당 공부시간과 완료 기한을 정하지 않고, **역량과 산출물을 기준으로 진도**를 결정한다. 통과 기준은 교육용 제안이며 공인 자격 기준이나 성능 보장이 아니다.

## 목차

1. [목표와 사용법](#s1)
2. [전체 로드맵과 선후 관계](#s2)
3. [시작 전 진단](#s3)
4. [단계별 상세 커리큘럼](#s4)
5. [프로젝트와 평가 기준](#s5)
6. [실습 환경과 장비 선택](#s6)
7. [핵심 논문 읽기 순서](#s7)
8. [심화 전문 분야](#s8)
9. [연구 방법과 재현성](#s9)
10. [학교 수업·연구실·진로 연결](#s10)
11. [처음 실행할 과제와 학습 운영](#s11)
12. [별도 체크리스트 사용법](#s12)

CS 과목 바로가기: [자료구조와 알고리즘](#cs-1) · [컴퓨터 구조](#cs-2) · [운영체제](#cs-3) · [네트워크](#cs-4).

<a id="s1"></a>
## 1. 목표와 사용법

추천하는 중심 경로는 **CS 공통 기반과 수학·딥러닝 → 기구학·제어 → RL 기초와 BC → 매니퓰레이터 RL → 시각 기반 IL → offline RL과 실제 시스템 → 재현·연구**이다. CS 공통 기반은 자료구조·알고리즘, 컴퓨터 구조, 운영체제, 네트워크로 구성하며 나머지 학습과 병행한다.

이 분야의 전문성은 다음 다섯 가지를 함께 갖추는 데 있다.

| 역량 | 실제로 할 수 있어야 하는 일 |
|---|---|
| 로봇 시스템 이해 | 좌표계, 관절, 그리퍼, 센서, 제어 주기와 접촉을 설명하고 오류를 찾는다. |
| 학습 이론 이해 | 정책의 목적함수와 가정을 설명하고, 왜 학습이 불안정하거나 실패하는지 분석한다. |
| 구현 | 데이터 수집부터 학습, rollout, 체크포인트, 결과 집계까지 연결한다. |
| 실험 | 강한 baseline과 공정하게 비교하고 일반화·불확실성·실패 유형을 보고한다. |
| 연구 | 기존 결과를 재현하고, 검증 가능한 질문을 만들어 새로운 증거를 얻는다. |

**필수**는 공통 기반, **심화**는 독립 연구에 필요한 확장, **선택**은 관심 분야에 따라 고르는 내용이다. 선택 과정을 전부 마칠 필요는 없다. 통과 과제를 이미 해결할 수 있는 단원은 복습만 하고 넘어간다. 교재는 주교재 한 권을 중심으로 공부하고 나머지는 해당 개념을 보충할 때 사용한다.

이 문서는 핵심 지식을 빠짐없이 연결하는 역량 지도다. 모든 논문과 세부 분야를 끝내야 연구를 시작할 수 있다는 뜻은 아니다. 기초 프로젝트가 생기는 시점부터 연구실 활동을 병행한다. 전문가로서의 판단력은 여러 문제를 해결하며 축적해야 한다.

사용자에게는 **PyTorch 기초 → 시뮬레이터의 작은 정책 → SO-ARM-101의 데이터·제어 파이프라인 → 실물 IL**을 실습의 중심으로 추천한다. RL의 탐색과 반복 실험은 먼저 시뮬레이터에서 익힌다. TurtleBot3는 ROS 2·센서·좌표계와 이동 로봇을 배우는 선택용 장비로 활용한다.

### 먼저 구별할 개념

| 개념 | 학습에 사용하는 정보 | 매니퓰레이터에서의 예 |
|---|---|---|
| 제어·계획 | 모델, 목표, 비용과 제약 | IK와 궤적 계획으로 그리퍼를 목표 자세로 이동 |
| Behavior Cloning, BC | 시연의 관측·행동 쌍 | 사람이 수집한 집기 동작을 지도학습 |
| Interactive IL | 정책이 방문한 상태에 대한 전문가의 추가 행동 라벨 | DAgger로 복구 동작 데이터 수집 |
| Online RL | 환경 상호작용과 보상 | 집기 성공 보상을 높이는 정책 개선 |
| Offline RL | 고정된 전이 데이터와 보상 또는 정의 가능한 보상 | 기존 로봇 로그로 정책 개선 |
| VLA | 영상·언어 조건에서 행동을 내는 모델 계열 | 언어 지시로 행동 생성. 학습 방식은 IL/RL 등과 조합 가능 |

RL과 IL은 경쟁 관계로만 볼 필요가 없다. 시연으로 초기 정책을 만들고 RL로 개선하거나, 계획기에서 시연을 생성하는 조합도 연구 대상이다. 행동 정책은 보통 저수준 제어기를 통해 실행되므로, 신경망 출력과 모터 명령의 관계를 반드시 이해해야 한다.

<a id="s2"></a>
## 2. 전체 로드맵과 선후 관계

단계는 마감일이 아니라 의존 관계를 나타낸다. 각 단계의 통과 기준을 충족하면 다음으로 이동하고, 학부 수업에서 이미 익힌 부분은 진단 과제로 확인한 뒤 생략한다. 로봇 학습에 필요한 이론은 실습과 반복해서 연결한다.

| 단계 | 주제 | 대표 산출물 |
|---|---|---|
| 0 | 개발 환경·수치 계산·실험 습관 | 재현 가능한 작은 학습 저장소 |
| CS-1~4 | 자료구조·알고리즘 / 컴퓨터 구조 / 운영체제 / 네트워크 | 탐색·버퍼 구현, 메모리 성능 측정, 동시성·통신 파이프라인 |
| 1 | 수학·확률·최적화 | 미분·최소제곱·확률 실습 |
| 2 | ML·딥러닝·PyTorch | 학습·검증·평가 파이프라인 |
| 3 | 기구학·동역학·접촉 | 2-link arm과 보유 로봇 모델의 FK/IK |
| 4 | 제어·계획·지각 기초 | 학습 없이 수행하는 pick-and-place |
| 5 | RL 이론·연속 제어 | 작은 환경의 PPO·SAC |
| 6 | BC·DAgger·시연 데이터 | 상태 기반 IL baseline |
| 7 | 매니퓰레이터 RL | reaching/pushing/pick-and-place RL |
| 8 | 시각·시퀀스·생성형 IL | ACT 또는 Diffusion Policy 재현 |
| 9 | Offline RL·model-based·RL+IL | BC와 offline/online 개선 비교 |
| 10 | 시스템 통합·sim-to-real | SO-ARM-101의 실제 rollout·전이 평가 |
| 11 | 논문 재현·독립 연구 | 재현 보고서와 연구 프로젝트 |

단계 1과 2는 연결해서 공부해도 된다. 단계 3~4의 기초와 단계 2가 확보되면 **단계 5의 RL과 단계 6의 BC를 병행**할 수 있다. SO-ARM-101 연결·센서 기록·작은 위치 명령 검증은 단계 3부터 시작하고, 자율 정책 실행은 기본 제어와 데이터 검증 이후 진행한다.

```text
수학 ───────────────┬── ML/DL ── RL 기초 ── 매니퓰레이터 RL ──┐
                    │          └─ BC/DAgger ── 시각·생성형 IL ─┤
                    │                                        ├─ RL+IL·전이·연구
                    └── 기구학/동역학 ── 제어·계획·지각 ─────┘
CS 공통 기반: 자료구조·알고리즘 / 컴퓨터 구조 / 운영체제 / 네트워크
              └─ 데이터 구조·계산 자원·동시성·통신을 전 과정에 적용
전 과정: 구현 → 실험 기록 → 실패 분석 → 재현 가능한 보고서
```

실습은 가능한 한 **한 로봇 모델과 하나의 시뮬레이션 환경군**을 유지한다. 관측·행동·성공 조건을 함께 바꾸면 알고리즘 차이를 해석하기 어렵다. 첫 번째 성공 경험은 특권 상태(state)를 사용하는 작은 과제로 만들고, 이후 RGB/RGB-D 관측으로 확장한다.

<a id="s3"></a>
## 3. 시작 전 진단

각 문제를 `0: 모름 / 1: 자료를 보며 가능 / 2: 독립적으로 구현·설명 가능`으로 표시한다. 총점보다 각 단계의 부족한 부분을 찾는 데 사용한다.

| 진단 문제 | 부족하면 복습할 단계 |
|---|---|
| NumPy 배열의 shape와 broadcasting을 설명하고 행렬 연산으로 반복문을 줄인다. | 0 |
| Git으로 버전을 관리하고, 가상환경에서 프로그램을 다시 실행한다. | 0 |
| 자료구조의 연산 비용과 BFS·Dijkstra의 적용 조건을 설명하고 작은 탐색 문제를 해결한다. | CS-1 |
| 정수·부동소수점·cache locality를 설명하고 배열 연산의 병목을 측정한다. | CS-2 |
| 프로세스·스레드·가상 메모리를 구별하고 bounded producer–consumer를 구현한다. | CS-3 |
| TCP byte stream과 UDP datagram을 구별하고 메시지 경계·timeout·지연 측정을 설명한다. | CS-4 |
| 최소제곱을 유도하고 SVD/pseudoinverse가 필요한 이유를 설명한다. | 1 |
| 합성함수의 Jacobian을 구하고 finite difference로 검증한다. | 1 |
| 조건부확률·기댓값·분산·MLE·KL divergence를 예제로 설명한다. | 1 |
| PyTorch 모델을 직접 학습하고 작은 배치를 의도적으로 overfit한다. | 2 |
| train/validation/test와 데이터 누수를 설명한다. | 2, 6 |
| 회전·동차변환을 합성하고 FK와 IK의 차이를 설명한다. | 3 |
| PD 제어의 overshoot와 지연의 영향을 실험으로 보인다. | 4 |
| Bellman equation, on-policy/off-policy, BC와 RL의 차이를 설명한다. | 5, 6 |

관련 항목을 모두 2로 평가하고 작은 구현 결과까지 보여줄 수 있으면 해당 단계는 빠르게 통과한다. 강의를 본 기억만으로 생략하지 않는다.

<a id="s4"></a>
## 4. 단계별 상세 커리큘럼

각 표의 **필수 개념은 정의·역할·가정·대표 예제를 설명할 수 있어야 할 범위**다. 모든 알고리즘의 전체 증명이나 처음부터 구현하는 과제가 필수라는 뜻은 아니다. 직접 구현할 범위는 해당 단계의 실습·통과 기준과 학습 깊이를 따른다. 한국어와 영어 원어를 함께 적었고, 주요 알고리즘은 약어도 병기했다.

로봇 학습 단계의 항목 ID는 `S단계-번호`, CS 과목은 `CS과목-번호` 형식이다. 예를 들어 `S02-07`은 단계 2의 자동미분, `CS3-06`은 운영체제의 동시성·동기화 영역이다. 별도 체크리스트에 같은 ID를 사용하므로 모르는 개념과 완료 증거를 연결해 기록할 수 있다. 표는 위에서 아래의 순서로 공부하되 앞 단계의 개념을 필요할 때 복습한다.

<a id="stage-0"></a>
### 단계 0. 개발·실험 기반

**목표:** 다른 사람이 같은 설정으로 내 실험을 다시 실행할 수 있게 만든다.

**반드시 알아야 할 개념**

| 항목 ID | 영역 | 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| S00-01 | Python 기본 구조 | 함수(Function), 클래스(Class), 모듈(Module), 패키지(Package), 반복자(Iterator), 제너레이터(Generator), 예외 처리(Exception Handling), 타입 힌트(Type Hint) | 데이터·모델·환경 코드를 읽고 모듈화 |
| S00-02 | 배열과 수치 계산 | 다차원 배열(N-dimensional Array), 형상(Shape), 자료형(Dtype), 인덱싱(Indexing), 브로드캐스팅(Broadcasting), 벡터화(Vectorization), 부동소수점 오차(Floating-point Error) | 관측·행동 배치 처리와 수치 오류 진단 |
| S00-03 | Tensor 입문 | Tensor, CPU/GPU Device, Reshape, Transpose, Permute, 행렬곱(Matrix Multiplication), 원소별 연산(Element-wise Operation) | PyTorch 입력과 출력의 차원 확인 |
| S00-04 | 프로그램 실행 환경 | 가상환경(Virtual Environment), 의존성 고정(Dependency Pinning), 환경 변수(Environment Variable), 상대·절대 경로(Relative/Absolute Path), 명령행 인자(Command-line Argument) | 학습 환경 재구성 |
| S00-05 | 개발 도구 | Shell, Process, Standard Input/Output, SSH, Git Commit, Branch, Diff, Merge | 원격 실험과 코드 변경 추적 |
| S00-06 | 검증·디버깅 | Assertion, Unit Test, 예외 추적(Traceback), 중단점(Breakpoint), 수치 미분 검사(Finite-difference Gradient Check) | 학습 전 작은 기능 검증 |
| S00-07 | 실험 기록 | Configuration, 의사난수 생성기(Pseudorandom Number Generator), Random Seed, Logging, Checkpoint, Profiling | 설정·결과·실행 비용 보존 |

**학습 깊이:** C++의 포인터·참조·객체 수명과 CMake는 로봇 SDK나 제어 코드를 수정할 때 추가한다. 단계 0의 PyTorch는 tensor 조작까지 익히고, 학습 API는 단계 2에서 체계적으로 다룬다.

**실습:** 선형회귀를 NumPy와 PyTorch로 각각 구현하고 동일 데이터에서 비교한다. 설정 파일, 실행 명령, seed, loss plot과 checkpoint를 저장한다.

**통과 기준:** 새 가상환경에서 README만 보고 학습·평가를 재실행한다. 학습 중단 후 재개가 가능하고, shape/device 오류를 스스로 찾는다.

자료: [MIT Missing Semester](https://missing.csail.mit.edu/)의 shell·Git·debugging·profiling, [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/).

<a id="cs-core"></a>
### CS 공통 기반 — 자료구조·알고리즘, 컴퓨터 구조, 운영체제, 네트워크

네 과목은 로봇 학습 코드를 이해하고 실제 시스템을 안정적으로 실행하기 위한 공통 기반이다. **단계 0과 함께 시작해 필요한 부분을 단계 2~10에서 다시 적용**한다. 이미 이수한 과목은 아래 실습과 통과 기준으로 확인하고 부족한 영역만 복습한다.

권장 선후 관계는 `프로그래밍 → 자료구조·알고리즘`, `프로그래밍·데이터 표현 → 컴퓨터 구조 → 운영체제`, `프로세스·I/O 기초 → 네트워크`다. 수학·PyTorch 학습과 병행할 수 있으며 네 과목 전체를 끝낼 때까지 첫 BC 실습을 미룰 필요는 없다.

표의 필수 범위는 개념·가정·대표 예제의 이해다. 과목별 직접 구현 범위와 심화 범위는 아래에서 구분한다. 항목은 `CS1-01`~`CS4-10`, 과목 완료 기준은 `GCS1`~`GCS4`로 체크리스트와 연결한다.

<a id="cs-1"></a>
#### CS-1. 자료구조와 알고리즘 — Data Structures and Algorithms

**목표:** 자료구조와 알고리즘을 정확성·시간·공간 비용으로 선택하고, 탐색·데이터 수집·학습 버퍼를 직접 설계한다.

**선수 지식:** 단계 0의 Python, 함수·반복문·재귀 기초. 집합·논리·수학적 귀납법·합과 점화식은 필요한 부분을 함께 복습한다.

| 항목 ID | 영역 | 반드시 알아야 할 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| CS1-01 | 정확성과 수학 기초 | 추상 자료형(Abstract Data Type, ADT), 명세(Specification), 사전·사후 조건(Precondition/Postcondition), 루프 불변식(Loop Invariant), 수학적 귀납법(Mathematical Induction), 종료성(Termination) | 자료구조·탐색 구현이 올바른지 설명 |
| CS1-02 | 복잡도 분석 | 시간·공간 복잡도(Time/Space Complexity), Big-O/Big-Omega/Big-Theta, 최악·평균 경우(Worst/Average Case), 분할상환 분석(Amortized Analysis), 점화식(Recurrence), Master Theorem의 적용 조건 | 데이터 크기와 실행 비용의 관계 추정 |
| CS1-03 | 선형 자료구조 | 배열(Array), 동적 배열(Dynamic Array), 연결 리스트(Linked List), 스택(Stack), 큐(Queue), 덱(Deque), 원형 버퍼(Circular/Ring Buffer) | 관측 history·rollout·replay 저장 |
| CS1-04 | 해시와 집합 | 해시 함수(Hash Function), 해시 테이블(Hash Table), 충돌 해결(Collision Resolution), Chaining, Open Addressing, Load Factor, Set/Map | 객체·episode·설정의 빠른 조회 |
| CS1-05 | 트리와 우선순위 | 트리 순회(Tree Traversal), 이진 탐색 트리(Binary Search Tree, BST), 균형 탐색 트리(Balanced Search Tree; AVL/Red–Black 개념), 이진 힙(Binary Heap), 우선순위 큐(Priority Queue) | 탐색 frontier와 우선 작업 관리 |
| CS1-06 | 검색과 정렬 | 이진 탐색(Binary Search), 안정 정렬(Stable Sorting), In-place Algorithm, Insertion Sort, Merge Sort, Quicksort, Heapsort, Counting/Radix Sort의 적용 조건 | timestamp 정렬과 검색·알고리즘 비교 |
| CS1-07 | 그래프 표현·순회 | 방향·무방향 그래프(Directed/Undirected Graph), 인접 리스트·행렬(Adjacency List/Matrix), Breadth-first Search(BFS), Depth-first Search(DFS), Connected Components, Topological Sort | 환경 그래프와 작업 의존 관계 분석 |
| CS1-08 | 최단 경로·휴리스틱 | Dijkstra's Algorithm, Bellman–Ford Algorithm, Floyd–Warshall Algorithm, A* Search, Admissible/Consistent Heuristic, Edge Relaxation | 격자 경로와 탐색 비용 비교 |
| CS1-09 | 연결성과 신장 트리 | Disjoint-set Union/Union–Find, Path Compression, Union by Rank/Size, Minimum Spanning Tree(MST), Kruskal's Algorithm, Prim's Algorithm | 그래프 구조·연결성 처리 |
| CS1-10 | 알고리즘 설계 전략 | Recursion, Divide and Conquer, Greedy Algorithm, Dynamic Programming(DP), Memoization, Tabulation, Backtracking, Optimal Substructure, Overlapping Subproblems | 문제 구조를 이용한 계산 감소 |
| CS1-11 | 계산 가능성과 난이도 입문 | Decision/Optimization Problem, Polynomial Time, Class P, Class NP, Polynomial-time Reduction, NP-hardness, NP-completeness | 최적해 계산의 한계와 휴리스틱의 역할 이해 |

**학습 깊이:** 정확성·복잡도·대표 예제는 필수다. 구현은 배열/큐/힙·해시 사용, 정렬 하나, 그래프 탐색과 DP 예제부터 진행한다. 균형 트리의 모든 구현과 NP-completeness 증명 모음은 심화로 둔다. 알고리즘의 DP와 RL의 Bellman backup은 공통 원리가 있지만 입력·모델 가정이 다른 문제 설정임을 구별한다.

**실습:** 합성 격자에서 BFS·Dijkstra·A*를 비교한다. BFS는 같은 간선 비용, Dijkstra는 비음수 비용이라는 조건을 확인하고, A*의 휴리스틱·재방문 처리에 따른 최적성을 검증한다. 이어 작은 replay ring buffer를 만들어 wrap-around, 유효 길이, transition 정렬과 episode 경계를 확인한다.

**통과 기준(GCS1):** 자료구조별 연산 비용을 설명하고, 작은 그래프의 기준 해와 탐색 결과를 대조한다. 입력 크기에 따른 실행 시간과 확장 노드 수를 기록하며, replay buffer가 덮어쓰기·샘플링·episode 경계를 올바르게 처리함을 보인다.

주자료: [MIT 6.006 Introduction to Algorithms](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/)의 자료구조·정렬·그래프·DP 강의와 문제. A*는 기존 단계 4의 계획 자료와 연결한다.

<a id="cs-2"></a>
#### CS-2. 컴퓨터 구조 — Computer Architecture

**목표:** 프로그램과 tensor 연산이 CPU·메모리·GPU에서 실행되는 과정을 이해하고 실제 학습·추론 병목을 측정한다.

**선수 지식:** 단계 0의 프로그래밍·배열. C의 포인터(Pointer), 주소(Address), 배열, 구조체(Struct), 메모리 수명은 짧은 예제로 보완한다.

| 항목 ID | 영역 | 반드시 알아야 할 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| CS2-01 | 수와 데이터 표현 | Binary/Hexadecimal Representation, Signed/Unsigned Integer, Two's Complement, Integer Overflow, Endianness, IEEE 754 Floating-point, Rounding, NaN/Infinity, FP32/FP16/BF16의 정밀도·범위 | 센서 자료형과 학습 수치 오차 이해 |
| CS2-02 | 디지털 논리 | Boolean Algebra, Logic Gate, Combinational/Sequential Logic, Multiplexer, Adder, Flip-flop, Register, Finite State Machine(FSM) | 연산·상태 저장의 하드웨어 원리 |
| CS2-03 | 명령어와 CPU | Stored-program Concept, Instruction Set Architecture(ISA), Assembly Language, Addressing Mode, Register File, Arithmetic Logic Unit(ALU), Datapath, Control Unit, Fetch–Decode–Execute | 고수준 코드와 명령 실행 연결 |
| CS2-04 | 프로그램의 기계 표현 | Compile–Assemble–Link–Load, Machine Code, Calling Convention, Call Stack, Stack Frame, Pointer Arithmetic, Memory Alignment | 배열·함수 호출·메모리 접근 비용 이해 |
| CS2-05 | 파이프라인 | Instruction Pipelining, Instruction-level Parallelism, Structural/Data/Control Hazard, Forwarding, Stall, Branch Prediction | 명령 처리량과 의존성 분석 |
| CS2-06 | 메모리 계층 | Register/Cache/DRAM/Storage Hierarchy, Cache Line, Temporal/Spatial Locality, Cache Hit/Miss, Set Associativity, Write-through/Write-back, Memory Latency/Bandwidth | 영상·배치 데이터 이동의 병목 진단 |
| CS2-07 | 성능 분석 | CPU Time, Clock Rate, Cycles per Instruction(CPI), Latency versus Throughput, Amdahl's Law, Profiling, Compute-bound/Memory-bound Workload | 최적화할 계층을 측정으로 결정 |
| CS2-08 | 병렬 하드웨어 | Multicore, SIMD, SIMT, GPU Thread/Block 개념, Memory Coalescing, Cache Coherence, Memory Consistency, False Sharing | CPU·GPU 병렬 연산과 메모리 상호작용 |
| CS2-09 | 장치와 데이터 이동 | Interrupt, Polling, Direct Memory Access(DMA), Memory-mapped I/O, Device Driver, Host–Device Transfer, Accelerator Synchronization | 카메라·센서·GPU 전송 경로 이해 |

**학습 깊이:** 디지털 논리·ISA·메모리·병렬성의 원리와 코드 수준 분석은 필수다. CPU 전체 설계, 상세 coherence protocol, CUDA kernel 최적화는 필요할 때 심화한다. GPU나 thread 수가 늘면 항상 빨라진다고 가정하지 않는다.

**실습:** 같은 결과를 계산하는 배열 연산을 접근 순서·contiguous layout·batch 크기별로 비교한다. 가능한 경우 CPU와 GPU를 비교하되 데이터 전송·warm-up·필요한 동기화를 포함한 측정과 연산 자체 측정을 구별한다. GPU가 없으면 CPU cache·배열 접근 실험으로 완료할 수 있다.

**통과 기준(GCS2):** 정수·부동소수점 표현과 cache locality를 설명한다. 동일한 수치 결과 또는 허용 오차를 확인하면서 실행 시간·메모리 사용량을 측정하고, 연산량·메모리 접근·전송 중 병목을 증거로 구분한다. 처리량과 단일 제어 요청의 지연을 따로 보고한다.

주자료: [MIT 6.004 Computation Structures](https://ocw.mit.edu/courses/6-004-computation-structures-spring-2017/)의 디지털 논리·ISA·파이프라인·cache. 보충: [CS:APP 공식 사이트](https://csapp.cs.cmu.edu/)의 데이터 표현·기계 수준 프로그램·메모리 계층 자료. CS:APP 웹 보조자료와 교재 본문은 구별하며, 교재는 도서관 등에서 확보한다.

<a id="cs-3"></a>
#### CS-3. 운영체제 — Operating Systems

**목표:** 로봇의 센서 수집·추론·저장 프로세스가 자원을 공유하는 방식을 이해하고 지연·동기화·메모리·종료 문제를 진단한다.

**선수 지식:** 프로세스 실행·파일 입출력·Git 등 단계 0의 도구 사용. CS-2의 주소·메모리·interrupt 기초를 익힌 뒤 심화하되 병행 가능하다.

| 항목 ID | 영역 | 반드시 알아야 할 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| CS3-01 | OS와 실행 경계 | Kernel/User Mode, System Call, Trap/Exception, Interrupt, Protection, Resource Management | 프로그램·커널·장치 드라이버 역할 구별 |
| CS3-02 | 프로세스와 스레드 | Process, Thread, Process State, Context Switch, Process Creation, fork/exec와 spawn 개념, Signal, Exit Status | 수집·학습 worker의 실행·종료 이해 |
| CS3-03 | 스케줄링 | Preemptive/Non-preemptive Scheduling, Round Robin, Priority Scheduling, Multi-level Feedback Queue(MLFQ), CPU Affinity, Starvation, Priority Inversion | 처리량과 제어 응답성의 차이 분석 |
| CS3-04 | 가상 메모리 | Virtual/Physical Address, Address Space, Paging, Page Table, Translation Lookaside Buffer(TLB), Page Fault, Demand Paging, Copy-on-Write, Swapping/Thrashing | worker 메모리·page fault·메모리 부족 진단 |
| CS3-05 | 메모리 관리 | Stack/Heap Allocation, Allocation/Deallocation, Internal/External Fragmentation, Memory Leak, Memory Mapping(mmap), Page Replacement(FIFO/LRU/Clock) | 대규모 데이터셋 로딩과 버퍼 관리 |
| CS3-06 | 동시성·동기화 | Concurrency versus Parallelism, Race Condition, Critical Section, Atomic Operation, Mutex, Semaphore, Condition Variable, Monitor, Memory Ordering | producer–consumer와 공유 상태의 정확성 |
| CS3-07 | 교착과 진행성 | Deadlock, Coffman Conditions, Deadlock Prevention/Avoidance/Detection, Livelock, Starvation, Lock Ordering | 멈춘 데이터 수집·worker 원인 분석 |
| CS3-08 | 프로세스 간 통신·I/O | Inter-process Communication(IPC), Pipe, Message Queue, Shared Memory, Socket, Blocking/Nonblocking I/O, Synchronous/Asynchronous I/O, Bounded Buffer, Backpressure | 센서·추론·저장 단계의 연결 |
| CS3-09 | 파일·지속성·권한 | File Descriptor, File System, Directory, Metadata/Inode, Buffer/Page Cache, Flush versus Durable Write, Journaling, Crash Consistency, Permissions | 시연·체크포인트 손실과 접근 오류 이해 |
| CS3-10 | 시간·실시간 실행 | Monotonic/Wall Clock, Timer, Deadline, Jitter, Hard/Soft Real-time, Worst-case Execution Time(WCET), Graceful Shutdown, Resource Cleanup | 지연 측정과 deadline 초과 시 처리 |

**학습 깊이:** 운영체제를 구현하기보다 프로세스·메모리·동기화·I/O의 동작을 설명하고 사용자 프로그램에서 검증하는 수준을 우선한다. 커널 수정·RTOS 포팅·실시간 스케줄링 증명은 심화다. 언어 런타임의 제약을 확인하며 Python thread 수와 실제 CPU 병렬 실행을 동일시하지 않는다.

**실습:** 합성 카메라 frame을 만드는 producer와 처리·저장 consumer를 bounded queue로 연결한다. 처리 속도를 일부러 낮추고 block/drop-oldest/drop-newest 정책을 비교한다. 정상 종료와 worker 오류를 시험하고 queue 길이·메모리·처리 지연을 기록한다. 실제 로봇 제어 명령 없이 먼저 검증한다.

**통과 기준(GCS3):** race condition을 재현·수정하고 프로세스·스레드·IPC 선택 이유를 설명한다. 느린 consumer에서도 queue·메모리가 무제한 증가하지 않으며 worker 종료 후 자원이 정리됨을 확인한다. 지연 분포와 deadline miss를 기록하고, 일반 OS·sleep·평균 FPS만으로 hard real-time을 보장하지 않는다.

주교재: [Operating Systems: Three Easy Pieces, OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/)의 Virtualization·Concurrency·Persistence와 공개 과제. 필요한 시스템 프로그래밍은 [CS:APP](https://csapp.cs.cmu.edu/)의 프로세스·가상 메모리·I/O·동시성 자료로 보충한다.

<a id="cs-4"></a>
#### CS-4. 컴퓨터 네트워크 — Computer Networks

**목표:** 로봇·PC·학습 서버 사이의 데이터 전달을 이해하고 지연·누락·연결 실패를 계층별로 진단한다.

**선수 지식:** 단계 0의 프로그램 실행과 파일/바이트 처리, CS-3의 프로세스·I/O 기초. 처음에는 localhost에서 실습한다.

| 항목 ID | 영역 | 반드시 알아야 할 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| CS4-01 | 계층과 통신 모델 | OSI/TCP-IP Layering, Encapsulation/Decapsulation, Packet Switching, Client–Server, Peer-to-Peer, End-to-end Principle | 연결 문제를 계층별로 분해 |
| CS4-02 | 성능과 시간 | Bandwidth, Throughput, Goodput, Transmission/Propagation/Queueing Delay, Round-trip Time(RTT), Jitter, Packet Loss, Clock Synchronization | 평균 속도와 최신 관측 도착 시간 구별 |
| CS4-03 | 링크·무선 통신 | Ethernet, MAC Address, Frame, Switch, Error Detection/CRC, Wi-Fi, Shared Medium, Wireless Interference | 로봇·PC의 유무선 링크 특성 이해 |
| CS4-04 | 주소와 네트워크 계층 | IPv4/IPv6, Subnet Mask, CIDR, IP Forwarding, Routing Table, Default Gateway, ARP(IPv4), ICMP, MTU, Fragmentation/Path MTU 개념 | 같은 LAN·다른 subnet·경로 문제 진단 |
| CS4-05 | 설정·응용 프로토콜 | DHCP, DNS, Port Number, HTTP Request/Response, Serialization/Deserialization, Data Schema | 주소 발견과 메시지 형식 정의 |
| CS4-06 | 전송 계층 | UDP Datagram, TCP Byte Stream, Connection Establishment/Termination, Sequence Number, Acknowledgment, Retransmission, Flow Control, Congestion Control, Head-of-line Blocking | 전달 보장·누락·지연의 trade-off 이해 |
| CS4-07 | Socket 프로그래밍 | Socket, bind/listen/accept/connect, Partial Read/Write, Message Framing, Length Prefix, Blocking/Nonblocking Socket, Timeout, Connection Reset | 메시지가 나뉘거나 합쳐져 수신되는 경우 처리 |
| CS4-08 | 응용 수준 신뢰성 | Application Sequence ID, Timestamp, Heartbeat, Retry, Duplicate Detection, Idempotency, Backpressure, Stale-message Rejection | 중복·오래된 관측·재연결 처리 |
| CS4-09 | 접근·보호·진단 | NAT, Firewall, Authentication/Authorization, Transport Layer Security(TLS), SSH, ping, traceroute, Socket Inspection, Packet Capture(Wireshark/tcpdump) | 허가된 장비 통신을 설정하고 패킷 흐름 진단 |
| CS4-10 | 로봇 미들웨어 연결 | ROS 2 Publish–Subscribe, Middleware Abstraction(RMW), Data Distribution Service(DDS), Discovery, Quality of Service(QoS), Reliability, History/Depth, Durability, Deadline, Lifespan, Liveliness | TurtleBot3 센서 topic의 통신 조건 이해 |

**학습 깊이:** TCP/IP 계층과 socket·지연 측정은 필수다. ROS 2/DDS 행은 네트워크 원리를 로봇에 연결하는 내용이며 설치한 ROS 배포판·RMW를 기준으로 익힌다. DDS를 모든 ROS 2 구성의 유일한 구현으로 가정하지 않는다. BGP 상세·대규모 분산 합의·네트워크 보안 심화는 선택이다.

**실습:** localhost에서 합성 센서 메시지를 TCP·UDP로 송수신한다. TCP에는 길이 기반 framing, UDP에는 sequence ID와 timeout을 넣고, 응용 계층에서 지연·누락·중복을 주입해 처리 결과를 비교한다. 이어 본인이 관리하는 TurtleBot3의 읽기 전용 telemetry를 관찰한다. 서로 다른 장치의 clock 동기화가 검증되지 않았다면 RTT와 장치 내부 처리 지연을 측정한다.

**통과 기준(GCS4):** TCP send/recv 호출과 메시지 경계가 일치하지 않는 경우를 처리한다. 재연결·누락·중복·오래된 메시지의 처리 규칙을 설명하고 RTT·처리량·누락을 기록한다. Reliable 전송이나 QoS deadline 설정을 최신성·실제 기한 준수의 보장으로 해석하지 않는다.

주자료: [Kurose & Ross 공개 강의](https://gaia.cs.umass.edu/kurose_ross/online_lectures.htm), [저자 제공 Wireshark 실습](https://gaia.cs.umass.edu/kurose_ross/wireshark.php). 공개 강의·실습과 교재 전체의 무료 제공은 다르다. 로봇 연결은 [ROS 2 공식 QoS 실습 문서 소스](https://github.com/ros2/ros2_documentation/blob/rolling/source/ROS-Framework/interfaces/topics/Working-with-topics/Quality-of-Service.rst)를 참고하고 설치 배포판에 맞는 문서를 사용한다.

<a id="stage-1"></a>
### 단계 1. 수학·확률·최적화

**목표:** 논문의 목적함수를 읽고 미분하며, 로봇의 좌표와 불확실성을 다룬다.

**반드시 알아야 할 개념**

| 항목 ID | 영역 | 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| S01-01 | 벡터 공간 | 선형 독립(Linear Independence), 기저(Basis), 차원(Dimension), 계수(Rank), 열 공간(Column Space), 영공간(Null Space), 내적(Inner Product), 노름(Norm), 직교 투영(Orthogonal Projection) | 로봇의 자유도·중복성과 최소제곱 이해 |
| S01-02 | 행렬 분해·수치 해법 | 고유값·고유벡터(Eigenvalue/Eigenvector), 고유값 분해(Eigendecomposition), 특이값 분해(Singular Value Decomposition, SVD), 양의 정부호·준정부호(Positive Definite/Semidefinite), 조건수(Condition Number), 최소제곱(Least Squares), Moore–Penrose 의사역행렬(Pseudoinverse) | 특이점과 불안정한 역행렬 계산 진단 |
| S01-03 | 다변수 미분 | 편미분(Partial Derivative), 연쇄법칙(Chain Rule), 기울기(Gradient), Jacobian, Hessian, Taylor 전개(Taylor Expansion), 행렬 미분(Matrix Calculus) | 역전파·기구학·최적화 유도 |
| S01-04 | 확률 기초 | 조건부확률(Conditional Probability), Bayes 정리, 독립·조건부 독립(Independence/Conditional Independence), 결합·주변 분포(Joint/Marginal Distribution), 기댓값(Expectation), 분산(Variance), 공분산(Covariance) | 관측 불확실성과 확률 정책 |
| S01-05 | 분포와 표본 | Bernoulli 분포, Categorical 분포, Uniform 분포, Gaussian/Multivariate Gaussian 분포, 큰 수의 법칙(Law of Large Numbers, LLN), 중심극한정리(Central Limit Theorem, CLT), Monte Carlo 추정 | 연속 행동 샘플링과 성공률 추정 |
| S01-06 | 통계 추정 | 우도(Likelihood), 최대우도추정(Maximum Likelihood Estimation, MLE), 최대사후추정(Maximum a Posteriori Estimation, MAP), 편향·분산(Bias/Variance), 신뢰구간(Confidence Interval), Bootstrap | 손실함수와 실험 불확실성 해석 |
| S01-07 | 정보이론 | 엔트로피(Entropy), 교차 엔트로피(Cross-entropy), Kullback–Leibler Divergence(KL Divergence), Jensen 부등식 | 확률 정책·SAC·변분 학습 이해 |
| S01-08 | 비제약 최적화 | 목적함수(Objective Function), 경사하강법(Gradient Descent), 확률적 경사하강법(Stochastic Gradient Descent, SGD), Momentum, Adam, 정규화(Regularization), 볼록성(Convexity) | 신경망과 정책의 파라미터 학습 |
| S01-09 | 제약 최적화 | 가능집합(Feasible Set), Lagrange Multiplier, Lagrangian, Karush–Kuhn–Tucker 조건(KKT Conditions), 이차계획법(Quadratic Programming, QP)의 기본 형태 | 관절·접촉 제약과 MPC 연결 |
| S01-10 | 미분방정식과 이산화 | 상미분방정식(Ordinary Differential Equation, ODE), 초기값 문제(Initial Value Problem), 선형화(Linearization), Euler Integration, Runge–Kutta Integration, 이산화(Discretization), 수치 안정성(Numerical Stability) | 동역학·제어·생성모델 sampling의 기초 |

**학습 깊이:** KKT·QP·ODE는 기본 형태와 의미를 이해하는 수준에서 시작한다. 측도론이나 모든 최적화 증명을 끝내는 것을 다음 단계의 조건으로 삼지 않는다.

**심화:** 확률과정·Markov chain, stochastic approximation, constrained optimization, manifold/Lie group의 미분. 측도론 전체를 시작 조건으로 삼을 필요는 없다.

**실습:** ridge regression을 유도·구현한다. 비선형 함수의 analytic/autograd/finite-difference Jacobian을 비교한다. 성공·실패 표본으로 성공률과 구간 추정을 계산한다.

**통과 기준:** 최소제곱·Gaussian log-likelihood·chain rule을 설명하고, ill-conditioned 행렬에서 역행렬을 직접 계산하는 방식의 문제를 실험으로 보인다.

주교재: [Mathematics for Machine Learning](https://mml-book.com/) 2~7장. 부족한 부분만 [MIT 18.06](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/), [Harvard Stat 110](https://stat110.hsites.harvard.edu/), [Boyd & Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)으로 보충한다.

<a id="stage-2"></a>
### 단계 2. 머신러닝·딥러닝·표현 학습

**목표:** BC와 actor/critic을 직접 학습할 수 있는 딥러닝 역량을 확보한다.

**반드시 알아야 할 개념**

| 항목 ID | 영역 | 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| S02-01 | 지도학습 문제 | 회귀(Regression), 분류(Classification), 선형회귀(Linear Regression), 로지스틱 회귀(Logistic Regression), 경험적 위험 최소화(Empirical Risk Minimization, ERM) | BC와 actor/critic 학습의 공통 기반 |
| S02-02 | 손실·확률 출력 | 평균제곱오차(Mean Squared Error, MSE), 교차 엔트로피(Cross-entropy), 음의 로그우도(Negative Log-likelihood, NLL), Gaussian NLL, 다봉성(Multimodality) | 연속 행동 회귀와 확률 정책의 손실 선택 |
| S02-03 | 일반화·모델 선택 | 과소적합·과적합(Underfitting/Overfitting), Bias–Variance Trade-off, Train/Validation/Test Split, Data Leakage, Distribution Shift, Hyperparameter Selection, Early Stopping | 좋은 학습 loss와 좋은 정책을 구별 |
| S02-04 | 전처리·정규화 | Standardization, Min–Max Scaling, Feature Normalization, Data Augmentation, L2 Regularization, Weight Decay, Dropout | 입력·행동 scale을 맞추고 과적합 제어 |
| S02-05 | 기본 신경망 | 다층 퍼셉트론(Multilayer Perceptron, MLP), Affine Layer, ReLU, Tanh, Sigmoid, Forward Pass, Backpropagation, Xavier/Glorot Initialization, He/Kaiming Initialization | 작은 정책과 가치 함수 직접 구현 |
| S02-06 | 학습 안정화 | Mini-batch SGD, Adam, AdamW, Learning-rate Schedule, Gradient Clipping, Vanishing/Exploding Gradients, Batch Normalization, Layer Normalization | loss·gradient·학습률 문제 진단 |
| S02-07 | PyTorch 자동미분 | Computational Graph, Reverse-mode Automatic Differentiation, Leaf Tensor, requires_grad, backward, Gradient Accumulation, detach, no_grad | actor와 critic에서 gradient 경로를 정확히 제어 |
| S02-08 | PyTorch 학습 구성 | nn.Module, Parameter, forward, Dataset, DataLoader, Optimizer, zero_grad, optimizer.step, state_dict, model.train, model.eval | 학습·검증·저장·복원 루프 구현 |
| S02-09 | 시각 표현 입문 | 합성곱(Convolution), Kernel, Stride, Padding, Receptive Field, Convolutional Neural Network(CNN), Residual Connection, ResNet | 시각 BC를 위한 이미지 encoder |
| S02-10 | 전이·표현 학습 입문 | Feature Representation, Pretraining, Transfer Learning, Frozen Encoder, Fine-tuning, Self-supervised Learning, Contrastive Learning | 공개 encoder의 사용 방식과 비교 조건 이해 |

**학습 깊이:** 행 1~8과 작은 MLP를 먼저 완료한다. CNN·표현 학습은 여기서 개념을 익히고 단계 8의 visual BC에서 구현을 확장한다. RNN·Transformer·CVAE·diffusion의 상세 내용은 단계 8에서 배운다.

**실습:** 선형회귀 → 작은 MLP 회귀 → 저장된 관측에서 행동을 회귀하는 작은 모델 순서로 구현한다. CNN 학습을 시작하면 이미지에서 물체의 2D 위치를 회귀한다. 데이터·모델·학습률을 각각 한 가지씩 바꾸는 ablation을 수행한다.

**통과 기준:** 작은 배치는 overfit되지만 validation이 나빠지는 이유를 진단한다. `train()`/`eval()`, gradient 흐름, checkpoint·정규화 저장을 정확하게 처리한다.

주자료: [Stanford CS229 강의노트](https://cs229.stanford.edu/main_notes.pdf)의 지도학습·일반화, [Dive into Deep Learning](https://d2l.ai/)의 PyTorch·CNN·sequence·attention. 시각 분야 보충은 [Stanford CS231n 2025](https://cs231n.stanford.edu/2025/). 세 강의를 모두 완강할 필요는 없다.

<a id="stage-3"></a>
### 단계 3. 로봇 기구학·동역학·접촉

**목표:** 정책이 출력하는 행동이 로봇에서 어떤 움직임과 힘을 만드는지 설명한다.

**반드시 알아야 할 개념**

| 항목 ID | 영역 | 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| S03-01 | 로봇 구조 | 자유도(Degrees of Freedom, DoF), Revolute Joint, Prismatic Joint, Kinematic Chain, Actuator, Encoder, End-effector, Gripper, Workspace | SO-101의 팔 관절과 gripper 자유도 구별 |
| S03-02 | 좌표·변환 | Reference Frame, World/Base/Tool/Camera Frame, Active/Passive Rotation, Homogeneous Transformation, Change of Coordinates, Transformation Composition | 관측·목표·명령의 좌표계를 일치 |
| S03-03 | 회전 표현 | Special Orthogonal Group SO(3), Rotation Matrix, Euler Angles, Axis–Angle, Unit Quaternion, Quaternion Double Cover, Rotation Geodesic Distance | 회전 오차와 표현의 특이성 이해 |
| S03-04 | 강체 운동의 기하 | Special Euclidean Group SE(3), Lie Algebra so(3)/se(3), Exponential/Logarithmic Map, Screw Axis, Twist, Wrench, Adjoint Representation | 자세·속도·힘을 일관된 표현으로 연결 |
| S03-05 | 정기구학 | Forward Kinematics(FK), Product of Exponentials(PoE), Denavit–Hartenberg Parameters(DH Parameters) | 관절값에서 tool 자세 계산; DH는 해석할 수 있는 수준 |
| S03-06 | 속도 기구학 | Space Jacobian, Body Jacobian, Differential Kinematics, Kinematic Singularity, Manipulability, Kinematic Redundancy, Null-space Motion | 도달 방향·특이점·여유 자유도 분석 |
| S03-07 | 역기구학 | Inverse Kinematics(IK), Numerical IK, Newton–Raphson Method, Jacobian Pseudoinverse, Damped Least Squares(DLS), Joint-limit Constraints, Position/Orientation Error | 가능한 목표와 불가능한 목표를 구분 |
| S03-08 | 강체 동역학 | Mass Matrix, Inertia Tensor, Lagrange Equations, Newton–Euler Equations, Coriolis/Centrifugal Terms, Gravity Compensation, Forward/Inverse Dynamics | 토크·가속도·외력의 관계 이해 |
| S03-09 | 접촉·파지 | Unilateral Contact Constraint, Coulomb Friction, Friction Cone, Contact Mode, Grasp Matrix, Grasp Wrench Space, Force Closure, Form Closure, Antipodal Grasp, Slip | 물체를 잡고 유지하는 물리 조건 이해 |
| S03-10 | 로봇 모델 파일 | Unified Robot Description Format(URDF), MuJoCo XML Format(MJCF), Visual/Collision Geometry, Inertial Parameters, Joint Limits, Actuator Model | 외형 모델과 실제 동작 모델의 차이 진단 |

**학습 깊이:** Lie group와 grasp 이론은 기본 연산·가정·기하적 의미를 필수로 익힌다. 모든 접촉 최적화와 Lie group 증명은 심화다. SO-101에 임의의 6D 자세가 모두 도달 가능하다고 가정하지 않는다.

핵심 식은 각 기호·좌표계·가정을 이해하고 유도한 뒤 구현과 연결한다.

```text
T_world_tool(q) = T_world_base · T_base_tool(q)
V = J(q) q_dot                         # 같은 frame·표현을 사용
tau_equiv = J(q)^T F                  # 같은 frame의 wrench를 등가 관절 힘으로 변환
M(q) q_ddot + C(q,q_dot) q_dot + g(q) = tau + J(q)^T F_ext
```

마지막 식의 F_ext는 환경이 로봇에 가하는 외력 wrench다. 정적 평형에서 그 외력을 상쇄하는 actuator torque는 중력과 부호를 포함해 이 식으로 계산한다. 기본적인 강체 모델이며 필요에 따라 마찰·구동기 동역학 등을 추가한다. `q`는 관절 위치, `tau`는 관절 힘/토크다. Cartesian pose 변화량을 쓰는 정책은 이를 IK나 operational-space controller를 통해 관절 명령으로 바꾼다.

**실습:** 2-link planar arm의 FK·Jacobian·IK를 직접 구현한 뒤 SO-ARM-101의 실제 관절 구성과 모델로 확장한다. 일반적인 6/7-DoF 연구용 arm은 시뮬레이터에서 추가로 익힌다. singularity 근처에서 pseudoinverse와 damping을 비교하고, quaternion을 단순히 뺄 때 생기는 문제도 확인한다.

**통과 기준:** FK를 simulator와 비교하고 오차를 수치화한다. numerical Jacobian과 해석 Jacobian이 일치하는지 확인한다. IK 실패가 도달 불가능·관절 제한·특이점·잘못된 frame 중 무엇 때문인지 분류한다.

주교재: [Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/)의 강체 운동·FK·IK·Jacobian·동역학. 접촉과 파지는 [MIT Robotic Manipulation](https://manipulation.csail.mit.edu/)을 함께 읽는다.

<a id="stage-4"></a>
### 단계 4. 제어·모션 플래닝·지각

**목표:** 학습 없이 동작하는 baseline을 만들고, 관측과 제어의 오류를 구별한다.

**반드시 알아야 할 개념**

| 항목 ID | 영역 | 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| S04-01 | 시스템 표현 | State-space Model, Equilibrium Point, Linearization, Continuous-time/Discrete-time System, Controllability, Observability | 제어 가능한 상태와 추정 가능한 상태 구별 |
| S04-02 | 피드백 제어 | Feedback/Feedforward Control, P/PD/PID Control, Tracking Error, Steady-state Error, Overshoot, Settling Time, Actuator Saturation, Integral Windup, Anti-windup | 추종 성능과 포화·지연 문제 분석 |
| S04-03 | 모델 기반 제어·안정성 | Gravity Compensation, Computed-torque Control, Linear Quadratic Regulator(LQR), Riccati Equation, Lyapunov Stability, Lyapunov Function | 학습 정책의 제어 baseline과 안정성 직관 |
| S04-04 | Task-space·접촉 제어 | Joint-space Control, Task-space Control, Operational-space Control, Impedance Control, Admittance Control, Stiffness, Damping, Hybrid Motion–Force Control | 위치 명령과 힘·접촉 반응의 차이 이해 |
| S04-05 | 경로 계획 | Configuration Space(C-space), Free/Obstacle Space, Collision Checking, Probabilistic Roadmap(PRM), Rapidly-exploring Random Tree(RRT), RRT-Connect | 충돌 없는 관절 경로 생성 |
| S04-06 | 궤적·최적제어 | Path versus Trajectory, Trajectory Interpolation, Time Parameterization, Trajectory Optimization, Model Predictive Control(MPC), Receding Horizon | 속도·가속도·실시간 재계획을 고려 |
| S04-07 | 카메라 기하 | Pinhole Camera Model, Camera Intrinsics/Extrinsics, Radial/Tangential Distortion, Camera Calibration, Hand–Eye Calibration, Reprojection Error | 픽셀과 로봇 base 좌표의 연결 |
| S04-08 | 3D 지각 | RGB-D, Depth Back-projection, Point Cloud, Object Segmentation, Perspective-n-Point(PnP), Iterative Closest Point(ICP), Rigid Registration, 6D Object Pose | 물체 위치·자세 추정; task에 필요한 범위로 구현 |
| S04-09 | 상태 추정 | Bayesian Filtering, Kalman Filter(KF), Extended Kalman Filter(EKF), Process Noise, Measurement Noise, Sensor Fusion | 불완전하고 잡음 있는 센서 정보 해석 |
| S04-10 | 시간·실행 계층 | Sampling Period, Zero-order Hold(ZOH), Control Frequency, Policy Frequency, Latency, Jitter, Timestamp Synchronization | 학습·센서·제어 주기의 차이 진단 |

**학습 깊이:** LQR·Lyapunov·KF/EKF·MPC는 식의 의미와 작은 예제를 필수로 익히고, 비선형 최적제어의 상세 유도는 단계 9나 심화 과정으로 연결한다. 토크·힘·impedance 제어는 지원되는 시뮬레이터에서 실습하며 SO-101의 표준 position interface와 구분한다.

**실습:** 물체 위치가 알려진 pick-and-place를 IK·계획·PD/impedance로 수행한다. 이후 카메라 추정 위치를 사용하고, 지각 오차와 제어 오차를 따로 측정한다. 물체를 집고 유지한 뒤 지정 위치에 놓는 전체 성공 조건을 정의한다.

**통과 기준:** 학습 정책이 없더라도 기본 task가 수행된다. gain·latency·mass를 바꾸었을 때 추종 오차와 진동의 변화가 설명된다. 실제 물체 크기와 작업 허용오차에 맞춰 위치·회전 오차 기준을 정한다.

주자료: [MIT Robotic Manipulation](https://manipulation.csail.mit.edu/), [Underactuated Robotics](https://underactuated.mit.edu/)의 LQR·Lyapunov·trajectory optimization, [Planning Algorithms](https://lavalle.pl/planning/), [Szeliski, Computer Vision](https://szeliski.org/Book/)의 camera geometry. 계획·비전 교재 전체가 이 단계의 필수 범위는 아니다.

<a id="stage-5"></a>
### 단계 5. 강화학습의 원리와 연속 제어

**목표:** PPO와 SAC의 데이터 흐름·목적함수·실패 원인을 설명하고 작은 환경에서 구현한다.

**반드시 알아야 할 개념**

| 항목 ID | 영역 | 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| S05-01 | 순차 의사결정 | Markov Decision Process(MDP), Partially Observable MDP(POMDP), Markov Property, State, Observation, Action, Transition Kernel, Reward, Policy, Return, Discount Factor, Horizon | task를 학습 문제로 정의 |
| S05-02 | 가치 함수 | State-value Function V, Action-value Function Q, Advantage Function A, Bellman Expectation Equation, Bellman Optimality Equation, Bellman Operator, Contraction Mapping | 가치 추정과 정책 개선의 관계 이해 |
| S05-03 | 표 기반 학습 | Policy Evaluation, Policy Improvement, Policy Iteration, Value Iteration, Monte Carlo Prediction, Temporal-difference Learning TD(0), n-step Return, TD(lambda), SARSA, Q-learning | 정확한 해가 있는 작은 환경에서 원리 검증 |
| S05-04 | 학습 데이터·탐색 | On-policy/Off-policy Learning, Exploration–Exploitation Trade-off, Epsilon-greedy, Importance Sampling, Bootstrapping, Function Approximation, Deadly Triad | 상호작용과 재사용 데이터의 가정 구별 |
| S05-05 | 심층 가치 학습 | Deep Q-Network(DQN), Experience Replay, Target Network, Double Q-learning, Overestimation Bias | 연속 제어 actor–critic의 안정화 장치 이해 |
| S05-06 | 정책 기울기 | Policy Gradient Theorem, Likelihood-ratio/Score-function Estimator, REINFORCE, Reward-to-go, Baseline, Actor–Critic, Generalized Advantage Estimation(GAE) | 정책 목적함수와 gradient 유도 |
| S05-07 | PPO | Proximal Policy Optimization(PPO), Importance Probability Ratio, Clipped Surrogate Objective, Rollout Buffer, Advantage Normalization, Entropy Bonus, Approximate KL | 수집 정책에서 지나치게 벗어나는 update 감시 |
| S05-08 | 결정론적 연속 제어 | Deterministic Policy Gradient(DPG), Deep Deterministic Policy Gradient(DDPG), Twin Delayed DDPG(TD3), Clipped Double Q-learning, Delayed Policy Update, Target Policy Smoothing | Q 오차와 actor update의 상호작용 이해 |
| S05-09 | SAC | Soft Actor-Critic(SAC), Maximum-entropy RL, Soft Bellman Backup, Stochastic Actor, Twin Critics, Entropy Temperature, Automatic Temperature Tuning, Reparameterization Trick | 연속 행동의 탐색과 데이터 재사용 |
| S05-10 | 연속 행동·target 구현 | Squashed Gaussian Policy, Tanh Change-of-variables Log-probability Correction, Action Rescaling, Polyak/Soft Target Update, Replay Warm-up, Update-to-data Ratio(UTD) | 수식과 실제 actor/critic 코드를 일치 |
| S05-11 | 종료 의미 | Terminal State, Termination, Time-limit Truncation, Bootstrap Mask, Final Observation, Vectorized Environment Auto-reset | 잘못된 value target과 episode 혼합 방지 |

**학습 깊이:** 이론 흐름은 MDP·tabular → policy gradient·actor–critic → PPO/SAC 순서다. DQN·DDPG·TD3의 작동 원리는 필수지만 전부 처음부터 구현할 필요는 없다. PPO 또는 SAC 하나를 직접 구현하고 다른 하나는 검증된 구현과 비교한다.

SAC에서는 bounded continuous action을 위한 tanh transform과 log-probability 보정을 공부한다. entropy·reward scale·action scale이 서로 미치는 영향도 확인한다. PPO는 수집 정책과 업데이트 정책의 차이를 제한하는 방식이며, 임의의 오래된 replay를 그대로 쓰는 off-policy 알고리즘으로 취급하지 않는다.

**직접 구현:** tabular value iteration/Q-learning, REINFORCE, 작은 PPO 또는 SAC 중 하나. 나머지 하나는 검증된 구현을 읽고 핵심 update를 수정해 본다. DQN·DDPG·TD3까지 모두 처음부터 작성할 필요는 없다.

**실습:** 작은 tabular 환경 → Pendulum 계열 → 간단한 reaching. state 관측과 작은 네트워크로 시작한다. random policy와 단순 제어 baseline을 함께 평가한다.

**통과 기준:** replay/rollout buffer, target, actor loss, critic loss를 코드와 식으로 연결한다. 적어도 3개 학습 seed로 학습 추세와 최종 평가를 보고한다. 개선이 없으면 reward·action·termination·normalization·gradient부터 진단한다.

중요한 구현 사항: episode reset은 `terminated or truncated`에서 수행하지만, 일반적인 외부 시간 제한에 의한 truncation은 value bootstrap을 제거할 이유가 아니다. 실제 finite-horizon task의 종료 의미와 최종 관측을 구별한다. [Gymnasium 시간 제한 문서](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

주자료: [Sutton & Barto, Reinforcement Learning](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)의 Open Access 자료, [Berkeley CS285 2023](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/)의 policy gradient·actor–critic·Q-learning. 논문은 7절의 PPO·SAC·TD3를 참고한다.

<a id="stage-6"></a>
### 단계 6. 모방학습·시연 데이터·DAgger

**목표:** 시연을 올바르게 수집·가공하고, BC가 실패하는 이유를 closed-loop 실험으로 설명한다.

**반드시 알아야 할 개념**

| 항목 ID | 영역 | 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| S06-01 | 시연의 구조 | Demonstration, Trajectory/Episode, Observation–Action Pair, Proprioception, Observation History, Action Label, Timestamp Alignment | 로봇 로그를 학습 데이터로 정의 |
| S06-02 | 기본 모방학습 | Behavior Cloning(BC), Supervised Policy Learning, MSE Regression, Maximum-likelihood Policy Estimation, Gaussian Policy, Mixture Density Network(MDN) | 시연 행동의 평균과 분포 학습 |
| S06-03 | 순차 분포 변화 | Covariate Shift, Expert/Learner State Distribution, Compounding Error, State-distribution Coverage, Recovery Demonstration | offline loss가 rollout 성공률과 다른 이유 분석 |
| S06-04 | 상호작용형 IL | Dataset Aggregation(DAgger), Expert Oracle, Expert Query Budget, Learner Rollout, Dataset Aggregation, Policy Mixing | 정책이 방문한 상태의 추가 라벨 수집 |
| S06-05 | 부분 관측·인과 문제 | Partial Observability, History-conditioned Policy, Hidden State, Causal Confusion, Privileged Information | 실행 때 없는 정보나 잘못된 상관에 의존하는 정책 진단 |
| S06-06 | 데이터 품질 | Demonstrator Variability, Multimodal Action Distribution, Idle Frames, Failed Demonstration, Corrective Demonstration, Temporal Misalignment | 수집·정제 방식의 효과 분석 |
| S06-07 | 데이터 분할 | Episode-level Split, Session-level Split, Object-instance Holdout, Overlapping-window Leakage, Train-only Normalization | 인접 프레임과 수집 조건의 누수 방지 |
| S06-08 | 실행 평가 | Open-loop Action Prediction, Closed-loop Rollout, Action Prediction Error, Task Success Rate, Autonomous/Assisted Success | 저장 데이터 평가와 실제 자율 수행 구별 |

**학습 깊이:** DAgger를 위한 전문가 라벨은 일반 offline 데이터 파일에 자동으로 존재하지 않는다. SO-101의 leader-follower 경로에서는 기록된 action이 조작자의 목표 명령인지 follower의 측정 위치인지 반드시 확인한다.

시연 데이터는 `D = {(o_t, a_t)}`로 나타낼 수 있고, history 기반이면 입력이 `h_t = (o_{t-k:t}, a_{t-k:t-1})` 등이 된다. 미래 관측을 실행 시점의 입력에 넣지 않는다.

```text
BC:      minimize E_(h,a)~D [ -log pi_theta(a | h) ]
MSE BC:  minimize E_(h,a)~D [ ||f_theta(h) - a||^2 ]
```

MSE는 행동 분포가 여러 봉우리를 가질 때 두 유효 동작 사이의 무효 동작을 만들 수 있다. validation action loss가 낮다고 물체를 안정적으로 집는다는 뜻은 아니다.

**시뮬레이션 실습:** 공개 manipulation 데이터로 작은 MLP BC를 만들고 scripted expert가 가능한 reaching 과제에서 BC와 DAgger를 비교한다. DAgger는 learner가 방문한 상태에서 정답 행동을 줄 expert가 있어야 한다. 고정된 시연 파일만으로 일반적인 DAgger를 실행할 수는 없다.

**SO-ARM-101 실습:** leader·follower 캘리브레이션 → 원격 조작 확인 → 단일 물체 이동 시연 수집 → episode 재생·검사 → train/validation/test 분리 → 작은 BC baseline. 카메라가 있다면 RGB와 joint state를 함께 기록한다. 카메라가 없다면 먼저 관절 상태·동작 기록을 검증하고 시각 IL은 공개 데이터로 시작한다. 물체 위치가 관측에 없는 joint-only 정책은 변하는 물체 위치에 일반화하기 어렵다.

**통과 기준:** 한 episode의 모든 field·shape·단위·주기·행동 의미를 설명한다. episode 단위로 분할하고 train 데이터만으로 정규화한다. 정책을 실제로 실행해 성공률·실패 영상·초기 상태를 남긴다. 복구 시연을 추가했을 때 어떤 실패가 줄었는지 비교한다.

자료: [robomimic](https://robomimic.github.io/docs/), [DAgger 원논문](https://arxiv.org/abs/1011.0686), [LeRobot 공식 문서](https://huggingface.co/docs/lerobot/index).

<a id="stage-7"></a>
### 단계 7. 매니퓰레이터 강화학습

**목표:** RL 알고리즘을 로봇의 보상·행동·접촉·리셋 설계와 연결한다.

**반드시 알아야 할 개념**

| 항목 ID | 영역 | 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| S07-01 | 조작 task 구성 | Reaching, Pushing, Grasping, Lifting, Pick-and-place, Initial-state Distribution, Reset Distribution, Success Predicate | task의 난이도·시작 조건·성공을 명시 |
| S07-02 | 목표 조건화 | Goal-conditioned Policy, Goal-conditioned Value Function, Desired Goal, Achieved Goal, Goal Distribution | 같은 정책으로 여러 목표 처리 |
| S07-03 | 보상 설계 | Sparse/Dense Reward, Reward Scaling, Reward Shaping, Potential-based Reward Shaping, Reward Hacking, Success Detection | 보상을 얻는 행동과 실제 성공 구별 |
| S07-04 | HER | Hindsight Experience Replay(HER), Goal Relabeling, Future-goal Sampling, Reward Recalculation, Goal-dependent Termination | 실패 경험 재사용의 가정·label 일관성 검증 |
| S07-05 | 행동·제어 표현 | Joint-position/Velocity/Torque Action, Absolute/Delta Action, End-effector Pose Action, Base/Tool-frame Action, Gripper Command, Action Bounds | 정책 출력과 내부 controller의 관계 분석 |
| S07-06 | 관측 설계 | Proprioceptive Observation, Object-state Observation, Visual Observation, Privileged State, Observation History, Sensor Noise, Observation/Action Delay | 시뮬레이터 정보와 실제 획득 정보 구별 |
| S07-07 | 탐색·교육과정 | Curriculum Learning, Goal Curriculum, Initial-state Curriculum, Demonstration Initialization, Exploration Noise | 희소 성공 task의 학습 경로 구성 |
| S07-08 | 시뮬레이션 실행 | Physics Timestep, Control Decimation/Action Repeat, Contact Solver, Vectorized Simulation, Domain Randomization | 물리·제어·학습 주기의 차이 이해 |
| S07-09 | 자원·일반화 | Sample Efficiency, Wall-clock Efficiency, Environment Interaction Budget, ID/OOD Evaluation, Goal Interpolation/Extrapolation | 학습 비용과 목표 일반화 범위 평가 |

**학습 깊이:** joint torque와 6D end-effector action은 알고리즘·시뮬레이터에서 배울 표현이다. 실제 SO-101에서 가능한 명령·도달 자세와 동일시하지 않는다.

Potential-based shaping의 대표 형태는 `F(s,s') = gamma * Phi(s') - Phi(s)`이다. 정책 보존 결과에는 가정과 terminal 처리가 있으므로 임의의 distance bonus가 항상 원래 문제와 같다고 생각하지 않는다. HER 역시 goal 재라벨링의 정당성, reward/종료 재계산, task 구조를 확인해야 한다.

**실습:** 하나의 goal-conditioned 시뮬레이션 과제에서 sparse SAC, SAC+HER, dense reward baseline을 비교한다. 우선 두 조건으로 pilot을 하고, 작동을 확인한 후 비교를 확장한다. BC도 같은 task의 기준선으로 둔다.

**통과 기준:** 성공률·환경 상호작용 수·실행 시간·학습 seed별 변동을 함께 보고한다. 성공으로 오판된 grasp, table collision, 물체를 튕겨 보내는 reward hacking 등을 검사한다. 알고리즘이 개선되지 않아도 원인을 증거로 좁힐 수 있어야 한다.

**보유 장비와의 연결:** 이 단계의 대규모 탐색은 시뮬레이터에서 수행한다. Fetch/Panda 등의 정책은 SO-ARM-101에 그대로 실행할 수 있는 파일이 아니다. 기구·관측·행동·제어·데이터가 다르므로 SO-ARM-101의 실물 IL 경로를 병행하고, 맞춤 sim-to-real은 별도 연구 과제로 둔다.

자료: [Gymnasium-Robotics Fetch](https://robotics.farama.org/envs/fetch/pick_and_place/), [HER](https://arxiv.org/abs/1707.01495), [robosuite controllers](https://robosuite.ai/docs/modules/controllers.html). 문서 예제와 설치 환경의 task 버전이 다를 수 있으므로 registry와 package version을 확인한다.

<a id="stage-8"></a>
### 단계 8. 시각·시퀀스·생성형 모방학습

**목표:** visual BC와 ACT/Diffusion Policy를 이해하고, 데이터와 추론 지연까지 고려해 평가한다.

**반드시 알아야 할 개념**

| 항목 ID | 영역 | 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| S08-01 | 시각 모방학습 | Visual Behavior Cloning, CNN/ResNet Encoder, Image Normalization, Random Crop/Color Augmentation, Proprioceptive Fusion, Frozen Encoder, Fine-tuning | 영상과 관절 상태로 행동 예측 |
| S08-02 | 시간 정보 | Frame Stacking, Recurrent Neural Network(RNN), Long Short-Term Memory(LSTM), Recurrent BC(BC-RNN), Hidden-state Reset, Sequence Batching, Padding Mask | 부분 관측과 episode 경계 처리 |
| S08-03 | Attention | Scaled Dot-product Attention, Self-attention, Cross-attention, Multi-head Attention, Positional Encoding, Transformer Encoder/Decoder, Causal Mask | 관측과 행동 시퀀스의 의존 관계 모델링 |
| S08-04 | 잠재변수 모델 | Latent Variable, Variational Autoencoder(VAE), Conditional VAE(CVAE), Evidence Lower Bound(ELBO), KL Regularization, Posterior/Conditional Prior, Reparameterization Trick | 다양한 시연 행동과 ACT의 학습 구조 이해 |
| S08-05 | ACT | Action Chunking with Transformers(ACT), Action Chunking, Conditional Action-sequence Prediction, Temporal Ensembling, Chunk Boundary | 미래 행동 묶음 생성과 실행 방식 이해 |
| S08-06 | Diffusion | Denoising Diffusion Probabilistic Model(DDPM), Forward Noising Process, Noise Schedule, Conditional Denoising, Noise Prediction, Score Function, Reverse Sampling | 행동 분포를 점진적으로 생성하는 원리 |
| S08-07 | Diffusion Policy | Conditional Action Diffusion, Observation Horizon, Prediction Horizon, Action Execution Horizon, Receding-horizon Control, Multimodal Action Generation | 정책의 예측 범위와 실제 실행 길이 구별 |
| S08-08 | Flow matching 입문 | Conditional Flow Matching, Probability Path, Velocity Field, Vector-field Regression, ODE Sampling | diffusion과 다른 연속 생성 방법의 기본 원리 |
| S08-09 | 실시간 추론 | Inference Latency, Latency Jitter, Stale Observation, Synchronous/Asynchronous Inference, Open-loop Execution Length, Action Discontinuity | 생성 품질과 제어 응답성의 trade-off 분석 |

**학습 깊이:** 표의 방법들은 개념과 데이터 흐름을 이해할 필수 범위다. 실습은 visual BC 뒤 ACT 또는 Diffusion Policy 하나부터 수행한다. Causal mask와 padding mask는 다른 목적이며, 모든 ACT 입력에 같은 causal mask를 적용하는 것은 아니다. Flow matching은 toy 예제로 익히고 대형 VLA 학습은 선택으로 둔다.

**반드시 구별:** action chunking은 시간적 출력 구조, diffusion/flow는 생성 방법, VLA는 시각·언어·행동의 모델 구성이다. 서로 배타적인 분류가 아니며 함께 쓰일 수 있다.

**실습:** 같은 데이터와 task로 작은 BC/BC-RNN을 만든 뒤 **ACT 또는 Diffusion Policy 하나를 먼저 재현**한다. SO-ARM-101 실물 경로는 LeRobot에서 현재 지원하는 ACT 학습·실행 흐름을 먼저 확인하는 것이 자연스럽다. 두 모델을 모두 대규모 학습하는 것은 초기 필수 과제가 아니다.

**비교 변수:** 시연 수, execution horizon, history 길이, camera view, encoder freeze/fine-tune 중 한 가지를 고른다. 다른 조건은 고정하고 평균 성공률뿐 아니라 latency·복구 능력·행동 진동을 비교한다.

**통과 기준:** prediction horizon과 실제 open-loop 실행 길이를 구별한다. 수집·학습·추론의 action scaling·camera preprocessing·control frequency가 일치한다. 학습 loss가 좋아졌지만 rollout이 나빠진 사례를 분석한다.

자료: [ACT 논문](https://arxiv.org/abs/2304.13705), [Diffusion Policy 프로젝트](https://diffusion-policy.cs.columbia.edu/), [Flow Matching](https://arxiv.org/abs/2210.02747). 각 알고리즘의 가정과 계산 비용을 배우기 위한 대표 자료이며 최신 성능 순위를 뜻하지 않는다.

<a id="stage-9"></a>
### 단계 9. Offline RL·RL+IL·model-based learning

**목표:** 시연을 따라 하는 것과 기존 데이터를 이용해 정책을 개선하는 것의 차이를 이해한다.

**반드시 알아야 할 개념**

| 항목 ID | 영역 | 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| S09-01 | Offline RL 문제 | Offline/Batch Reinforcement Learning, Behavior Policy, Dataset Support/Coverage, Distribution Shift, Extrapolation Error, Out-of-distribution Action | 데이터 밖 행동의 가치 추정 문제 이해 |
| S09-02 | 보수적·제약적 학습 | Conservative Q-Learning(CQL), Conservative Value Estimation, Value Regularization, Behavior/Policy Constraint | Q 과대평가 억제와 과도한 보수성의 비용 |
| S09-03 | IQL | Implicit Q-Learning(IQL), Expectile Regression, Asymmetric Squared Loss, Value/Q Fitting, Advantage-weighted Behavior Cloning | 분포 밖 행동의 직접 평가를 피하는 학습 구조 |
| S09-04 | Offline-to-online | BC Initialization, Demonstration Replay, Replay Mixing Ratio, Advantage-weighted Actor–Critic(AWAC), Reinforcement Learning with Prior Data(RLPD), Online Fine-tuning | 시연과 추가 상호작용의 연결 |
| S09-05 | 보상·정책 평가 | Reward Annotation, Terminal Label, Reward Classifier, False Positive/Negative, Offline Policy Evaluation(OPE), Importance-sampling OPE, Fitted Q Evaluation(FQE) | 보상 오류와 offline 평가의 한계 이해 |
| S09-06 | 학습 동역학 | Learned Dynamics Model, One-step/Multi-step Prediction Error, Model Ensemble, Epistemic/Aleatoric Uncertainty, Compounding Model Error, Model Exploitation | 좋은 예측 loss가 좋은 제어를 보장하지 않는 이유 |
| S09-07 | 모델 기반 계획 | Model-based RL, Model Predictive Control(MPC), Random Shooting, Cross-entropy Method(CEM), Probabilistic Ensembles with Trajectory Sampling(PETS) | 불확실한 동역학에서 행동 후보 평가 |
| S09-08 | 잔차·적응 | Residual Reinforcement Learning, Residual Action, Baseline Controller/Policy, Action Saturation, System Identification, Domain Randomization | 기존 제어와 학습 보정을 결합 |

**학습 깊이:** CQL·IQL은 둘 다 개념을 배우고 하나를 재현한다. OPE의 importance sampling과 FQE는 지원집합·분포 변화에 따른 한계를 이해하는 수준으로 시작한다. PETS·residual RL의 전체 구현은 연구 방향에 따라 추가한다.

**실습 A:** 같은 reward-labeled simulation 데이터에서 BC와 IQL 또는 CQL을 비교한다. expert-only와 mixed-quality 데이터 조건을 비교한다.

**실습 B:** BC, scratch RL, demonstration을 활용한 RL을 같은 online interaction budget으로 비교한다. 사람의 데이터 수집·개입·reset 비용도 기록한다.

**실습 C — 심화:** 작은 환경에서 dynamics ensemble+MPC를 만들고, 모델의 one-step loss와 실제 rollout 성능이 어긋나는 사례를 찾는다. A를 먼저 완료하고 B/C는 연구 방향에 따라 확장한다.

**통과 기준:** BC가 offline RL보다 좋은 결과도 설명한다. 데이터 coverage와 reward 품질을 확인하지 않고 offline RL이 항상 시연을 능가한다고 주장하지 않는다. IQL/CQL 등 하나의 update를 식과 코드로 설명한다.

자료: [CQL](https://arxiv.org/abs/2006.04779), [IQL](https://arxiv.org/abs/2110.06169), [AWAC](https://arxiv.org/abs/2006.09359), [PETS](https://arxiv.org/abs/1805.12114). 실제 robot RL의 시스템 사례는 [HIL-SERL](https://hil-serl.github.io/)에서 읽되, 논문의 장비·제어·인간 개입 조건을 SO-ARM-101에 그대로 가정하지 않는다.

<a id="stage-10"></a>
### 단계 10. SO-ARM-101 시스템 통합과 실제 평가

**목표:** 실제 장비에서 데이터를 수집하고 자율 정책을 평가하며, 실패가 어느 계층에서 발생했는지 찾는다.

**반드시 알아야 할 개념**

| 항목 ID | 영역 | 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| S10-01 | 장비·캘리브레이션 | Leader/Follower Teleoperation, Motor ID, Serial Communication, Calibration, Joint Zero Offset, Joint Range, Calibration Identifier | 장비 세션과 데이터·정책의 기준을 일치 |
| S10-02 | 명령과 관측 | Position Setpoint, Measured Joint Position, Commanded versus Executed Action, Joint Ordering, Action Normalization/Denormalization, Gripper Scaling | 목표 명령과 실제 움직임의 차이 분석 |
| S10-03 | 데이터·정책 파이프라인 | Observation/Action Schema, Data Recording, Dataset Visualization, Policy Checkpoint, Preprocessing/Postprocessing, Policy Rollout | 수집·학습·실행의 인터페이스 연결 |
| S10-04 | 시간 동기화 | Sensor Timestamp, Clock Offset, Sampling Frequency, End-to-end Latency, Camera Buffering, Dropped Frame, Control Deadline | 관측 노후화와 행동 지연 진단 |
| S10-05 | 실행 제한·복구 | Joint/Velocity/Workspace Limit, Rate Limiting, Command Clipping, Communication Timeout, Watchdog, Stop Mechanism, Reset/Recovery Procedure | 관측·명령 실패 시 동작 처리 |
| S10-06 | 실물 오차 | Backlash, Mechanical Compliance, Joint Friction, Actuator Saturation, Calibration Drift, Camera Extrinsic Drift, Payload Variation | simulator와 다른 동작의 원인 분리 |
| S10-07 | 전이의 기본 개념 | Reality Gap, System Identification, Dynamics/Visual Domain Randomization, Observation/Action Delay Randomization, Sim-to-real Transfer, Real-world IL | 실물 직접학습과 시뮬레이션 전이 구별 |
| S10-08 | 실물 실험 평가 | Held-out Initial Conditions, Session Shift, Autonomous/Assisted Success, Human Intervention Count, Reset Cost, Failure Taxonomy | 새 세션·조건에서 자율 동작을 검증 |

**학습 깊이:** 위 항목은 SO-101 실습의 공통 개념이다. 맞춤 simulator·실제 online RL·TurtleBot3와의 통합은 선택 프로젝트다. ROS 2 자체는 LeRobot 기반 IL의 선행 필수 조건이 아니다.

**실물 실습 순서:** 카메라/관절 로그 → 낮은 속도의 작은 명령 → leader-follower 조작 → 학습 없는 시퀀스 → 시연 기반 정책 → 독립 평가 → 실패 데이터 보강. 처음에는 단단히 고정한 follower와 가벼운 물체로 짧은 집기·놓기 과제를 택한다. 사람이 즉시 멈출 수 있는 상태에서 실행 범위를 넓힌다.

**제어 이론과 장비의 관계:** impedance·토크 제어는 중요한 필수 이론이지만, 해당 인터페이스를 지원하지 않는 실물 장비에 그대로 실습할 수는 없다. 이 부분은 시뮬레이터에서 공부하고, SO-ARM-101에서는 확인된 드라이버의 joint target 인터페이스를 사용한다.

**Sim-to-real 심화:** 실제 로봇에 맞는 기구·actuator·action mapping을 만든 뒤 system identification, domain randomization, sensor/action delay와 현실적 noise를 적용한다. randomization 범위는 측정과 가설을 바탕으로 정하고 테스트 조건은 분리한다. 정확한 digital twin 제작을 첫 BC 학습의 필수 조건으로 삼지 않는다.

**통과 기준:** 새로운 실행 세션에서 setup과 평가를 재현한다. 같은 task의 nominal 조건과 별도 물체 위치·조명 조건을 비교한다. sim stress test만 수행했다면 sim-to-real을 검증했다고 쓰지 않는다. 실물에서 직접 시연으로 학습했다면 real-world IL이라고 명확히 표현한다.

**실제 RL — 후반 선택:** 이미 검증한 baseline, task reward, reset 방식, 제한된 행동 범위와 감독 체계가 확보된 다음 고려한다. 보상 penalty나 action clipping만으로 안전성이 보장된다고 보지 않는다. 처음부터 실물에서 무작위 탐색하는 실습은 이 커리큘럼의 시작 과제가 아니다.

<a id="stage-11"></a>
### 단계 11. 재현에서 독립 연구로

**목표:** 알려진 방법의 실행을 넘어, 질문·비교·증거를 갖춘 결과물을 만든다.

**반드시 알아야 할 개념**

| 항목 ID | 영역 | 개념의 정확한 명칭 | 로봇 학습과의 연결 |
|---|---|---|---|
| S11-01 | 문제와 가설 | Research Question, Falsifiable Hypothesis, Operational Definition, Assumption, Scope | 무엇을 어떤 증거로 검증할지 명확히 정의 |
| S11-02 | 비교 실험 | Baseline, Control Condition, Independent/Dependent Variable, Confounder, Ablation Study, Sensitivity Analysis | 변경한 방법의 효과와 다른 요인의 효과 구별 |
| S11-03 | 재현성 | Computational Reproducibility, Independent Replication, Code/Environment Versioning, Dataset Versioning, Run Manifest | 같은 artifact 재실행과 독립 구현 검증을 구별 |
| S11-04 | 선택·평가 분리 | Validation-based Model Selection, Hyperparameter Search Budget, Held-out Test Set, Test-set Leakage, Evaluation Protocol | 결과를 보고 유리한 조건을 고르는 오류 방지 |
| S11-05 | 통계 단위 | Training Seed, Evaluation Episode, Independent Replicate, Within-seed/Between-seed Variability, Clustered Data | 여러 rollout과 독립 학습 실행을 구별 |
| S11-06 | 불확실성·효과 | Effect Size, Confidence Interval, Bootstrap, Hierarchical/Cluster Bootstrap, Binomial Proportion Interval, Statistical versus Practical Significance | 작은 성공률 차이의 증거 수준 평가 |
| S11-07 | 일반화·타당성 | In-distribution/Out-of-distribution Generalization, Internal/External Validity, Pretraining Data Contamination, Distribution Shift | 실험 결론을 적용할 범위 명시 |
| S11-08 | 연구 보고 | Negative Result, Limitation, Failure Analysis, Dataset Card, Experiment Log, Artifact, Cost Accounting | 실패·제약·비용을 포함해 검증 가능한 결과 제공 |

**학습 깊이:** 재현성 용어의 정의는 분야별 차이가 있으므로 보고서에서 사용한 뜻을 명시한다. 통계 기법을 모두 적용하는 것보다 독립 표본의 단위와 실험 가정에 맞는 분석을 선택하는 것이 중요하다.

권장 순서:

1. 논문 한 편을 고르고 task·데이터·관측·action·controller·evaluation protocol을 표로 만든다.
2. 공식 코드의 작은 설정을 실행한 뒤 핵심 결과 하나를 재현한다.
3. 차이가 생기면 데이터·버전·제어·학습·평가 중 어디서 왔는지 분해한다.
4. 변경할 변수를 하나 정하고 반증 가능한 가설을 작성한다.
5. 실험 전 비교군·데이터·성공 조건·seed·상호작용 예산·model selection 방법을 정한다.
6. 실패 결과를 포함해 분석하고, 후속 실험은 그 분석에 필요한 것만 추가한다.
7. 코드·설정·표·영상·한계가 들어 있는 재현 보고서 또는 연구 보고서를 작성한다.

추천 첫 질문:

- SO-ARM-101에서 동일한 시연 수로 학습할 때, 복구 시연을 포함하면 초기 물체 위치 변화에 대한 성공률이 달라지는가?
- 같은 visual policy에서 execution horizon을 바꾸면 제어 지연과 실패 후 복구 능력은 어떻게 달라지는가?
- 배경·조명 변화에서 encoder를 고정하는 것과 fine-tuning하는 것의 차이는 데이터 양에 따라 달라지는가?
- 같은 simulation dataset에서 BC와 IQL의 차이를 시연 품질·reward 품질·coverage로 설명할 수 있는가?

**통과 기준:** 개선 여부와 무관하게 다른 사람이 결론을 검증할 자료가 있다. 알고리즘 변경의 효과와 데이터·하드웨어·계산량 증가의 효과를 구별한다. 독립 연구를 시작할 때는 연구실 멘토나 동료의 설계 검토를 받으면 시행착오를 줄일 수 있다.

<a id="s5"></a>
## 5. 프로젝트와 평가 기준

단계별 작은 실습을 아래 프로젝트에 합친다. 같은 코드를 매번 새로 만들기보다 데이터·평가 모듈을 재사용한다.

| 프로젝트 | 범위와 비교 | 완료 증거 |
|---|---|---|
| P0. PyTorch 작은 학습기 | NumPy 회귀와 PyTorch MLP, 저장·복원 | shape 설명, gradient 검사, train/validation plot, README |
| PCS. CS 통합 실습 — 선택 | 과목별 실습을 합성 센서 producer → bounded queue → 처리 worker → socket/파일 저장으로 연결 | 자료구조 선택 근거, 병목·queue 길이·지연·누락·종료 처리 결과 |
| P1. 로봇팔 제어 baseline | 2-link FK/IK → simulator pick-and-place → SO-101의 제한된 자세·경로 검증 | 좌표계 그림, 추종 오차, joint limit·IK 실패 분석 |
| P2. 시연 데이터 파이프라인 | SO-101 leader/follower 기록·재생·분할, 공개 sim 데이터의 BC | dataset card, split 목록, action 의미, 작은 BC rollout |
| P3. 매니퓰레이터 RL | 시뮬레이터의 SAC/PPO 기준선과 HER 또는 reward 비교 | 학습곡선, 다중 seed, 성공률·환경 step·실패 영상 |
| P4. 실제 로봇 IL | 같은 SO-101 task에서 단순 BC와 ACT 또는 지원되는 다른 sequence policy | 독립 test, autonomous success, latency, 실패 분류 |
| P5. 논문 재현 | simulation BC-RNN/ACT/Diffusion 또는 offline RL 한 방법 | 원논문과 설정 차이표, 재현 결과, 원인 분석 |
| P6. 독립 연구 | 데이터 효율·horizon·일반화·RL+IL 중 질문 하나 | 사전 실험 계획, ablation, 불확실성, 연구 보고서 |

**P4의 추천 과제:** 한 종류의 가벼운 물체를 집어 넉넉한 용기에 놓기. 처음에는 고정된 카메라와 작업 영역을 유지하고, 이후 물체 위치 → 배경·조명 → 새 물체 순으로 변화를 추가한다. “용기에 도달함”과 “물체가 실제로 용기 안에 놓였음”을 구별한다.

### 모든 학습 프로젝트에 적용할 평가 방식

1. **과제를 먼저 정의한다.** 초기 상태, 관측, 행동, 제어 주기, 제한 시간, 성공·실패·중단 조건을 문서화한다. 성공 판정은 평가 결과를 본 뒤 바꾸지 않는다.
2. **데이터는 episode 단위로 나눈다.** 같은 episode의 인접 프레임이나 겹친 action chunk를 train/test에 섞지 않는다. 새로운 물체 일반화를 주장하려면 물체 instance도 분리한다.
3. **Validation으로 선택하고 test로 보고한다.** normalization은 train에서 계산한다. test를 보고 checkpoint·학습률·horizon을 반복 선택하면 별도 test가 필요하다.
4. **Closed-loop rollout으로 평가한다.** 저장된 시연을 재생하거나 action MSE만 측정하는 것은 자율 동작 평가가 아니다.
5. **학습 seed와 평가 episode를 구별한다.** 같은 policy의 100회 실행은 100개 독립 학습 결과가 아니다. 시뮬레이션 학습용 비교는 예를 들어 3개 training seed와 seed별 30회 이상 평가로 시작하고, 결론의 불확실성에 따라 확대한다. 이것이 통계적 충분성을 보장하지는 않는다.
6. **실물 평가는 시행 횟수와 조건을 공개한다.** pilot의 소수 성공을 일반화하지 않는다. 학습 seed, 수집 날짜, 초기 물체 배치, calibration 세션의 변동을 가능한 범위에서 분리한다.
7. **공정한 비교를 한다.** 시연 수·관측 정보·action/controller·학습 상호작용·모델 선택 예산을 맞춘다. 맞출 수 없는 차이는 비교표에 적는다.
8. **여러 지표를 남긴다.** 성공률, 완료 시간, timeout·낙하·충돌·정지, 추론 지연, reset 횟수, 인간 개입, 데이터 수집량, 환경 step, compute 사용량을 기록한다. 측정할 센서가 없는 contact force를 추정 없이 실제 측정값처럼 보고하지 않는다.
9. **불확실성을 보여준다.** seed별 원자료와 평균/중앙값을 남긴다. 성공률은 평가 episode 수와 이항 구간 추정 등을 함께 보고, 여러 seed를 합칠 때 seed 내 상관을 고려한다. 적은 seed로 작은 성능 차이를 확정하지 않는다.
10. **실패를 보존한다.** 전체 평가 로그, 대표 성공·실패 영상, 실패 taxonomy를 함께 제시한다. 사람의 도움을 받은 assisted success는 autonomous success와 분리한다.

학습 데이터와 같은 조건(ID), 새로운 위치·물체·배경(OOD), 동역학·지연 변화는 따로 보고한다. 여러 task를 평가하면 task별 결과를 먼저 보여주고, 종합 지표는 보조로 사용한다. 평가 방법의 중요성은 [Statistical Precipice 논문](https://arxiv.org/abs/2108.13264)을 참고한다.

### 실험 결과표 양식

| 방법 | 데이터·관측 | 학습 seed | 평가 횟수 | 성공률·구간 | 지연 | 실패 유형 | 추가 비용 |
|---|---|---|---|---|---|---|---|
| BC baseline | 기록 | 기록 | 기록 | 기록 | 기록 | 기록 | 시연·reset |
| 비교 방법 | 기록 | 기록 | 기록 | 기록 | 기록 | 기록 | compute·개입 |

수치가 좋은 경우만 채우지 않는다. 재현에 실패한 결과도 설정과 원인이 분명하면 학습 성과다.

<a id="s6"></a>
## 6. 실습 환경과 보유 장비 선택

### 6.1 권장 기본 구성

**추천 구성:** Python·NumPy·PyTorch → 작은 Gymnasium 환경 → MuJoCo 기반 manipulation 환경 한 묶음 → 실제 SO-ARM-101용 LeRobot. ROS 2는 TurtleBot3·실물 시스템 연동을 배우는 경로로 사용한다.

| 도구 | 역할 | 사용할 때 |
|---|---|---|
| [PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/) | 학습 모델·autograd | 모든 학습의 기반 |
| [Gymnasium](https://gymnasium.farama.org/introduction/basic_usage/) | RL 환경 인터페이스 | 작은 환경의 RL 디버깅 |
| [MuJoCo](https://mujoco.readthedocs.io/en/stable/) | 물리 시뮬레이션 | 로봇 모델·접촉·제어 이해 |
| [Gymnasium-Robotics](https://robotics.farama.org/envs/fetch/pick_and_place/) | goal-conditioned manipulation 과제 | HER·희소 보상 실습 |
| [robosuite](https://robosuite.ai/) + [robomimic](https://robomimic.github.io/docs/) | 조작 환경 + demonstration 학습 | simulation BC·시각 IL의 기본 선택 |
| [ManiSkill](https://maniskill.readthedocs.io/en/latest/) | 조작 시뮬레이션·RL/IL | 병렬 실험·물체 일반화가 필요할 때 대안 |
| [LeRobot](https://huggingface.co/docs/lerobot/index) | 실제 로봇 데이터·정책 파이프라인 | SO-ARM-101 실습 |
| [CleanRL](https://docs.cleanrl.dev/) | 읽기 쉬운 RL 구현 | PPO/SAC update 이해 |
| [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) | 기준선과 실험 유틸리티 | 직접 구현과 비교 |
| [Drake](https://drake.mit.edu/) | 모델 기반 제어·계획·최적화 | MIT Manipulation 실습과 심화 |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 다중 task·지식 전이 benchmark | 기본 IL 이후 선택 |

이 표는 설치 체크리스트가 아니다. 먼저 작은 RL 환경과 하나의 manipulation 묶음을 고르고 필요한 경우에만 추가한다. GPU 병렬 시뮬레이션 프레임워크의 전환은 첫 BC 성공의 선행 조건이 아니다.

의존성은 프로젝트별로 분리한다. OS·Python·PyTorch·simulator·driver·dataset·controller·code commit을 기록하고, 논문 시대의 환경과 현재 환경을 무조건 섞지 않는다. [robomimic 데이터 안내](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html)는 simulator 버전 변화가 재현 결과에 영향을 줄 수 있음을 설명한다. [ManiSkill replay 안내](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html)는 CPU/GPU replay의 차이를 다룬다.

### 6.2 SO-ARM-101 leader·follower 활용

공식 LeRobot 문서에서는 **SO-101**이라는 이름을 사용한다. [SO-101 문서](https://huggingface.co/docs/lerobot/main/en/so101)와 [SO-ARM100 하드웨어 저장소](https://github.com/TheRobotStudio/SO-ARM100)를 참고하되 SO-100/SO-101 부품·모델·설정을 구별한다.

SO-101은 팔의 자세를 만드는 5개 관절과 gripper 구동으로 구성된다. 모터 6개를 임의의 6D end-effector pose를 제어할 수 있는 6-DoF arm과 동일시하지 않는다. 확인한 표준 LeRobot follower 코드는 position mode에서 관절 목표 위치를 사용한다. 단위·정규화·joint order는 실제 설치한 설정을 기준으로 확인한다. [공식 follower 소스](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py).

보유 세트에 맞춘 프로젝트 경로:

| 순서 | 할 일 | 남길 기록 |
|---|---|---|
| H1 | leader/follower 식별, 기존 조립·모터 설정 확인 | 모델·포트·모터·소프트웨어 버전 |
| H2 | 각 arm calibration과 범위 확인 | calibration 파일, joint order·단위 |
| H3 | 좁은 작업 영역의 teleoperation 확인 | 지연·추종 오차·그리퍼 동작 |
| H4 | 짧은 시연을 소량 수집하고 시각화 | episode·action·timestamp·영상 점검 |
| H5 | 초기 상태를 다양화하며 시연 확장 | dataset card, split, 수집 조건 |
| H6 | BC/ACT 학습과 독립 평가 | checkpoint, 성공률·실패 영상 |
| H7 | 실패 유형에 맞춘 추가 데이터 수집 | 추가 데이터의 효과 비교 |

[LeRobot 실제 IL 튜토리얼](https://huggingface.co/docs/lerobot/main/en/il_robots)의 setup/teleoperate/record/train/eval 흐름을 사용한다. 연결한 `main` 문서는 source 버전이므로 stable package를 설치했다면 문서의 버전 선택도 맞춘다. 이미 조립·설정된 장비에 모터 초기 설정을 무조건 반복할 필요는 없다. 이 문서는 학습 경로이며, 실제 설치·연결 검증을 수행한 결과는 아니다.

### 6.3 TurtleBot3 활용 — 선택

TurtleBot3로 배울 내용은 manipulator의 주제와 겹치는 시스템 역량을 우선한다.

- ROS 2 node/topic/service/action, launch, parameter, QoS, `tf2`, URDF와 RViz.
- `map → odom → base_link → sensor` 좌표 관계, odometry·IMU·LiDAR의 역할.
- sensor timestamp, ROS bag 기록·재생, command latency·통신 단절 처리.
- SLAM·localization·navigation의 기본 흐름. learned navigation은 관심이 생길 때 추가한다.

**완료 과제:** 주행 데이터와 센서 bag을 저장하고, RViz와 `tf2`로 프레임을 설명한 뒤, navigation 실패 하나를 localization·planning·control·센서 문제로 분해한다.

자료: ROBOTIS 공식 [Quick Start](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/), [Basic Operation](https://emanual.robotis.com/docs/en/platform/turtlebot3/basic_operation/), [SLAM](https://emanual.robotis.com/docs/en/platform/turtlebot3/slam/), [Navigation](https://emanual.robotis.com/docs/en/platform/turtlebot3/navigation/). Burger/Waffle/Waffle Pi 모델과 현재 OS·ROS 배포판에 맞는 탭을 사용한다. 새로운 배포판으로의 재설치를 선행 과제로 정하지 않는다.

**SO-ARM-101을 TurtleBot3에 올리는 mobile manipulation은 선택 연구다.** 공식 TurtleBot3 manipulation 예제의 OpenMANIPULATOR 구성과 SO-101의 결합은 별개다. 장착 구조·무게중심·전원·통신·base-to-arm transform·충돌 모델을 검증해야 한다. 첫 통합 과제는 `이동 → base 정지 → 물체 재관측 → arm 조작`으로 구성하고, 동시 주행·조작은 이후로 둔다. [ROBOTIS Manipulation 문서](https://emanual.robotis.com/docs/en/platform/turtlebot3/manipulation/).

### 6.4 계산 자원에 따른 조절

현재 GPU와 카메라 보유 여부는 확인되지 않았다. 장비 구매를 전제로 진도를 막지 않고 다음처럼 나눈다.

| 사용 가능한 자원 | 먼저 할 수 있는 것 | 범위 조절 |
|---|---|---|
| CPU 중심 | 수학·기구학, 작은 PyTorch 모델, 상태 BC, 소규모 RL, 데이터 검증 | 이미지·대형 모델 학습은 작은 공개 예제나 공유 자원으로 분리 |
| 단일 GPU | visual BC, 작은 ACT/Diffusion 실험 | batch·영상 해상도·history·camera 수를 줄이고 메모리 측정 |
| 연구실 GPU/공유 자원 | 반복 seed, 더 큰 dataset, pretrained policy adaptation | 전체 VLA 사전학습보다 작은 fine-tuning부터 |
| 카메라 미확보 | joint 상태·동작 검증, simulator/public image dataset 실습 | 실제 visual policy에는 동기화된 관측 카메라가 필요 |

GPU 모델명만으로 가능 여부를 단정하지 않는다. 실제 설정으로 smoke test를 실행해 peak VRAM, 데이터 로딩 시간, 추론 지연을 측정한다. 단일 작업의 작은 모델로 pipeline을 검증한 뒤 계산량을 늘린다.

<a id="s7"></a>
## 7. 핵심 논문 읽기 순서

논문은 해당 단계에서 읽는다. 처음부터 목록 전체를 읽고 나서 구현을 시작하지 않는다. 아래 연도는 대표 발표·공개 연도로, preprint와 학회 발표 연도가 다른 경우를 표시했다. 대표 개념을 학습하기 위한 목록이며 2026년의 성능 순위가 아니다.

### 7.1 공통 핵심 — 정독할 12개 읽기 단위

| 순서 | 논문 | 연결 단계 | 읽고 답할 질문 |
|---|---|---|---|
| C1 | [Generalized Advantage Estimation, GAE (2015 preprint)](https://arxiv.org/abs/1506.02438) | 5 | lambda와 bootstrap이 bias·variance를 어떻게 바꾸는가? |
| C2 | [Proximal Policy Optimization, PPO (2017)](https://arxiv.org/abs/1707.06347) | 5 | ratio clipping은 무엇을 제한하고 무엇을 보장하지 않는가? |
| C3 | [Addressing Function Approximation Error in Actor-Critic Methods, TD3 (2018)](https://arxiv.org/abs/1802.09477) | 5 | twin Q·delayed update·target smoothing의 역할은 각각 무엇인가? |
| C4 | [Soft Actor-Critic (2018)](https://arxiv.org/abs/1801.01290)와 [Algorithms and Applications (2018)](https://arxiv.org/abs/1812.05905) | 5 | entropy objective·temperature·실제 actor/critic loss가 어떻게 연결되는가? |
| C5 | [DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (2011; 2010 preprint)](https://arxiv.org/abs/1011.0686) | 6 | learner가 방문한 상태의 label은 누가 제공하며 오류 누적을 어떻게 줄이는가? |
| C6 | [What Matters in Learning from Offline Human Demonstrations for Robot Manipulation (2021)](https://arxiv.org/abs/2108.03298) | 6, 8 | 데이터 품질·history·관측 선택의 효과를 알고리즘 효과와 구별할 수 있는가? |
| C7 | [Hindsight Experience Replay, HER (2017)](https://arxiv.org/abs/1707.01495) | 7 | 실패 trajectory를 다른 목표의 경험으로 재사용할 조건은 무엇인가? |
| C8 | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware / ACT (2023)](https://arxiv.org/abs/2304.13705) | 8 | action chunk·CVAE·temporal ensembling이 각각 해결하는 문제는 무엇인가? |
| C9 | [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (2023)](https://arxiv.org/abs/2303.04137) | 8 | 생성 분포·horizon·실행 지연을 어떻게 연결하는가? |
| C10 | [Conservative Q-Learning, CQL (2020)](https://arxiv.org/abs/2006.04779) | 9 | 보수적인 value 추정이 필요한 이유와 지나칠 때의 비용은? |
| C11 | [Offline Reinforcement Learning with Implicit Q-Learning, IQL (2022; 2021 preprint)](https://arxiv.org/abs/2110.06169) | 9 | expectile과 advantage-weighted policy extraction이 어떻게 작동하는가? |
| C12 | [Deep Reinforcement Learning at the Edge of the Statistical Precipice (2021)](https://arxiv.org/abs/2108.13264) | 실험 시작부터 | seed가 적을 때 어떤 결론을 내릴 수 있고 어떤 결론은 어려운가? |

C4는 두 편을 묶은 읽기 단위다. **직접 재현할 최소 범위는 PPO/SAC 중 하나, BC, ACT/Diffusion 중 하나**로 잡고, 나머지는 수식·가정·코드와 실험 설정을 읽어 이해한다. Offline RL을 연구하면 IQL/CQL 재현을 추가한다.

### 7.2 연구 방향별 심화 읽기

| 분야 | 자료 | 집중할 내용 |
|---|---|---|
| RL+IL | [AWAC (2020)](https://arxiv.org/abs/2006.09359) | offline 데이터와 online 개선의 연결 |
| RL+IL | [Efficient Online RL with Offline Data / RLPD (2023)](https://arxiv.org/abs/2302.02948) | replay 구성·데이터 비율·구현 선택 |
| 실제 RL 시스템 | [HIL-SERL (2024 preprint)](https://arxiv.org/abs/2410.21845) | 데모·사람 개입·보상 분류기·실제 운영 비용 |
| 기존 제어와 RL | [Residual Reinforcement Learning for Robot Control (2018 preprint)](https://arxiv.org/abs/1812.03201) | 명령 보정 범위와 기존 제어기의 역할 |
| Model-based RL | [Deep RL in a Handful of Trials using Probabilistic Dynamics Models / PETS (2018)](https://arxiv.org/abs/1805.12114) | 모델 불확실성·ensemble·MPC |
| 시각 RL | [Mastering Visual Continuous Control / DrQ-v2 (2021)](https://arxiv.org/abs/2107.09645) | augmentation·표현·학습 구현 |
| Sim-to-real | [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization (2017 preprint)](https://arxiv.org/abs/1710.06537) | 현실 오차와 randomization 분포 |
| 제약 RL | [Constrained Policy Optimization (2017)](https://arxiv.org/abs/1705.10528) | CMDP·기대 제약과 실제 시스템 보장의 차이 |
| IL 이론 | [Generative Adversarial Imitation Learning, GAIL (2016)](https://arxiv.org/abs/1606.03476) | occupancy matching·expert reward의 모호성 |
| 생성 모델 | [Flow Matching for Generative Modeling (2023; 2022 preprint)](https://arxiv.org/abs/2210.02747) | conditional vector field·ODE sampling |
| 데이터 생성 | [MimicGen (2023)](https://arxiv.org/abs/2310.17596) | 소수 시연 확장에 필요한 task 구조와 split |
| 데이터 인터페이스 | [Universal Manipulation Interface, UMI (2024)](https://arxiv.org/abs/2402.10329) | 수집 방식·relative action·latency matching |
| 다중 task 평가 | [LIBERO (2023)](https://arxiv.org/abs/2306.03310) | 공간·물체·목표 전이 구별 |
| Generalist policy | [Octo (2024)](https://arxiv.org/abs/2405.12213) | observation/action 공간 적응과 diffusion policy |
| VLA | [OpenVLA (2024)](https://arxiv.org/abs/2406.09246) | VLM·action token·fine-tuning |
| VLA와 flow | [π0 (2024)](https://arxiv.org/abs/2410.24164) | 언어·영상 표현과 continuous action 생성 |

VLA는 기초를 대체하는 시작점이 아니다. pretrained checkpoint를 fine-tuning할 때도 calibration, action 의미, 데이터 범위와 closed-loop 평가가 필요하다. 모델 전체의 사전학습은 개인 학습 커리큘럼의 필수 과제로 두지 않는다.

### 7.3 논문 한 편을 읽는 방법

1. **문제:** 정확히 어떤 task·관측·행동·하드웨어·데이터 조건을 푸는가?
2. **가정:** expert, reward, privileged state, reset, 언어 label, 사전학습 자원은 어디서 오는가?
3. **방법:** 목적함수에서 최적화하는 변수는 무엇이며 실제 코드의 어느 update인가?
4. **증거:** 가장 강한 baseline은 무엇이고, 데이터·계산량·관측 조건이 공정한가?
5. **한계:** 실패하는 task·분포·하드웨어 조건은 무엇인가?
6. **재현:** 내 자원으로 검증할 수 있는 가장 작은 주장은 무엇인가?

초록만 읽고 “방법을 배웠다”고 기록하지 않는다. 핵심 식, 알고리즘 그림, 실험표 하나를 자기 말로 설명하고, 코드의 입력부터 출력까지 추적한다.

<a id="s8"></a>
## 8. 심화 전문 분야 — 공통 기반 이후 하나씩 선택

| 트랙 | 추가로 배울 내용 | 적합한 연구 산출물 |
|---|---|---|
| 데이터 효율적인 IL | active collection, recovery/correction, data selection, multi-demonstrator, augmentation | 같은 시연 예산에서 수집 전략 비교 |
| 접촉·정밀 조작 | compliance, hybrid force/motion, tactile·force sensing, contact dynamics, insertion | 접촉 조건 변화의 성공·오차 분석 |
| Dexterous/bimanual | 손의 기구학, in-hand manipulation, 손·양팔 협응, retargeting | simulator 기반 조작 재현; 추가 장비는 연구 여건에 따라 |
| 3D·기하 기반 학습 | point cloud, 3D representation, SE(3) equivariance, pose uncertainty | 2D/3D 표현의 데이터 효율·일반화 비교 |
| Long-horizon | options, skill learning, hierarchical RL, subgoal, TAMP, failure detection·recovery | 다단계 task의 실패 감지·재시도 시스템 |
| Offline-to-online RL | distribution shift, conservative methods, replay mixing, online adaptation | 초기 개선·성능 붕괴·상호작용 비용 비교 |
| Model-based·world model | latent dynamics, uncertainty, imagined rollout, MPC, model exploitation | 모델 오차와 실제 제어 성능의 관계 |
| Sim-to-real | system identification, dynamics/visual randomization, adaptation, residual learning | SO-101 모델-실물 오차 분해와 실제 전이 |
| Generalist·VLA | language grounding, cross-embodiment, action tokenizer/head, PEFT, continual learning | 작은 checkpoint adaptation과 데이터 중복 검사 |
| 이론 | occupancy measure, inverse RL, online learning/regret, policy gradient·sample complexity | 작은 문제의 유도·가정·반례와 실험 |
| 제약·신뢰성 | CMDP, risk-sensitive RL, uncertainty/OOD detection, control barrier function, runtime monitoring | 성능과 제약 위반의 trade-off 분석 |
| Mobile manipulation | navigation uncertainty, base-arm frames, whole-body planning, coordinated control | TurtleBot3와 arm의 단계적 통합 |

새로운 분야를 추가할 때는 필요한 하드웨어와 관측을 먼저 확인한다. 예를 들어 촉각 센서가 없는 SO-101 실험만으로 tactile policy를 검증했다고 할 수는 없다. 실물 장비로 할 수 있는 범위와 simulation 연구 범위를 구분하면 된다.

처음 전문화하기에 맞는 후보는 **SO-101의 시연 데이터 효율**, **action chunking과 지연**, **visual IL의 물체·배경 일반화**다. 이미 보유한 leader·follower 세트를 활용하면서 공통 핵심 이론과 연결할 수 있다는 이유로 추천한다.

<a id="s9"></a>
## 9. 연구 방법과 재현성

### 9.1 반드시 남길 네 가지 문서

**실험 계획서**

```text
연구 질문:
가설과 이를 반박할 수 있는 결과:
task / robot / observation / action / controller:
baseline과 변경할 변수 한 가지:
고정할 데이터·초기 상태·학습·평가 조건:
training seed / evaluation episode 계획:
성공 조건 / 보조 지표 / 중단 조건:
checkpoint와 hyperparameter를 선택할 validation 규칙:
test 조건:
예상되는 confounder와 분리 방법:
```

**Dataset card**

```text
수집 장비·날짜·조작자·task:
camera / calibration / joint order / action 단위·의미:
sampling frequency / timestamp / delay:
episode 수·길이·성공·실패·복구 포함 여부:
train / validation / test episode ID:
물체·배경·수집 세션별 split:
normalization을 계산한 train subset:
라이선스·출처·공개 범위:
전처리·제외한 episode와 이유:
```

**Run manifest**

```text
code commit / OS / Python / framework / simulator·driver version:
dataset version·hash / task version / controller 설정:
config / seed / checkpoint / 실행 명령:
action horizon·policy frequency / observation history:
학습 interaction / 평가 interaction / reset / 사람 개입:
compute 사용량 / peak memory / inference latency:
평가 raw log / 동영상 / 결과표:
```

**결과 보고서**

```text
문제와 가설 → 방법 → 실험 조건 → 결과 → 실패 분석 → 한계
핵심 그림: 학습곡선, 데이터 양 또는 조건 변화 곡선
핵심 표: baseline 비교, ablation, 실패 유형, 비용
재현 안내: 데이터 접근, 환경 설치, train/eval 명령
```

### 9.2 자주 발생하는 오류

| 증상·오류 | 먼저 확인할 것 |
|---|---|
| BC loss는 낮고 rollout은 실패 | covariate shift, normalization, timestamp, action label·control frequency |
| SAC critic 값이 폭주 | reward scale, terminal target, Q target·gradient 경로, action/log-prob 계산 |
| PPO가 update 이후 급격히 무너짐 | old log-prob, ratio·KL, advantage, learning rate, rollout 처리 |
| 시뮬레이션 결과가 재현되지 않음 | physics backend·버전, controller, dataset state replay, reset seed |
| 팔이 엉뚱한 방향으로 움직임 | frame, joint order, 부호, radian/degree, normalization, calibration |
| 길게 실행할수록 지연·진동 | camera buffer, 추론 시간, control frequency, chunk 실행 길이·명령 큐 |
| validation만 지나치게 좋음 | 같은 episode/window의 split 누수, test 사용, 미래 프레임 |
| 실제에서만 실패 | 관측 차이, friction·backlash, 위치 오차, delay, reset·데이터 분포 |
| 보상은 높지만 task 실패 | success predicate와 reward 분리, reward hacking·classifier false positive |

vectorized environment의 auto-reset에서는 다음 episode의 초기 관측을 직전 episode의 마지막 관측처럼 저장하지 않도록 확인한다. HER에서 goal을 바꾸면 필요한 reward와 종료 의미를 다시 계산한다. 체크포인트만 저장하고 normalization·processor·controller 설정을 잃어버리지 않는다.

### 9.3 일반화와 데이터 오염

- 같은 물체의 새 위치, 새 물체 instance, 새 task, 새 robot embodiment는 서로 다른 일반화 문제다.
- pretrained model의 학습 데이터와 benchmark가 겹쳤는지 확인한다. 확인 불가능하면 그 사실을 적고 완전한 zero-shot 일반화를 주장하지 않는다.
- synthetic trajectory가 동일 원본 시연에서 생성되었다면 원본 단위의 split도 검토한다.
- 알고리즘별 튜닝 기회·데이터 수집량·사람 개입이 다르면 그 차이를 결과의 일부로 보고한다.
- 큰 모델이나 더 긴 학습이 개선의 원인일 수 있으므로 핵심 비교에는 적절한 규모·비용 baseline을 포함한다.

### 9.4 논문 탐색 습관

CoRL·RSS·ICRA·IROS에서 manipulation 시스템과 실험을, NeurIPS·ICML·ICLR에서 RL/IL·생성 모델·학습 이론을 함께 살핀다. 학회 이름보다 문제·가정·코드·평가 품질을 기준으로 읽는다. 새로운 논문을 추가할 때는 기존 커리큘럼의 어떤 항목을 보완하는지 한 문장으로 적는다.

논문을 읽는 데 영어가 병목이면 abstract·method·experiment에서 반복되는 용어를 개인 용어집에 쌓는다. 번역은 보조로 쓰고 수식, action 정의, 성공 조건은 원문과 대조한다.

<a id="s10"></a>
## 10. 학교 수업·연구실·진로 연결

CS 학부에서 배운 내용은 CS-1~4의 진단·실습으로 점검하고, 로봇학·제어·연속 수학과 연결한다. 아래 과목명이 학교에 없다면 대응하는 강의나 개별 연구로 대체할 수 있다.

| 우선순위 | 과목 | 확인할 핵심 내용 |
|---|---|---|
| 공통 기반·병행 | 자료구조와 알고리즘 | 정확성·복잡도, 배열·해시·트리·그래프, 탐색·정렬·DP — CS-1 |
| 공통 기반·병행 | 컴퓨터 구조 | 수 표현·ISA·파이프라인·cache·CPU/GPU·성능 측정 — CS-2 |
| 공통 기반·병행 | 운영체제(Operating Systems) | 프로세스·메모리·동기화·IPC·파일·실행 지연 — CS-3 |
| 공통 기반·병행 | 컴퓨터 네트워크 | TCP/IP·socket·전송 신뢰성·지연·로봇 미들웨어 — CS-4 |
| 가장 먼저 | 선형대수·확률통계·최적화 | 수치 선형대수, 확률 모델, 제약 최적화 |
| 가장 먼저 | 머신러닝·딥러닝 | supervised learning, PyTorch 구현, 일반화 |
| 가장 먼저 | 로봇공학·Robot Manipulation | 기구학·동역학·Jacobian·접촉 |
| 가장 먼저 | 제어공학 | feedback·선형 시스템·안정성·LQR |
| 다음 | 강화학습 | MDP, actor–critic, continuous control |
| 다음 | 컴퓨터비전 | camera geometry·표현 학습 |
| 연구 방향에 따라 | 최적제어·비선형 제어·모션 플래닝 | MPC·trajectory optimization·제약 |
| 연구 방향에 따라 | 확률 로보틱스·추정 | filtering·localization·sensor fusion |
| 연구 방향에 따라 | 실시간·임베디드·병렬 컴퓨팅 | latency·driver·profiling·실험 처리량 |
| 연구 방향에 따라 | 고급 확률·통계 학습 이론 | 연구 가정과 보장·실험 불확실성 |

연구실에는 모든 내용을 끝내고 들어갈 필요가 없다. P1~P2 정도의 작은 결과물이 있으면 다음 자료를 갖추어 관심 연구실과 상담할 수 있다.

- 재현 가능한 저장소 하나와 짧은 실행 영상.
- 성공만 나열하지 않은 실험·실패 분석 보고서.
- 관심 논문 한 편과 구체적인 후속 질문.
- 사용할 수 있는 장비와 현재 할 수 있는 일, 배우려는 기술.

학부 졸업 프로젝트라면 **SO-101에서 BC와 ACT를 비교하고, 시연 수 또는 horizon의 영향 하나를 분석하는 범위**가 적합한 출발점이다. 연구 질문의 참신함은 지도교수·관련 문헌을 통해 확인한다. 시뮬레이션 RL·실제 IL·VLA·mobile manipulation을 한 프로젝트의 동시 필수 목표로 잡지 않는다.

진로별로 마지막 비중을 조정한다. 연구 중심이라면 수식·재현·가설과 실험 설계를, robotics learning engineer라면 데이터 파이프라인·제어 통합·latency·장애 분석을 더 깊게 한다. 두 경로 모두 공통 기반은 필요하다.

<a id="s11"></a>
## 11. 처음 실행할 과제와 학습 운영

### 11.1 지금 시작할 여덟 과제

기간을 정하지 않고 순서와 완료 조건만 둔다. 수학 복습과 로봇 기본 확인은 병행할 수 있다.

1. **현재 환경 기록:** Python·OS·GPU 유무, SO-101 leader/follower 상태, 카메라 유무, TurtleBot3 모델·ROS 배포판을 한 파일에 정리한다.
2. **Tensor 실습:** `(batch, features)`와 `(batch, time, channels, height, width)`를 예로 shape·index·reshape·broadcasting·device·dtype을 설명한다.
3. **Autograd 실습:** 작은 함수의 미분을 손으로 구하고 PyTorch와 비교한다. `detach`, `no_grad`, gradient 누적의 차이를 확인한다.
4. **학습 루프 작성:** Dataset/DataLoader → `nn.Module` → forward → loss → `zero_grad` → backward → optimizer step을 직접 작성한다.
5. **학습 검증:** 작은 batch overfit, validation, `train()`/`eval()`, checkpoint 저장·복원까지 완료한다. optimizer와 normalization 등 재개·추론에 필요한 상태도 보존한다.
6. **2-link arm 구현:** FK와 numerical Jacobian을 만들고 목표점까지 IK로 움직인다. joint limit와 도달 불가능 목표를 처리한다.
7. **SO-101 기본 경로 검증:** 현재 버전의 공식 문서로 캘리브레이션·leader-follower 동작·관측 기록을 확인한다. 소수의 짧은 episode를 저장해 영상과 action을 대조한다.
8. **첫 BC 실행:** 공개 상태 기반 데이터나 충분한 task 관측이 있는 작은 데이터로 MLP BC를 만든다. offline loss와 closed-loop 결과를 따로 기록한다.

TurtleBot3는 위 경로가 안정된 뒤 ROS 실습에 사용해도 충분하다. SO-101을 바로 TurtleBot3에 장착하는 것이 시작 과제는 아니다.

### 11.2 PyTorch 숙련도 확인

세부 확인 항목은 [별도 체크리스트의 PyTorch 실습 점검](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_checklist_ko.md#pytorch-practice>)으로 옮겼다. 단계 2의 개념과 구현 결과를 함께 점검한 뒤 다음 과제로 진행한다.

### 11.3 한 학습 단위를 끝내는 방식

```text
개념 하나 선택
→ 작은 예제로 손계산
→ 최소 구현
→ 의도적으로 오류·분포 변화를 넣어 보기
→ 결과와 실패 원인 기록
→ 논문의 같은 개념과 연결
→ 다음 과제 선택
```

진행 중인 핵심 프로젝트는 하나를 유지한다. 막히면 환경·데이터·수학·학습·제어·평가 중 어느 부분인지 먼저 분류하고, 그 부분을 작게 재현한다. 강의 수강량 대신 작동하는 코드와 설명 가능한 실험을 진도의 기준으로 삼는다.

학습 노트는 `오늘 설명할 수 있게 된 개념 / 실행한 실험 / 관찰한 실패 / 다음에 확인할 가설` 네 항목이면 충분하다. 학습 비율이나 주당 공부시간을 고정할 필요는 없다.

<a id="s12"></a>
## 12. 별도 체크리스트 사용법

[학습·프로젝트 체크리스트](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_checklist_ko.md>)에서 단계 0~11과 CS-1~4의 필수 개념, PyTorch 실습, 프로젝트와 연구 준비도를 점검한다. 본문은 개념·실습·자료를 읽는 문서로, 체크리스트는 진도를 기록하는 문서로 사용한다.

개념 이름을 본 적 있다는 이유만으로 완료 표시를 하지 않는다. 개념을 자기 말로 설명하고, 해당 단계에서 요구한 구현·실험·분석 증거를 남긴다. 선택 심화는 연구 방향에 맞게 골라 수행하며, 주당 공부시간이나 완료 기한을 지정할 필요는 없다.

---

**자료 이용 메모:** 링크는 공식 대학 강의·저자/프로젝트 페이지·원논문·공식 문서를 우선해 확인했다. 링크 확인과 소프트웨어 실행 검증은 다르며, 이 문서를 작성하면서 사용자의 로봇이나 학습 코드를 실행하지는 않았다. 패키지·문서의 `main/latest`는 바뀔 수 있으므로 실제 실습에서는 선택한 release 또는 commit을 고정한다. 학습 순서·프로젝트·통과 기준은 사용자 목표와 보유 장비에 맞춘 교육 설계다.
