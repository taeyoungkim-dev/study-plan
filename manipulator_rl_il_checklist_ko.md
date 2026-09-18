# 로봇 매니퓰레이터 RL·IL 학습 체크리스트

> 개정일: 2026-09-18. 대상: CS 학부 4학년, PyTorch 기초 보완, SO-ARM-101 leader·follower 세트 및 TurtleBot3 보유. 주당 공부시간·마감일을 고정하지 않는다.

개념 설명·실습·자료는 [전체 커리큘럼](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md>)에 있다. S 항목은 로봇 학습 단계, CS 항목은 CS 공통 과목에 대응한다. G·GCS 항목은 단계·과목 완료 기준이며, F 항목은 학습 구현·실물 실행을 점검하는 보충 항목이다.

## 사용법

- 개념 항목은 나열된 용어의 정의·역할·가정·대표 예제를 설명할 수 있을 때 체크한다. 이름을 읽어본 것만으로 완료 처리하지 않는다.
- 구현 깊이는 본문의 각 단계에서 정한 범위를 따른다. 개념 체크가 모든 방법의 전체 구현을 뜻하지는 않는다.
- 단계·과목 완료 과제(G0~G11, GCS1~GCS4)는 코드·실험표·영상·노트 등의 증거가 있을 때 체크한다.
- 체크박스를 `[ ]`에서 `[x]`로 변경한다. 선택 과정은 적용하지 않으면 메모에 '미선택'을 적는다.
- 개념별 추가 기록이 필요하면 항목 아래에 증거 경로와 복습할 질문을 적는다.
- 단계 1·2의 복습은 병행할 수 있고, 단계 2와 기구학·기본 제어를 익힌 뒤 단계 5의 RL과 단계 6의 BC를 병행할 수 있다. 모든 이론 항목을 끝내야 첫 로봇 실습을 시작할 수 있는 것은 아니다.

## 바로가기

- [단계 0. 개발·실험 기반](#check-stage-0)
- [CS-1. 자료구조와 알고리즘](#check-cs-1)
- [CS-2. 컴퓨터 구조](#check-cs-2)
- [CS-3. 운영체제](#check-cs-3)
- [CS-4. 컴퓨터 네트워크](#check-cs-4)
- [단계 1. 수학·확률·최적화](#check-stage-1)
- [단계 2. 머신러닝·딥러닝·표현 학습](#check-stage-2)
- [단계 3. 로봇 기구학·동역학·접촉](#check-stage-3)
- [단계 4. 제어·모션 플래닝·지각](#check-stage-4)
- [단계 5. 강화학습의 원리와 연속 제어](#check-stage-5)
- [단계 6. 모방학습·시연 데이터·DAgger](#check-stage-6)
- [단계 7. 매니퓰레이터 강화학습](#check-stage-7)
- [단계 8. 시각·시퀀스·생성형 모방학습](#check-stage-8)
- [단계 9. Offline RL·RL+IL·model-based learning](#check-stage-9)
- [단계 10. SO-ARM-101 시스템 통합과 실제 평가](#check-stage-10)
- [단계 11. 재현에서 독립 연구로](#check-stage-11)
- [PyTorch 실습 점검](#pytorch-practice)
- [학습 구현·실물 실행 점검](#implementation-gates)
- [프로젝트 완료 점검](#project-gates)
- [실험·평가 점검](#evaluation-gates)
- [TurtleBot3 선택 과정](#turtlebot-track)
- [최종 역량 점검](#final-competencies)

<a id="check-stage-0"></a>
## 단계 0. 개발·실험 기반

개념·실습·자료: [커리큘럼 단계 0](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#stage-0>).

- [ ] **S00-01 · Python 기본 구조** — 함수(Function), 클래스(Class), 모듈(Module), 패키지(Package), 반복자(Iterator), 제너레이터(Generator), 예외 처리(Exception Handling), 타입 힌트(Type Hint).
- [ ] **S00-02 · 배열과 수치 계산** — 다차원 배열(N-dimensional Array), 형상(Shape), 자료형(Dtype), 인덱싱(Indexing), 브로드캐스팅(Broadcasting), 벡터화(Vectorization), 부동소수점 오차(Floating-point Error).
- [ ] **S00-03 · Tensor 입문** — Tensor, CPU/GPU Device, Reshape, Transpose, Permute, 행렬곱(Matrix Multiplication), 원소별 연산(Element-wise Operation).
- [ ] **S00-04 · 프로그램 실행 환경** — 가상환경(Virtual Environment), 의존성 고정(Dependency Pinning), 환경 변수(Environment Variable), 상대·절대 경로(Relative/Absolute Path), 명령행 인자(Command-line Argument).
- [ ] **S00-05 · 개발 도구** — Shell, Process, Standard Input/Output, SSH, Git Commit, Branch, Diff, Merge.
- [ ] **S00-06 · 검증·디버깅** — Assertion, Unit Test, 예외 추적(Traceback), 중단점(Breakpoint), 수치 미분 검사(Finite-difference Gradient Check).
- [ ] **S00-07 · 실험 기록** — Configuration, 의사난수 생성기(Pseudorandom Number Generator), Random Seed, Logging, Checkpoint, Profiling.

- [ ] **G0 · 단계 완료 과제** — 새 가상환경에서 README만 보고 작은 NumPy 계산·회귀 스크립트를 실행하고 설정·seed·결과를 저장한다. 기본 tensor의 shape/device 오류를 찾는다. 신경망 학습의 중단·재개는 단계 2에서 검증한다.

증거 기록: 코드/노트/실험 결과 경로 = __________; 아직 부족한 개념 = __________.

<a id="check-cs-core"></a>
## CS 공통 기반

자료구조·알고리즘, 컴퓨터 구조, 운영체제, 네트워크를 단계 0과 병행한다. 이미 익힌 과목은 완료 과제로 확인하고 필요한 부분을 복습한다. `CS1~CS4` 항목과 `GCS1~GCS4` 완료 기준은 본문에 대응하며, 주당 시간과 마감일을 고정하지 않는다.

<a id="check-cs-1"></a>
### CS-1. 자료구조와 알고리즘 — Data Structures and Algorithms

개념·실습·자료: [커리큘럼 CS-1](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#cs-1>).

- [ ] **CS1-01 · 정확성과 수학 기초** — 추상 자료형(Abstract Data Type, ADT), 명세(Specification), 사전·사후 조건(Precondition/Postcondition), 루프 불변식(Loop Invariant), 수학적 귀납법(Mathematical Induction), 종료성(Termination).
- [ ] **CS1-02 · 복잡도 분석** — 시간·공간 복잡도(Time/Space Complexity), Big-O/Big-Omega/Big-Theta, 최악·평균 경우(Worst/Average Case), 분할상환 분석(Amortized Analysis), 점화식(Recurrence), Master Theorem의 적용 조건.
- [ ] **CS1-03 · 선형 자료구조** — 배열(Array), 동적 배열(Dynamic Array), 연결 리스트(Linked List), 스택(Stack), 큐(Queue), 덱(Deque), 원형 버퍼(Circular/Ring Buffer).
- [ ] **CS1-04 · 해시와 집합** — 해시 함수(Hash Function), 해시 테이블(Hash Table), 충돌 해결(Collision Resolution), Chaining, Open Addressing, Load Factor, Set/Map.
- [ ] **CS1-05 · 트리와 우선순위** — 트리 순회(Tree Traversal), 이진 탐색 트리(Binary Search Tree, BST), 균형 탐색 트리(Balanced Search Tree; AVL/Red–Black 개념), 이진 힙(Binary Heap), 우선순위 큐(Priority Queue).
- [ ] **CS1-06 · 검색과 정렬** — 이진 탐색(Binary Search), 안정 정렬(Stable Sorting), In-place Algorithm, Insertion Sort, Merge Sort, Quicksort, Heapsort, Counting/Radix Sort의 적용 조건.
- [ ] **CS1-07 · 그래프 표현·순회** — 방향·무방향 그래프(Directed/Undirected Graph), 인접 리스트·행렬(Adjacency List/Matrix), Breadth-first Search(BFS), Depth-first Search(DFS), Connected Components, Topological Sort.
- [ ] **CS1-08 · 최단 경로·휴리스틱** — Dijkstra's Algorithm, Bellman–Ford Algorithm, Floyd–Warshall Algorithm, A* Search, Admissible/Consistent Heuristic, Edge Relaxation.
- [ ] **CS1-09 · 연결성과 신장 트리** — Disjoint-set Union/Union–Find, Path Compression, Union by Rank/Size, Minimum Spanning Tree(MST), Kruskal's Algorithm, Prim's Algorithm.
- [ ] **CS1-10 · 알고리즘 설계 전략** — Recursion, Divide and Conquer, Greedy Algorithm, Dynamic Programming(DP), Memoization, Tabulation, Backtracking, Optimal Substructure, Overlapping Subproblems.
- [ ] **CS1-11 · 계산 가능성과 난이도 입문** — Decision/Optimization Problem, Polynomial Time, Class P, Class NP, Polynomial-time Reduction, NP-hardness, NP-completeness.

- [ ] **GCS1 · 과목 완료 과제** — 자료구조별 연산 비용을 설명하고, 작은 그래프의 기준 해와 탐색 결과를 대조한다. 입력 크기에 따른 실행 시간과 확장 노드 수를 기록하며, replay buffer가 덮어쓰기·샘플링·episode 경계를 올바르게 처리함을 보인다.

증거 기록: 코드/노트/측정 결과 경로 = __________; 복습할 개념 = __________.

<a id="check-cs-2"></a>
### CS-2. 컴퓨터 구조 — Computer Architecture

개념·실습·자료: [커리큘럼 CS-2](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#cs-2>).

- [ ] **CS2-01 · 수와 데이터 표현** — Binary/Hexadecimal Representation, Signed/Unsigned Integer, Two's Complement, Integer Overflow, Endianness, IEEE 754 Floating-point, Rounding, NaN/Infinity, FP32/FP16/BF16의 정밀도·범위.
- [ ] **CS2-02 · 디지털 논리** — Boolean Algebra, Logic Gate, Combinational/Sequential Logic, Multiplexer, Adder, Flip-flop, Register, Finite State Machine(FSM).
- [ ] **CS2-03 · 명령어와 CPU** — Stored-program Concept, Instruction Set Architecture(ISA), Assembly Language, Addressing Mode, Register File, Arithmetic Logic Unit(ALU), Datapath, Control Unit, Fetch–Decode–Execute.
- [ ] **CS2-04 · 프로그램의 기계 표현** — Compile–Assemble–Link–Load, Machine Code, Calling Convention, Call Stack, Stack Frame, Pointer Arithmetic, Memory Alignment.
- [ ] **CS2-05 · 파이프라인** — Instruction Pipelining, Instruction-level Parallelism, Structural/Data/Control Hazard, Forwarding, Stall, Branch Prediction.
- [ ] **CS2-06 · 메모리 계층** — Register/Cache/DRAM/Storage Hierarchy, Cache Line, Temporal/Spatial Locality, Cache Hit/Miss, Set Associativity, Write-through/Write-back, Memory Latency/Bandwidth.
- [ ] **CS2-07 · 성능 분석** — CPU Time, Clock Rate, Cycles per Instruction(CPI), Latency versus Throughput, Amdahl's Law, Profiling, Compute-bound/Memory-bound Workload.
- [ ] **CS2-08 · 병렬 하드웨어** — Multicore, SIMD, SIMT, GPU Thread/Block 개념, Memory Coalescing, Cache Coherence, Memory Consistency, False Sharing.
- [ ] **CS2-09 · 장치와 데이터 이동** — Interrupt, Polling, Direct Memory Access(DMA), Memory-mapped I/O, Device Driver, Host–Device Transfer, Accelerator Synchronization.

- [ ] **GCS2 · 과목 완료 과제** — 정수·부동소수점 표현과 cache locality를 설명한다. 동일한 수치 결과 또는 허용 오차를 확인하면서 실행 시간·메모리 사용량을 측정하고, 연산량·메모리 접근·전송 중 병목을 증거로 구분한다. 처리량과 단일 제어 요청의 지연을 따로 보고한다.

증거 기록: 코드/노트/측정 결과 경로 = __________; 복습할 개념 = __________.

<a id="check-cs-3"></a>
### CS-3. 운영체제 — Operating Systems

개념·실습·자료: [커리큘럼 CS-3](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#cs-3>).

- [ ] **CS3-01 · OS와 실행 경계** — Kernel/User Mode, System Call, Trap/Exception, Interrupt, Protection, Resource Management.
- [ ] **CS3-02 · 프로세스와 스레드** — Process, Thread, Process State, Context Switch, Process Creation, fork/exec와 spawn 개념, Signal, Exit Status.
- [ ] **CS3-03 · 스케줄링** — Preemptive/Non-preemptive Scheduling, Round Robin, Priority Scheduling, Multi-level Feedback Queue(MLFQ), CPU Affinity, Starvation, Priority Inversion.
- [ ] **CS3-04 · 가상 메모리** — Virtual/Physical Address, Address Space, Paging, Page Table, Translation Lookaside Buffer(TLB), Page Fault, Demand Paging, Copy-on-Write, Swapping/Thrashing.
- [ ] **CS3-05 · 메모리 관리** — Stack/Heap Allocation, Allocation/Deallocation, Internal/External Fragmentation, Memory Leak, Memory Mapping(mmap), Page Replacement(FIFO/LRU/Clock).
- [ ] **CS3-06 · 동시성·동기화** — Concurrency versus Parallelism, Race Condition, Critical Section, Atomic Operation, Mutex, Semaphore, Condition Variable, Monitor, Memory Ordering.
- [ ] **CS3-07 · 교착과 진행성** — Deadlock, Coffman Conditions, Deadlock Prevention/Avoidance/Detection, Livelock, Starvation, Lock Ordering.
- [ ] **CS3-08 · 프로세스 간 통신·I/O** — Inter-process Communication(IPC), Pipe, Message Queue, Shared Memory, Socket, Blocking/Nonblocking I/O, Synchronous/Asynchronous I/O, Bounded Buffer, Backpressure.
- [ ] **CS3-09 · 파일·지속성·권한** — File Descriptor, File System, Directory, Metadata/Inode, Buffer/Page Cache, Flush versus Durable Write, Journaling, Crash Consistency, Permissions.
- [ ] **CS3-10 · 시간·실시간 실행** — Monotonic/Wall Clock, Timer, Deadline, Jitter, Hard/Soft Real-time, Worst-case Execution Time(WCET), Graceful Shutdown, Resource Cleanup.

- [ ] **GCS3 · 과목 완료 과제** — race condition을 재현·수정하고 프로세스·스레드·IPC 선택 이유를 설명한다. 느린 consumer에서도 queue·메모리가 무제한 증가하지 않으며 worker 종료 후 자원이 정리됨을 확인한다. 지연 분포와 deadline miss를 기록하고, 일반 OS·sleep·평균 FPS만으로 hard real-time을 보장하지 않는다.

증거 기록: 코드/노트/측정 결과 경로 = __________; 복습할 개념 = __________.

<a id="check-cs-4"></a>
### CS-4. 컴퓨터 네트워크 — Computer Networks

개념·실습·자료: [커리큘럼 CS-4](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#cs-4>).

- [ ] **CS4-01 · 계층과 통신 모델** — OSI/TCP-IP Layering, Encapsulation/Decapsulation, Packet Switching, Client–Server, Peer-to-Peer, End-to-end Principle.
- [ ] **CS4-02 · 성능과 시간** — Bandwidth, Throughput, Goodput, Transmission/Propagation/Queueing Delay, Round-trip Time(RTT), Jitter, Packet Loss, Clock Synchronization.
- [ ] **CS4-03 · 링크·무선 통신** — Ethernet, MAC Address, Frame, Switch, Error Detection/CRC, Wi-Fi, Shared Medium, Wireless Interference.
- [ ] **CS4-04 · 주소와 네트워크 계층** — IPv4/IPv6, Subnet Mask, CIDR, IP Forwarding, Routing Table, Default Gateway, ARP(IPv4), ICMP, MTU, Fragmentation/Path MTU 개념.
- [ ] **CS4-05 · 설정·응용 프로토콜** — DHCP, DNS, Port Number, HTTP Request/Response, Serialization/Deserialization, Data Schema.
- [ ] **CS4-06 · 전송 계층** — UDP Datagram, TCP Byte Stream, Connection Establishment/Termination, Sequence Number, Acknowledgment, Retransmission, Flow Control, Congestion Control, Head-of-line Blocking.
- [ ] **CS4-07 · Socket 프로그래밍** — Socket, bind/listen/accept/connect, Partial Read/Write, Message Framing, Length Prefix, Blocking/Nonblocking Socket, Timeout, Connection Reset.
- [ ] **CS4-08 · 응용 수준 신뢰성** — Application Sequence ID, Timestamp, Heartbeat, Retry, Duplicate Detection, Idempotency, Backpressure, Stale-message Rejection.
- [ ] **CS4-09 · 접근·보호·진단** — NAT, Firewall, Authentication/Authorization, Transport Layer Security(TLS), SSH, ping, traceroute, Socket Inspection, Packet Capture(Wireshark/tcpdump).
- [ ] **CS4-10 · 로봇 미들웨어 연결** — ROS 2 Publish–Subscribe, Middleware Abstraction(RMW), Data Distribution Service(DDS), Discovery, Quality of Service(QoS), Reliability, History/Depth, Durability, Deadline, Lifespan, Liveliness.

- [ ] **GCS4 · 과목 완료 과제** — TCP send/recv 호출과 메시지 경계가 일치하지 않는 경우를 처리한다. 재연결·누락·중복·오래된 메시지의 처리 규칙을 설명하고 RTT·처리량·누락을 기록한다. Reliable 전송이나 QoS deadline 설정을 최신성·실제 기한 준수의 보장으로 해석하지 않는다.

증거 기록: 코드/노트/측정 결과 경로 = __________; 복습할 개념 = __________.

<a id="check-stage-1"></a>
## 단계 1. 수학·확률·최적화

개념·실습·자료: [커리큘럼 단계 1](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#stage-1>).

- [ ] **S01-01 · 벡터 공간** — 선형 독립(Linear Independence), 기저(Basis), 차원(Dimension), 계수(Rank), 열 공간(Column Space), 영공간(Null Space), 내적(Inner Product), 노름(Norm), 직교 투영(Orthogonal Projection).
- [ ] **S01-02 · 행렬 분해·수치 해법** — 고유값·고유벡터(Eigenvalue/Eigenvector), 고유값 분해(Eigendecomposition), 특이값 분해(Singular Value Decomposition, SVD), 양의 정부호·준정부호(Positive Definite/Semidefinite), 조건수(Condition Number), 최소제곱(Least Squares), Moore–Penrose 의사역행렬(Pseudoinverse).
- [ ] **S01-03 · 다변수 미분** — 편미분(Partial Derivative), 연쇄법칙(Chain Rule), 기울기(Gradient), Jacobian, Hessian, Taylor 전개(Taylor Expansion), 행렬 미분(Matrix Calculus).
- [ ] **S01-04 · 확률 기초** — 조건부확률(Conditional Probability), Bayes 정리, 독립·조건부 독립(Independence/Conditional Independence), 결합·주변 분포(Joint/Marginal Distribution), 기댓값(Expectation), 분산(Variance), 공분산(Covariance).
- [ ] **S01-05 · 분포와 표본** — Bernoulli 분포, Categorical 분포, Uniform 분포, Gaussian/Multivariate Gaussian 분포, 큰 수의 법칙(Law of Large Numbers, LLN), 중심극한정리(Central Limit Theorem, CLT), Monte Carlo 추정.
- [ ] **S01-06 · 통계 추정** — 우도(Likelihood), 최대우도추정(Maximum Likelihood Estimation, MLE), 최대사후추정(Maximum a Posteriori Estimation, MAP), 편향·분산(Bias/Variance), 신뢰구간(Confidence Interval), Bootstrap.
- [ ] **S01-07 · 정보이론** — 엔트로피(Entropy), 교차 엔트로피(Cross-entropy), Kullback–Leibler Divergence(KL Divergence), Jensen 부등식.
- [ ] **S01-08 · 비제약 최적화** — 목적함수(Objective Function), 경사하강법(Gradient Descent), 확률적 경사하강법(Stochastic Gradient Descent, SGD), Momentum, Adam, 정규화(Regularization), 볼록성(Convexity).
- [ ] **S01-09 · 제약 최적화** — 가능집합(Feasible Set), Lagrange Multiplier, Lagrangian, Karush–Kuhn–Tucker 조건(KKT Conditions), 이차계획법(Quadratic Programming, QP)의 기본 형태.
- [ ] **S01-10 · 미분방정식과 이산화** — 상미분방정식(Ordinary Differential Equation, ODE), 초기값 문제(Initial Value Problem), 선형화(Linearization), Euler Integration, Runge–Kutta Integration, 이산화(Discretization), 수치 안정성(Numerical Stability).

- [ ] **G1 · 단계 완료 과제** — 최소제곱·Gaussian log-likelihood·chain rule을 설명하고, ill-conditioned 행렬에서 역행렬을 직접 계산하는 방식의 문제를 실험으로 보인다.

증거 기록: 코드/노트/실험 결과 경로 = __________; 아직 부족한 개념 = __________.

<a id="check-stage-2"></a>
## 단계 2. 머신러닝·딥러닝·표현 학습

개념·실습·자료: [커리큘럼 단계 2](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#stage-2>).

- [ ] **S02-01 · 지도학습 문제** — 회귀(Regression), 분류(Classification), 선형회귀(Linear Regression), 로지스틱 회귀(Logistic Regression), 경험적 위험 최소화(Empirical Risk Minimization, ERM).
- [ ] **S02-02 · 손실·확률 출력** — 평균제곱오차(Mean Squared Error, MSE), 교차 엔트로피(Cross-entropy), 음의 로그우도(Negative Log-likelihood, NLL), Gaussian NLL, 다봉성(Multimodality).
- [ ] **S02-03 · 일반화·모델 선택** — 과소적합·과적합(Underfitting/Overfitting), Bias–Variance Trade-off, Train/Validation/Test Split, Data Leakage, Distribution Shift, Hyperparameter Selection, Early Stopping.
- [ ] **S02-04 · 전처리·정규화** — Standardization, Min–Max Scaling, Feature Normalization, Data Augmentation, L2 Regularization, Weight Decay, Dropout.
- [ ] **S02-05 · 기본 신경망** — 다층 퍼셉트론(Multilayer Perceptron, MLP), Affine Layer, ReLU, Tanh, Sigmoid, Forward Pass, Backpropagation, Xavier/Glorot Initialization, He/Kaiming Initialization.
- [ ] **S02-06 · 학습 안정화** — Mini-batch SGD, Adam, AdamW, Learning-rate Schedule, Gradient Clipping, Vanishing/Exploding Gradients, Batch Normalization, Layer Normalization.
- [ ] **S02-07 · PyTorch 자동미분** — Computational Graph, Reverse-mode Automatic Differentiation, Leaf Tensor, requires_grad, backward, Gradient Accumulation, detach, no_grad.
- [ ] **S02-08 · PyTorch 학습 구성** — nn.Module, Parameter, forward, Dataset, DataLoader, Optimizer, zero_grad, optimizer.step, state_dict, model.train, model.eval.
- [ ] **S02-09 · 시각 표현 입문** — 합성곱(Convolution), Kernel, Stride, Padding, Receptive Field, Convolutional Neural Network(CNN), Residual Connection, ResNet.
- [ ] **S02-10 · 전이·표현 학습 입문** — Feature Representation, Pretraining, Transfer Learning, Frozen Encoder, Fine-tuning, Self-supervised Learning, Contrastive Learning.

- [ ] **G2 · 단계 완료 과제** — 작은 배치를 의도적으로 overfit해 학습 루프가 작동함을 보이고, 별도 validation 성능을 측정한다. `train()`/`eval()`과 gradient 흐름을 설명한다. 모델·optimizer·정규화와 사용 중인 scheduler 등의 상태를 저장·복원해 학습을 재개한다. validation 악화가 반드시 발생해야 완료하는 것은 아니다.

증거 기록: 코드/노트/실험 결과 경로 = __________; 아직 부족한 개념 = __________.

<a id="check-stage-3"></a>
## 단계 3. 로봇 기구학·동역학·접촉

개념·실습·자료: [커리큘럼 단계 3](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#stage-3>).

- [ ] **S03-01 · 로봇 구조** — 자유도(Degrees of Freedom, DoF), Revolute Joint, Prismatic Joint, Kinematic Chain, Actuator, Encoder, End-effector, Gripper, Workspace.
- [ ] **S03-02 · 좌표·변환** — Reference Frame, World/Base/Tool/Camera Frame, Active/Passive Rotation, Homogeneous Transformation, Change of Coordinates, Transformation Composition.
- [ ] **S03-03 · 회전 표현** — Special Orthogonal Group SO(3), Rotation Matrix, Euler Angles, Axis–Angle, Unit Quaternion, Quaternion Double Cover, Rotation Geodesic Distance.
- [ ] **S03-04 · 강체 운동의 기하** — Special Euclidean Group SE(3), Lie Algebra so(3)/se(3), Exponential/Logarithmic Map, Screw Axis, Twist, Wrench, Adjoint Representation.
- [ ] **S03-05 · 정기구학** — Forward Kinematics(FK), Product of Exponentials(PoE), Denavit–Hartenberg Parameters(DH Parameters).
- [ ] **S03-06 · 속도 기구학** — Space Jacobian, Body Jacobian, Differential Kinematics, Kinematic Singularity, Manipulability, Kinematic Redundancy, Null-space Motion.
- [ ] **S03-07 · 역기구학** — Inverse Kinematics(IK), Numerical IK, Newton–Raphson Method, Jacobian Pseudoinverse, Damped Least Squares(DLS), Joint-limit Constraints, Position/Orientation Error.
- [ ] **S03-08 · 강체 동역학** — Mass Matrix, Inertia Tensor, Lagrange Equations, Newton–Euler Equations, Coriolis/Centrifugal Terms, Gravity Compensation, Forward/Inverse Dynamics.
- [ ] **S03-09 · 접촉·파지** — Unilateral Contact Constraint, Coulomb Friction, Friction Cone, Contact Mode, Grasp Matrix, Grasp Wrench Space, Force Closure, Form Closure, Antipodal Grasp, Slip.
- [ ] **S03-10 · 로봇 모델 파일** — Unified Robot Description Format(URDF), MuJoCo XML Format(MJCF), Visual/Collision Geometry, Inertial Parameters, Joint Limits, Actuator Model.

- [ ] **G3 · 단계 완료 과제** — FK를 simulator와 비교하고 오차를 수치화한다. numerical Jacobian과 해석 Jacobian이 일치하는지 확인한다. IK 실패가 도달 불가능·관절 제한·특이점·잘못된 frame 중 무엇 때문인지 분류한다.

증거 기록: 코드/노트/실험 결과 경로 = __________; 아직 부족한 개념 = __________.

<a id="check-stage-4"></a>
## 단계 4. 제어·모션 플래닝·지각

개념·실습·자료: [커리큘럼 단계 4](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#stage-4>).

- [ ] **S04-01 · 시스템 표현** — State-space Model, Equilibrium Point, Linearization, Continuous-time/Discrete-time System, Controllability, Observability.
- [ ] **S04-02 · 피드백 제어** — Feedback/Feedforward Control, P/PD/PID Control, Tracking Error, Steady-state Error, Overshoot, Settling Time, Actuator Saturation, Integral Windup, Anti-windup.
- [ ] **S04-03 · 모델 기반 제어·안정성** — Gravity Compensation, Computed-torque Control, Linear Quadratic Regulator(LQR), Riccati Equation, Lyapunov Stability, Lyapunov Function.
- [ ] **S04-04 · Task-space·접촉 제어** — Joint-space Control, Task-space Control, Operational-space Control, Impedance Control, Admittance Control, Stiffness, Damping, Hybrid Motion–Force Control.
- [ ] **S04-05 · 경로 계획** — Configuration Space(C-space), Free/Obstacle Space, Collision Checking, Probabilistic Roadmap(PRM), Rapidly-exploring Random Tree(RRT), RRT-Connect.
- [ ] **S04-06 · 궤적·최적제어** — Path versus Trajectory, Trajectory Interpolation, Time Parameterization, Trajectory Optimization, Model Predictive Control(MPC), Receding Horizon.
- [ ] **S04-07 · 카메라 기하** — Pinhole Camera Model, Camera Intrinsics/Extrinsics, Radial/Tangential Distortion, Camera Calibration, Hand–Eye Calibration, Reprojection Error.
- [ ] **S04-08 · 3D 지각** — RGB-D, Depth Back-projection, Point Cloud, Object Segmentation, Perspective-n-Point(PnP), Iterative Closest Point(ICP), Rigid Registration, 6D Object Pose.
- [ ] **S04-09 · 상태 추정** — Bayesian Filtering, Kalman Filter(KF), Extended Kalman Filter(EKF), Process Noise, Measurement Noise, Sensor Fusion.
- [ ] **S04-10 · 시간·실행 계층** — Sampling Period, Zero-order Hold(ZOH), Control Frequency, Policy Frequency, Latency, Jitter, Timestamp Synchronization.

- [ ] **G4 · 단계 완료 과제** — 시뮬레이터에서 학습 없는 기본 task를 수행하고 gain·latency·mass 변화에 따른 추종 오차와 진동을 분석한다. 물체 크기와 작업 허용오차에 맞춰 위치·회전 오차 기준을 정한다. SO-101 실습에서는 지원되는 position target을 사용하며 토크·impedance 제어를 기본 지원한다고 가정하지 않는다.

증거 기록: 코드/노트/실험 결과 경로 = __________; 아직 부족한 개념 = __________.

<a id="check-stage-5"></a>
## 단계 5. 강화학습의 원리와 연속 제어

개념·실습·자료: [커리큘럼 단계 5](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#stage-5>).

- [ ] **S05-01 · 순차 의사결정** — Markov Decision Process(MDP), Partially Observable MDP(POMDP), Markov Property, State, Observation, Action, Transition Kernel, Reward, Policy, Return, Discount Factor, Horizon.
- [ ] **S05-02 · 가치 함수** — State-value Function V, Action-value Function Q, Advantage Function A, Bellman Expectation Equation, Bellman Optimality Equation, Bellman Operator, Contraction Mapping.
- [ ] **S05-03 · 표 기반 학습** — Policy Evaluation, Policy Improvement, Policy Iteration, Value Iteration, Monte Carlo Prediction, Temporal-difference Learning TD(0), n-step Return, Lambda Return, Eligibility Trace, TD(lambda), SARSA, Q-learning.
- [ ] **S05-04 · 학습 데이터·탐색** — On-policy/Off-policy Learning, Exploration–Exploitation Trade-off, Epsilon-greedy, Importance Sampling, Bootstrapping, Function Approximation, Deadly Triad.
- [ ] **S05-05 · 심층 가치 학습** — Deep Q-Network(DQN), Experience Replay, Target Network, Double Q-learning, Overestimation Bias.
- [ ] **S05-06 · 정책 기울기** — Policy Gradient Theorem, Likelihood-ratio/Score-function Estimator, REINFORCE, Reward-to-go, Baseline, Actor–Critic, Generalized Advantage Estimation(GAE).
- [ ] **S05-07 · PPO** — Proximal Policy Optimization(PPO), Importance Probability Ratio, Clipped Surrogate Objective, Rollout Buffer, Advantage Normalization, Entropy Bonus, Approximate KL.
- [ ] **S05-08 · 결정론적 연속 제어** — Deterministic Policy Gradient(DPG), Deep Deterministic Policy Gradient(DDPG), Twin Delayed DDPG(TD3), Clipped Double Q-learning, Delayed Policy Update, Target Policy Smoothing.
- [ ] **S05-09 · SAC** — Soft Actor-Critic(SAC), Maximum-entropy RL, Soft Bellman Backup, Stochastic Actor, Twin Critics, Entropy Temperature, Automatic Temperature Tuning, Reparameterization Trick.
- [ ] **S05-10 · 연속 행동·target 구현** — Squashed Gaussian Policy, Tanh Change-of-variables Log-probability Correction, Action Rescaling, Polyak/Soft Target Update, Replay Warm-up, Update-to-data Ratio(UTD).
- [ ] **S05-11 · 종료 의미** — Terminal State, Termination, Time-limit Truncation, Bootstrap Mask, Final Observation, Vectorized Environment Auto-reset.

- [ ] **G5 · 단계 완료 과제** — 선택한 PPO/SAC 구현의 buffer·target·actor/critic loss를 코드와 식으로 연결한다. 실제 termination에서는 bootstrap을 끊고, 외부 시간 제한 truncation에서는 reset 전 최종 관측으로 bootstrap하는지 작은 예제로 검증한다. 여러 학습 seed의 곡선과 최종 평가를 보고한다. 학습용 비교는 3개 seed부터 시작할 수 있으나, 그 수만으로 통계적 충분성을 주장하지 않는다.

종료 처리의 기준은 [Gymnasium 시간 제한 문서](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)를 따른다. task 자체에 정해진 유한 horizon의 종료는 외부 시간 제한과 다르며, Markov 상태를 구성하려면 남은 시간 정보도 고려한다.

증거 기록: 코드/노트/실험 결과 경로 = __________; 아직 부족한 개념 = __________.

<a id="check-stage-6"></a>
## 단계 6. 모방학습·시연 데이터·DAgger

개념·실습·자료: [커리큘럼 단계 6](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#stage-6>).

- [ ] **S06-01 · 시연의 구조** — Demonstration, Trajectory/Episode, Observation–Action Pair, Proprioception, Observation History, Action Label, Timestamp Alignment.
- [ ] **S06-02 · 기본 모방학습** — Behavior Cloning(BC), Supervised Policy Learning, MSE Regression, Maximum-likelihood Policy Estimation, Gaussian Policy, Mixture Density Network(MDN).
- [ ] **S06-03 · 순차 분포 변화** — Covariate Shift, Expert/Learner State Distribution, Compounding Error, State-distribution Coverage, Recovery Demonstration.
- [ ] **S06-04 · 상호작용형 IL** — Dataset Aggregation(DAgger), Expert Oracle, Expert Query Budget, Learner Rollout, Expert Action Relabeling, Policy Mixing.
- [ ] **S06-05 · 부분 관측·인과 문제** — Partial Observability, History-conditioned Policy, Hidden State, Causal Confusion, Privileged Information.
- [ ] **S06-06 · 데이터 품질** — Demonstrator Variability, Multimodal Action Distribution, Idle Frames, Failed Demonstration, Corrective Demonstration, Temporal Misalignment.
- [ ] **S06-07 · 데이터 분할** — Episode-level Split, Session-level Split, Object-instance Holdout, Overlapping-window Leakage, Train-only Normalization.
- [ ] **S06-08 · 실행 평가** — Open-loop Action Prediction, Closed-loop Rollout, Action Prediction Error, Task Success Rate, Autonomous/Assisted Success.

- [ ] **G6 · 단계 완료 과제** — 한 episode의 field·shape·단위·주기·행동 의미를 설명하고 episode 분할·train-only 정규화를 적용한다. 시뮬레이터 또는 실물에서 closed-loop BC를 평가하고 복구 시연 추가 전후의 결과를 비교한다. 단순 시연 추가와 learner 방문 상태에 대한 expert relabeling을 구별하며, DAgger 실습에서는 전문가와 질의 횟수를 명시한다.

증거 기록: 코드/노트/실험 결과 경로 = __________; 아직 부족한 개념 = __________.

<a id="check-stage-7"></a>
## 단계 7. 매니퓰레이터 강화학습

개념·실습·자료: [커리큘럼 단계 7](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#stage-7>).

- [ ] **S07-01 · 조작 task 구성** — Reaching, Pushing, Grasping, Lifting, Pick-and-place, Initial-state Distribution, Reset Distribution, Success Predicate.
- [ ] **S07-02 · 목표 조건화** — Goal-conditioned Policy, Goal-conditioned Value Function, Desired Goal, Achieved Goal, Goal Distribution.
- [ ] **S07-03 · 보상 설계** — Sparse/Dense Reward, Reward Scaling, Reward Shaping, Potential-based Reward Shaping, Reward Hacking, Success Detection.
- [ ] **S07-04 · HER** — Hindsight Experience Replay(HER), Goal Relabeling, Future-goal Sampling, Reward Recalculation, Goal-dependent Termination.
- [ ] **S07-05 · 행동·제어 표현** — Joint-position/Velocity/Torque Action, Absolute/Delta Action, End-effector Pose Action, Base/Tool-frame Action, Gripper Command, Action Bounds.
- [ ] **S07-06 · 관측 설계** — Proprioceptive Observation, Object-state Observation, Visual Observation, Privileged State, Observation History, Sensor Noise, Observation/Action Delay.
- [ ] **S07-07 · 탐색·교육과정** — Curriculum Learning, Goal Curriculum, Initial-state Curriculum, Demonstration Initialization, Exploration Noise.
- [ ] **S07-08 · 시뮬레이션 실행** — Physics Timestep, Control Decimation/Action Repeat, Contact Solver, Vectorized Simulation, Domain Randomization.
- [ ] **S07-09 · 자원·일반화** — Sample Efficiency, Wall-clock Efficiency, Environment Interaction Budget, ID/OOD Evaluation, Goal Interpolation/Extrapolation.

- [ ] **G7 · 단계 완료 과제** — 성공률·환경 상호작용 수·실행 시간·학습 seed별 변동을 함께 보고한다. 성공으로 오판된 grasp, table collision, 물체를 튕겨 보내는 reward hacking 등을 검사한다. 알고리즘이 개선되지 않아도 원인을 증거로 좁힐 수 있어야 한다.

증거 기록: 코드/노트/실험 결과 경로 = __________; 아직 부족한 개념 = __________.

<a id="check-stage-8"></a>
## 단계 8. 시각·시퀀스·생성형 모방학습

개념·실습·자료: [커리큘럼 단계 8](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#stage-8>).

- [ ] **S08-01 · 시각 모방학습** — Visual Behavior Cloning, CNN/ResNet Encoder, Image Normalization, Random Crop/Color Augmentation, Proprioceptive Fusion, Frozen Encoder, Fine-tuning.
- [ ] **S08-02 · 시간 정보** — Frame Stacking, Recurrent Neural Network(RNN), Long Short-Term Memory(LSTM), Recurrent BC(BC-RNN), Hidden-state Reset, Sequence Batching, Padding Mask.
- [ ] **S08-03 · Attention** — Scaled Dot-product Attention, Self-attention, Cross-attention, Multi-head Attention, Positional Encoding, Transformer Encoder/Decoder, Causal Mask.
- [ ] **S08-04 · 잠재변수 모델** — Latent Variable, Variational Autoencoder(VAE), Conditional VAE(CVAE), Evidence Lower Bound(ELBO), KL Regularization, Posterior/Conditional Prior, Reparameterization Trick.
- [ ] **S08-05 · ACT** — Action Chunking with Transformers(ACT), Action Chunking, Conditional Action-sequence Prediction, Temporal Ensembling, Chunk Boundary.
- [ ] **S08-06 · Diffusion** — Denoising Diffusion Probabilistic Model(DDPM), Forward Noising Process, Noise Schedule, Conditional Denoising, Noise Prediction, Score Function, Reverse Sampling.
- [ ] **S08-07 · Diffusion Policy** — Conditional Action Diffusion, Observation Horizon, Prediction Horizon, Action Execution Horizon, Receding-horizon Control, Multimodal Action Generation.
- [ ] **S08-08 · Flow matching 입문** — Conditional Flow Matching, Probability Path, Velocity Field, Vector-field Regression, ODE Sampling.
- [ ] **S08-09 · 실시간 추론** — Inference Latency, Latency Jitter, Stale Observation, Synchronous/Asynchronous Inference, Open-loop Execution Length, Action Discontinuity.

- [ ] **G8 · 단계 완료 과제** — ACT/Diffusion 중 하나를 BC 계열 baseline과 비교하고 prediction horizon과 실제 실행 길이를 구별한다. 수집·학습·추론의 action scaling·전처리·시간 간격이 호환되며, 다르면 변환·실행 방식을 기록한다. offline loss·closed-loop 성공률·추론 지연을 함께 분석한다. 특정 성능 역전이나 실패가 발생해야만 완료하는 것은 아니다.

증거 기록: 코드/노트/실험 결과 경로 = __________; 아직 부족한 개념 = __________.

<a id="check-stage-9"></a>
## 단계 9. Offline RL·RL+IL·model-based learning

개념·실습·자료: [커리큘럼 단계 9](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#stage-9>).

- [ ] **S09-01 · Offline RL 문제** — Offline/Batch Reinforcement Learning, Behavior Policy, Dataset Support/Coverage, Distribution Shift, Extrapolation Error, Out-of-distribution Action.
- [ ] **S09-02 · 보수적·제약적 학습** — Conservative Q-Learning(CQL), Conservative Value Estimation, Value Regularization, Behavior/Policy Constraint.
- [ ] **S09-03 · IQL** — Implicit Q-Learning(IQL), Expectile Regression, Asymmetric Squared Loss, Value/Q Fitting, Advantage-weighted Behavior Cloning.
- [ ] **S09-04 · Offline-to-online** — BC Initialization, Demonstration Replay, Replay Mixing Ratio, Advantage-weighted Actor–Critic(AWAC), Reinforcement Learning with Prior Data(RLPD), Online Fine-tuning.
- [ ] **S09-05 · 보상·정책 평가** — Reward Annotation, Terminal Label, Reward Classifier, False Positive/Negative, Offline Policy Evaluation(OPE), Importance-sampling OPE, Fitted Q Evaluation(FQE).
- [ ] **S09-06 · 학습 동역학** — Learned Dynamics Model, One-step/Multi-step Prediction Error, Model Ensemble, Epistemic/Aleatoric Uncertainty, Compounding Model Error, Model Exploitation.
- [ ] **S09-07 · 모델 기반 계획** — Model-based RL, Model Predictive Control(MPC), Random Shooting, Cross-entropy Method(CEM), Probabilistic Ensembles with Trajectory Sampling(PETS).
- [ ] **S09-08 · 잔차·적응** — Residual Reinforcement Learning, Residual Action, Baseline Controller/Policy, Action Saturation, System Identification, Domain Randomization.

- [ ] **G9 · 단계 완료 과제** — BC가 offline RL보다 좋은 결과도 설명한다. 데이터 coverage와 reward 품질을 확인하지 않고 offline RL이 항상 시연을 능가한다고 주장하지 않는다. IQL/CQL 등 하나의 update를 식과 코드로 설명한다.

증거 기록: 코드/노트/실험 결과 경로 = __________; 아직 부족한 개념 = __________.

<a id="check-stage-10"></a>
## 단계 10. SO-ARM-101 시스템 통합과 실제 평가

개념·실습·자료: [커리큘럼 단계 10](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#stage-10>).

- [ ] **S10-01 · 장비·캘리브레이션** — Leader/Follower Teleoperation, Motor ID, Serial Communication, Calibration, Joint Zero Offset, Joint Range, Calibration Identifier.
- [ ] **S10-02 · 명령과 관측** — Position Setpoint, Measured Joint Position, Commanded versus Executed Action, Joint Ordering, Action Normalization/Denormalization, Gripper Scaling.
- [ ] **S10-03 · 데이터·정책 파이프라인** — Observation/Action Schema, Data Recording, Dataset Visualization, Policy Checkpoint, Preprocessing/Postprocessing, Policy Rollout.
- [ ] **S10-04 · 시간 동기화** — Sensor Timestamp, Clock Offset, Sampling Frequency, End-to-end Latency, Camera Buffering, Dropped Frame, Control Deadline.
- [ ] **S10-05 · 실행 제한·복구** — Joint/Velocity/Workspace Limit, Rate Limiting, Command Clipping, Communication Timeout, Watchdog, Stop Mechanism, Reset/Recovery Procedure.
- [ ] **S10-06 · 실물 오차** — Backlash, Mechanical Compliance, Joint Friction, Actuator Saturation, Calibration Drift, Camera Extrinsic Drift, Payload Variation.
- [ ] **S10-07 · 전이의 기본 개념** — Reality Gap, System Identification, Dynamics/Visual Domain Randomization, Observation/Action Delay Randomization, Sim-to-real Transfer, Real-world IL.
- [ ] **S10-08 · 실물 실험 평가** — Held-out Initial Conditions, Session Shift, Autonomous/Assisted Success, Human Intervention Count, Reset Cost, Failure Taxonomy.

- [ ] **G10 · 단계 완료 과제** — 새로운 실행 세션에서 setup과 평가를 재현한다. 같은 task의 nominal 조건과 별도 물체 위치·조명 조건을 비교한다. sim stress test만 수행했다면 sim-to-real을 검증했다고 쓰지 않는다. 실물에서 직접 시연으로 학습했다면 real-world IL이라고 명확히 표현한다.

증거 기록: 코드/노트/실험 결과 경로 = __________; 아직 부족한 개념 = __________.

<a id="check-stage-11"></a>
## 단계 11. 재현에서 독립 연구로

개념·실습·자료: [커리큘럼 단계 11](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#stage-11>).

- [ ] **S11-01 · 문제와 가설** — Research Question, Falsifiable Hypothesis, Operational Definition, Assumption, Scope.
- [ ] **S11-02 · 비교 실험** — Baseline, Control Condition, Independent/Dependent Variable, Confounder, Ablation Study, Sensitivity Analysis.
- [ ] **S11-03 · 재현성** — Computational Reproducibility, Independent Replication, Code/Environment Versioning, Dataset Versioning, Run Manifest.
- [ ] **S11-04 · 선택·평가 분리** — Validation-based Model Selection, Hyperparameter Search Budget, Held-out Test Set, Test-set Leakage, Evaluation Protocol.
- [ ] **S11-05 · 통계 단위** — Training Seed, Evaluation Episode, Independent Replicate, Within-seed/Between-seed Variability, Clustered Data.
- [ ] **S11-06 · 불확실성·효과** — Effect Size, Confidence Interval, Bootstrap, Hierarchical/Cluster Bootstrap, Binomial Proportion Interval, Statistical versus Practical Significance.
- [ ] **S11-07 · 일반화·타당성** — In-distribution/Out-of-distribution Generalization, Internal/External Validity, Pretraining Data Contamination, Distribution Shift.
- [ ] **S11-08 · 연구 보고** — Negative Result, Limitation, Failure Analysis, Dataset Card, Experiment Log, Artifact, Cost Accounting.

- [ ] **G11 · 단계 완료 과제** — 개선 여부와 무관하게 연구 질문·가설·비교군·ablation·불확실성·한계를 포함한 보고서와 재현 자료를 제공한다. 알고리즘 변경의 효과와 데이터·하드웨어·계산량 증가의 효과를 구별한다.

증거 기록: 코드/노트/실험 결과 경로 = __________; 아직 부족한 개념 = __________.

<a id="pytorch-practice"></a>
## PyTorch 실습 점검

- [ ] tensor shape를 실행 전에 예상하고 `assert`로 확인할 수 있다.
- [ ] `requires_grad`, leaf tensor, `backward`, `detach`, `no_grad`의 역할을 설명한다.
- [ ] optimizer가 어떤 parameter를 업데이트하는지 확인한다.
- [ ] `model.train()`/`model.eval()`과 gradient enable/disable이 서로 다른 설정임을 안다.
- [ ] Dataset에서 observation/action을 올바른 timestep으로 반환한다.
- [ ] loss 감소와 gradient norm을 함께 확인한다.
- [ ] checkpoint를 다시 로드해 같은 입력의 출력을 비교한다.
- [ ] 연속 action의 scale, mask, normalization을 저장하고 복원한다.
- [ ] image augmentation과 normalization의 train/eval 차이를 이해한다.
- [ ] 처음 보는 학습 코드를 data → model → loss → update → eval 순으로 읽는다.
- [ ] seed 고정과 실행의 완전한 결정론을 구별한다. Python·NumPy·PyTorch·환경의 난수와 사용한 device·버전을 기록하고, checkpoint 재개 시 필요한 난수 상태도 보존한다.

서로 다른 플랫폼·버전에서 같은 seed가 완전히 동일한 결과를 보장하지 않는다는 점은 [PyTorch 재현성 문서](https://docs.pytorch.org/docs/stable/notes/randomness.html)를 참고한다.

<a id="implementation-gates"></a>
## 학습 구현·실물 실행 점검

관련 단계의 구현을 시작할 때 사용한다. F1~F3은 단계 5, F4~F6은 단계 6~8, F7~F9는 단계 10에 연결된다. 선택하지 않은 알고리즘·센서에는 해당 항목을 적용하지 않는다.

- [ ] **F1 · RL gradient 경로** — critic update의 Bellman target에는 gradient를 흘리지 않고, PPO의 저장된 old log-probability는 해당 update의 고정 기준으로 사용한다. SAC/TD3의 actor update에서는 행동을 거쳐 필요한 gradient가 actor에 전달되는지 확인한다. `[B]`와 `[B, 1]`의 잘못된 broadcasting도 검사한다.
- [ ] **F2 · Episode 경계** — reset 후 초기 관측을 직전 transition의 next observation으로 잘못 저장하지 않는다. GAE의 value bootstrap과 다음 episode로 넘어가지 않게 하는 재귀 계산 경계를 구별하고, recurrent hidden state도 올바르게 초기화한다.
- [ ] **F3 · 행동과 확률** — 사용한 stochastic policy에서 다차원 행동의 joint log-probability와 행동 변환을 일관되게 처리한다. SAC의 tanh 보정 및 PPO의 old/new probability ratio가 코드에서 가정한 행동 표본과 대응하는지 확인한다.
- [ ] **F4 · 시연 label의 의미** — 조작자의 원래 명령, clipping·변환 후 실제 전송 명령, follower의 측정 위치를 구별한다. 저장된 action label이 무엇인지 명시하고 관측 시점과 맞춘다. 측정 위치를 목표 명령과 동일하다고 가정하지 않는다.
- [ ] **F5 · Sequence와 loss** — history·action chunk가 다른 episode로 넘어가지 않으며 padding된 행동이 loss에 포함되지 않는지 확인한다. 유효 timestep과 action 차원에 대한 loss reduction을 명시한다.
- [ ] **F6 · 실제 관측의 충분성** — actor 입력이 실행 시점에 얻을 수 있는 정보로만 구성된다. 관절 상태만으로 물체 위치를 알 수 없는 task라면 영상·물체 추정치 등 필요한 관측을 추가하거나 task 조건을 제한한다.
- [ ] **F7 · SO-101 인터페이스** — 팔 관절 5개와 gripper 구동을 구별하고, 표준 LeRobot follower의 position target·joint order·정규화·gripper 범위를 확인한다. 모터 수를 임의의 6D tool pose 제어 가능성으로 해석하지 않는다.
- [ ] **F8 · 실행 제한과 중지** — 실제 사용하는 명령 경로에 관절 범위·변화량·timeout 처리가 적용되는지 낮은 위험의 작은 동작으로 확인한다. 사람이 즉시 중지할 방법과 재시작 절차를 확인하고, 소프트웨어 clipping만으로 충돌이 방지된다고 가정하지 않는다.
- [ ] **F9 · 지연·장시간 구동** — 실제 제어 간격·추론 지연·오래된 영상·누락 frame을 측정한다. 반복 구동 중 전원·통신·구동기 상태와 사용 가능한 온도·오류 진단을 확인하고 제조사 동작 범위에 맞게 중단 조건을 정한다.

장비 해석은 [SO-101 공식 문서](https://huggingface.co/docs/lerobot/main/en/so101)와 [follower 공식 구현](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)을 기준으로 한다. 특히 `send_action`의 반환값은 clipping 후 **전송한 명령**이며, 로봇이 이미 도달한 실측 자세를 뜻하지 않는다. 실제 설치한 release/commit의 동작을 확인한다.

<a id="project-gates"></a>
## 프로젝트 완료 점검

- [ ] **P0** — NumPy 회귀·PyTorch MLP, gradient 검사, 저장·복원, 실행 README를 완성했다.
- [ ] **PCS · 선택** — CS 과목별 실습을 합성 센서 producer·bounded queue·처리 worker·socket/파일 저장으로 연결하고 병목·지연·누락·종료 처리를 분석했다.
- [ ] **P1** — FK/IK·좌표계·제어 baseline을 구현하고 추종 오차와 실패를 분석했다.
- [ ] **P2** — 시연 수집·재생·검사·episode split·dataset card와 첫 BC를 완성했다.
- [ ] **P3** — 시뮬레이션 manipulation RL을 baseline과 비교하고 seed별 결과를 보존했다.
- [ ] **P4** — SO-101의 BC와 ACT 또는 다른 sequence policy를 독립 조건에서 평가했다.
- [ ] **P5** — 논문 결과 하나를 재현하고 원논문과의 설정·성능 차이를 설명했다.
- [ ] **P6** — 연구 질문·ablation·불확실성·실패 분석·재현 자료를 갖춘 보고서를 작성했다.

<a id="evaluation-gates"></a>
## 실험·평가 점검

아래는 새로운 핵심 실험을 시작하거나 최종 결과를 정리할 때 다시 사용하는 목록이다.

- [ ] task·관측·행동·controller·주기·초기 상태·성공 조건을 고정했다.
- [ ] train/validation/test를 episode 단위로 분리하고 필요한 object/session holdout을 적용했다.
- [ ] 정규화 통계를 train에서만 계산했고 미래 관측·겹친 window의 누수가 없다.
- [ ] validation으로 model을 선택하고 독립 test 결과를 보고했다.
- [ ] offline action loss와 closed-loop 성공률을 구별했다.
- [ ] training seed와 evaluation episode 수를 따로 기록했다.
- [ ] 비교군의 데이터·관측 정보·제어·상호작용·튜닝 예산을 맞추거나 차이를 공개했다.
- [ ] 성공률의 평가 횟수·불확실성과 seed별 원자료를 함께 제시했다.
- [ ] ID/OOD·새 물체·새 위치·새 세션·지연 변화 결과를 구별했다.
- [ ] autonomous/assisted success, 인간 개입·reset·compute 비용을 기록했다.
- [ ] checkpoint·processor·normalization·calibration·software version을 보존했다.
- [ ] 실패 영상·실험 한계·negative result를 결과에 포함했다.

<a id="turtlebot-track"></a>
## TurtleBot3 선택 과정

매니퓰레이터 RL·IL의 필수 완료 조건은 아니다. ROS 시스템 역량을 보완할 때 사용한다.

- [ ] 보유 TurtleBot3의 모델·OS·ROS 배포판·firmware를 기록했다.
- [ ] ROS 2 Node, Topic, Service, Action, Parameter, Launch, Quality of Service(QoS)를 설명한다.
- [ ] tf2, URDF, RViz와 map/odom/base_link/sensor frame의 관계를 설명한다.
- [ ] Odometry, IMU, LiDAR의 역할과 timestamp를 이해하고 ROS bag을 기록·재생한다.
- [ ] SLAM, Localization, Navigation의 역할을 구별하고 navigation 실패를 분석한다.
- [ ] 선택적으로 mobile manipulation을 진행한다면 장착·무게중심·전원·base-to-arm transform·충돌 모델을 검증했다.
- [ ] 선택 통합 프로젝트에서 '이동 → 정지 → 재관측 → 조작'을 검증했다.

<a id="final-competencies"></a>
## 최종 역량 점검

### CS 공통 기반

- [ ] 자료구조와 탐색·정렬·DP 알고리즘을 정확성·시간·공간 복잡도에 근거해 선택한다.
- [ ] 데이터 표현·메모리 계층·CPU/GPU 실행을 설명하고 학습·추론 병목을 측정한다.
- [ ] 프로세스·스레드·메모리·동기화·I/O를 이해하고 제한된 버퍼와 정상 종료를 구현한다.
- [ ] TCP/IP·socket·통신 지연·누락을 이해하고 메시지 경계와 오래된 데이터 처리 규칙을 구현한다.

### 공통 기반

- [ ] Jacobian·확률 정책·최적화 손실을 유도하고 수치적으로 검증한다.
- [ ] PyTorch 학습·추론·데이터·checkpoint 파이프라인을 직접 작성한다.
- [ ] SO(3)/SE(3), FK/IK, singularity, frame·단위 오류를 설명한다.
- [ ] 동역학·접촉·제어 주기·position/torque interface의 차이를 이해한다.
- [ ] 학습 없는 로봇 제어 baseline을 구성한다.
- [ ] camera geometry·calibration·지연이 관측에 미치는 영향을 분석한다.

### RL·IL 핵심

- [ ] MDP/POMDP, Bellman equation, policy gradient와 actor–critic을 설명한다.
- [ ] PPO/SAC 하나를 직접 구현하고 reference와 비교한다.
- [ ] 매니퓰레이터의 reward·action·reset·goal 설계를 설명한다.
- [ ] BC의 오류 누적과 DAgger의 expert 가정을 설명한다.
- [ ] demonstration을 수집·검사·분할하고 closed-loop BC를 평가한다.
- [ ] ACT/Diffusion 중 하나를 재현하고 horizon·latency·데이터 효과를 분석한다.
- [ ] offline RL과 IL의 차이, CQL/IQL의 동기와 한계를 설명한다.
- [ ] model-based·residual·RL+IL의 활용 조건을 비교한다.

### 실제 시스템과 연구

- [ ] SO-101 leader/follower 데이터 수집과 정책 실행을 재현한다.
- [ ] 실물 제어의 지원 범위를 알고, 학습 데이터와 실행 action 의미를 일치시킨다.
- [ ] nominal·새 위치·새 물체·배경·지연 변화의 평가를 구별한다.
- [ ] 성공 영상뿐 아니라 시행 수·seed·불확실성·실패·인간 개입을 보고한다.
- [ ] 논문 결과 하나를 재현하고 차이가 난 원인을 설명한다.
- [ ] 반증 가능한 연구 질문과 공정한 ablation을 설계한다.
- [ ] 다른 사람이 실행할 수 있는 코드·데이터 설명·보고서를 제공한다.
- [ ] 관심 전문 분야 하나에서 가정과 한계를 비판적으로 토론한다.

이 최종 역량 목록이 실제 산출물로 채워지면, 매니퓰레이터 RL/IL 연구를 독립적으로 진행할 기반이 갖춰진 것이다. 다음 학습 내용은 새로운 유행의 수가 아니라, 현재 실험에서 설명하지 못한 실패와 해결하려는 연구 질문에 따라 정한다.

---

완료 증거는 [커리큘럼의 연구 기록 양식](<C:/Users/user/Documents/Codex/2026-09-17/sks/outputs/manipulator_rl_il_curriculum_ko.md#s9>)을 활용해 정리한다. 이 파일은 진도 기록용이며 공인 자격 평가표가 아니다.
