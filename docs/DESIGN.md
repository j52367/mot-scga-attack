# 설계 문서 — 패러다임 중립 시간 일관성 공격

> 작성: 김영재 · 최초 2026-07-30
> SCGA(ICLR 2026)를 발판으로, MOT를 대상으로 한 생성형 전이 공격

---

## 1. 한 문장

기존 MOT 공격은 특정 트래커에만 존재하는 구성요소(ReID 헤드, track query, YOLOX grid)에 손실이 결박되어 계열을 넘지 못한다.
**우리는 모든 검출기에 존재하는 백본 feature와 박스만으로 시간 일관성을 정의하고, 이를 generator로 교란한다.**

---

## 2. 왜 이 표적인가 — 코드 수준 근거

### 2.1 ByteTrack의 cost matrix는 재료가 두 개뿐이다

`yolox/tracker/matching.py`

```python
def iou_distance(atracks, btracks):
    atlbrs = [track.tlbr for track in atracks]   # 칼만 예측 박스
    btlbrs = [track.tlbr for track in btracks]   # 검출 박스
    cost_matrix = 1 - ious(atlbrs, btlbrs)

def fuse_score(cost_matrix, detections):
    fuse_sim = (1 - cost_matrix) * det_scores     # 최종 = 1 - IoU x score
```

| 단계 | 비용 | 임계값 | 통과 조건 |
|---|---|---|---|
| 1차 매칭 | `1 - IoU x score` | 0.8 | IoU x score >= 0.2 |
| 2차 매칭 | `1 - IoU` (점수 무시) | 0.5 | IoU >= 0.5 |

**따라서 신뢰도만 떨어뜨리는 공격은 2차 매칭에서 복구된다.**
BBA가 ByteTrack에서 MOTA를 4%밖에 떨어뜨리지 못한 이유가 이것이다.
→ **박스 위치를 흔들어야 한다.**

### 2.2 gradient는 검출기까지만 흐른다

`byte_tracker.py :: update()`

```python
output_results = output_results.detach().cpu().numpy()   # <- 경계선
```

이 아래(칼만, 헝가리안 `lap.lapjv`, 임계값, 생명주기)는 numpy/C이고 이산적이라 관통 불가.
**손실은 이 선 위쪽(검출기 백본/출력)에 걸어야 한다.**

### 2.3 기존 공격은 아키텍처에 결박되어 있다

`byte_tracker.py :: attack_sg()`

```python
Ws = [W // s for s in [8, 16, 32]]      # YOLOX 3-스케일 grid
att_hm_ind = att_hm_ind + (n_i-1)*Ws[0] + (n_j-1)
loss_att += ((1 - outputs[:, -1][att_idx])**2 * log(outputs[:, -1][att_idx])).mean()
```

- grid 인덱스 산술, 채널 위치(`[:, -1]`, `[:, -2]`)가 **YOLOX head 규격 전용**
- CenterNet/DETR로 옮기려면 전부 재작성해야 함
- TraSA의 `PushPull`은 **re-ID 브랜치** 필요 → ReID 없는 ByteTrack엔 절반만 적용

→ **백본은 어느 검출기에나 있지만, head 구조는 검출기마다 다르다.**

---

## 3. 제안 방법

### 3.1 객체 서술자 (object descriptor)

프레임 `t`, 검출기 백본 feature `F_t`, 박스 `b_k`에 대해

```
f_k  = ROIAlign(F_t, b_k)  ->  GAP  ->  R^C        # 외형 성분
g_k  = (cx, cy, w, h) / (W, H, W, H)                # 기하 성분 (정규화)
d_k  = [ L2norm(f_k) ;  alpha * g_k ]  in R^(C+4)
```

- `F_t`: FPN 중간 레벨 (SCGA의 `layer_idx`에 대응, 층 선택은 ablation)
- **박스 좌표는 검출기가 예측한 미분 가능한 텐서를 사용** (numpy 변환 이전)
  → 기하 성분에도 gradient가 흘러 박스가 실제로 움직임
- `alpha`: 외형과 기하의 균형 (하이퍼파라미터)

### 3.2 손실

같은 ID `k`, 다른 ID `j`에 대해

```
L_decorr  =   mean_k  cos( d_t[k], d_{t+1}[k] )            # 최소화: 같은 객체를 시간축에서 분리
L_confuse = - mean_{k != j}  cos( d_t[k], d_{t+1}[j] )     # 최소화: 다른 객체를 서로 근접
L_attack  = L_decorr + lambda * L_confuse
```

- `L_decorr` → 추적 끊김 (트랙 lost)
- `L_confuse` → ID 교환 (IDSW 증가)

### 3.3 ⭐ 왜 이것이 패러다임 중립인가

이 손실은 **트래커의 내부 구조를 전혀 참조하지 않는다.**
필요한 입력은 `(프레임, 박스, ID)` 뿐이며, 칼만·헝가리안·ReID 헤드·track query 어느 것도 쓰지 않는다.

| 트래커 | 시간 일관성 판정 근거 | 우리 서술자에서 닿는 성분 |
|---|---|---|
| ByteTrack | 칼만 예측 박스와 검출 박스의 IoU | **기하 성분** `g_k` |
| DeepSORT / FairMOT | ReID 임베딩 코사인 유사도 | **외형 성분** `f_k` (상류) |
| MOTR / MeMOTR | track query hidden state 유사도 | **외형 성분** `f_k` (query가 백본에 attend) |

백본은 헤드보다 상류이므로, 하나를 교란하면 박스 회귀·ReID 헤드·query가 **함께** 흔들린다.

### 3.4 전체 학습 루프 (SCGA 대응)

```python
for (img_t, img_t1, boxes_t, boxes_t1, ids) in loader:      # MOT17 GT 사용
    # ---- SCGA와 동일 ----
    with torch.no_grad():
        _, ft_t  = teacher(img_t,  feat=True)
        _, ft_t1 = teacher(img_t1, feat=True)
    adv_t,  fs_t  = student(img_t,  feat=True)
    adv_t1, fs_t1 = student(img_t1, feat=True)
    adv_t  = project(adv_t,  img_t,  eps)                   # L-inf + [0,1]
    adv_t1 = project(adv_t1, img_t1, eps)
    gen_sim = hinge(ft_t, fs_t) + hinge(ft_t1, fs_t1)       # SCGA 일관성

    # ---- 새로 추가 ----
    F_t,  P_t  = detector.backbone_and_boxes(adv_t)         # 미분 가능
    F_t1, P_t1 = detector.backbone_and_boxes(adv_t1)
    d_t  = descriptor(F_t,  P_t)
    d_t1 = descriptor(F_t1, P_t1)
    loss_attack = L_decorr(d_t, d_t1, ids) + lam * L_confuse(d_t, d_t1, ids)

    # ---- SCGA와 동일 ----
    loss = loss_attack + 0.7 * gen_sim
    loss.backward(); optimG.step(); update_ema(student, teacher)
```

**generator 구조와 EMA는 SCGA 코드를 그대로 재사용한다.**
바뀌는 것은 데이터(프레임 쌍), surrogate(분류 CNN → 검출기), 손실뿐이다.

---

## 4. 기존 연구와의 위치

### 4.1 네 조건으로 본 위치

| | 생성형 | 패러다임 간 전이 | ID 연결 공격 | 시간 지속성 |
|---|---|---|---|---|
| BBA (Neural Networks 2024) | O | X (ByteTrack -4%) | 약함 | X (매 프레임) |
| TraSA (Applied Soft Computing 2024) | X (60 iter 화이트박스) | X (블랙박스를 future work로 명시) | O | O (평균 4.11 프레임) |
| **BankTweak (IJCAI 2025)** | **X (반복, 화이트박스)** | **X (블랙박스 미제안이라 자평)** | **O (최대 11.8배)** | **O** |
| FADE (CVPR 2026) | X (PGD) | X (계보 내만, 미해결로 명시) | O | O |
| **제안** | **O** | **목표** | **O** | **목표** |

### 4.1.1 ⭐ 인용 가능한 gap 문장 세 개

서로 다른 세 논문이 **각자 독립적으로 같은 빈틈을 명시**했다. 이것이 본 연구의 정당성이다.

| 논문 | 원문 |
|---|---|
| TraSA (ASOC 2024) | "black-box attacks are more practical, and the development of efficient black-box attack strategies constitutes **one of the significant directions for future research**" |
| **BankTweak (IJCAI 2025)** | "there has been little progress in MOT attacks due to its inherent challenge. Consequently, **an MOT attack based on black-box methodologies has yet to be proposed.** Although BankTweak is also a white-box attack..." |
| FADE (CVPR 2026) | "robust **cross-architecture black-box transferability remains an open challenge**" |

⚠️ **정확히 인용할 것.** BankTweak도 transfer 기반 블랙박스 변형을 시도했고
(Fig. 5, "화이트박스보다는 덜 효과적이지만 임계값은 충족"),
"아무도 시도조차 하지 않았다"가 아니라 **"본격적인 블랙박스 방법론이 제안되지 않았다"** 가 정확한 표현이다.

### 4.1.2 아키텍처 결박의 네 사례

기존 MOT 공격이 계열을 넘지 못하는 이유는 우연이 아니라 **표적 선택의 결과**다.
네 논문 모두 특정 트래커에만 존재하는 구성요소를 노렸고, 그것이 없는 트래커에는 닿지 않는다.

| 논문 | 표적 | ByteTrack에서 |
|---|---|---|
| BBA | CenterNet 계열 ReID 헤드 | MOTA -4% (사실상 실패) |
| TraSA | re-ID 브랜치 (`PushPull`) | 절반만 적용 (`CenterLeaping`만) |
| **BankTweak** | **ReID feature bank** | **적용 불가 — 실험 대상에서 제외됨** |
| FADE | MOTR track query | TBD 계열에 존재하지 않음 |

**BankTweak의 실험 대상이 DeepSORT / StrongSORT / MOTDT 셋뿐인 것이 이를 잘 보여준다.**
셋 다 feature bank를 가진 ReID 기반 트래커이며, bank가 없는 ByteTrack은
공격 방식이 원리적으로 적용되지 않아 실험조차 하지 못했다.

### 4.2 ⚠️ 손실 구조는 신규성이 아니다 — 반드시 인정하고 갈 것

**"같은 객체는 멀게, 다른 객체는 가깝게"라는 push-pull(대조학습 반전) 구조는 이미 확립되어 있다.**

| 영역 | 대표 연구 |
|---|---|
| ReID 도메인 | Adversarial Metric Attack for ReID (2019), Vulnerability of ReID to Metric Attacks (CVPRW 2020), Prompt-driven Transferable Attack on ReID (2025), Diffusion-based attack on ReID (2025) |
| MOT 도메인 | TraSA의 손실 이름이 아예 `PushPull` / BBA의 `L_inter`(객체 간) + `L_intra`(프레임 간) |

따라서 **"Re-ID가 학습한 것을 반대로 부순다"를 신규성으로 주장하면 즉시 반박당한다.**

**정확한 신규성 = 그 반전을 어디서 수행하는가:**

| | 반전을 수행하는 공간 | 적용 가능한 트래커 |
|---|---|---|
| 기존 ReID metric attack | ReID 임베딩 공간 | ReID가 있는 모델만 |
| TraSA `PushPull` | re-ID 브랜치 | FairMOT, JDE |
| BBA `L_inter/L_intra` | CenterNet dense ReID 맵 | CenterNet 계열 |
| **제안** | **검출기 백본 feature + 박스 기하** | **ReID 없는 ByteTrack, query 기반 MOTR 포함** |

**관련 연구 절에 쓸 문장 (초안):**

> Push-pull 형태의 metric 공격은 ReID 도메인에서 잘 확립되어 있고, MOT에서도 TraSA의 PushPull과
> BBA의 inter/intra 손실이 같은 구조를 취한다. 그러나 이들은 모두 **ReID 임베딩이 존재한다는 전제**
> 위에 있다. ByteTrack처럼 ReID를 쓰지 않는 트래커에서는 이 전제가 성립하지 않으며, 실제로 BBA는
> ByteTrack에서 MOTA를 4%만 저하시켰다. 본 연구는 동일한 목적을 아키텍처에 무관한 표현 위에서
> 달성한다.

### 4.3 교훈 — 반복해서 걸린 함정

우아하고 자연스러운 프레이밍일수록 이미 존재한다. 실제로 두 번 걸렸다.

| 떠올린 프레이밍 | 실제 |
|---|---|
| "공간적 일관성을 시간적 일관성으로 확장" | BMTC(IJCAI 2025), FADE의 `L_Decorr` 등 다수 존재 |
| "Re-ID가 학습한 것을 반대로 부순다" | push-pull metric attack으로 확립된 분야 |

**신규성 주장은 계속 "무엇을 하는가"가 아니라 "어디를 때리는가"로 수렴한다.**
새 프레이밍이 떠오르면 주장하기 전에 반드시 검색부터 할 것.

---

## 5. 아직 정하지 못한 것

| 항목 | 후보 | 정하는 방법 |
|---|---|---|
| 백본 층 선택 | FPN stride 8 / 16 / 32 | 층별 ablation (A4) |
| pooling | ROIAlign vs crop+GAP | 구현 난이도와 성능 비교 |
| `alpha` (외형 vs 기하) | **1.0 부근에서 탐색** (아래 5.1 참고) | grid search (A2) |
| `lambda` (decorr vs confuse) | 0.5 ~ 2 | grid search (A3) |
| 공격 대상 | 전체 객체 vs 특정 ID | TraSA는 특정 ID |
| generator 입력 | 프레임 개별 (옵션 A) vs 시퀀스 | 옵션 A로 시작 |

### 5.1 CPU 스모크 테스트로 확인한 사실 (2026-07-31)

`attack/smoke_test.py`를 CPU 환경(torch 2.13+cpu)에서 실행해 확인했다.

**① 박스 좌표에 gradient가 흐른다 — 설계의 전제가 성립**

`torchvision.ops.roi_align`은 feature map뿐 아니라 **박스 좌표에 대해서도 미분 가능**하다.
이것이 성립하지 않았다면 기하 성분을 넣어도 박스를 움직일 수 없어
ByteTrack에 닿는 통로가 없었을 것이다.

**② 외형과 기하의 기여도는 `alpha=1.0`에서 이미 균형에 가깝다**

| | 값 |
|---|---|
| 외형 성분 `f_k` L2 norm | 1.0000 (L2 정규화되어 있으므로 항상 1) |
| 기하 성분 `alpha * g_k` L2 norm (alpha=1.0) | 0.7957 |

| | 요소 수 | gradient 합 | **요소당 평균** |
|---|---|---|---|
| feature | 144,000 | 108.29 | 0.000752 |
| 박스 좌표 | 20 | 0.0156 | 0.000778 |

요소당 gradient 크기가 거의 같다. 따라서 **`alpha`는 1.0 부근에서 탐색**하면 되고,
극단적인 값(수십~수백)을 쓸 이유는 현재로선 없다.

> ⚠️ 주의: gradient 합계끼리 비교하면 텐서 크기 차이 때문에 수천 배 차이로 보인다.
> 반드시 **요소당 평균**으로 비교할 것.

**③ 손실의 최소화 방향이 의도와 일치**

| 상황 | `L_decorr` | 해석 |
|---|---|---|
| 같은 ID의 서술자가 동일 | +1.000 | 공격 실패 상태 |
| 정반대 | -1.000 | 공격 성공 상태 |

| 상황 | `L_confuse` | 해석 |
|---|---|---|
| 교차쌍이 서로 닮음 | -1.000 | 공격 성공 상태 |
| 무작위 | -0.095 | 중립 |

두 손실 모두 **최소화하면 의도한 방향으로 간다**.

**④ SCGA 유틸 동작 재현**

- `feature_map_similarity_hinge`: 동일 입력 → 0 (threshold 이상이면 벌점 없음), 무관 입력 → 0.65
- `project`: eps=10/255에서 최대 편차 정확히 0.0392, `[0,1]` 범위 유지

---

## 6. 검증이 먼저 필요한 가설

> **"시간적 일관성이 TBD와 TBP 두 패러다임의 공통 표면이다"** — 아직 가설

**사전 검증 (공격 학습 불필요, 추론만):**
1. MOT17 클린 영상에서 각 트래커의 시간 일관성 신호를 추출
   - ByteTrack: 칼만 예측 박스와 매칭된 검출 박스의 IoU
   - FairMOT: 매칭된 트랙-검출 쌍의 ReID 코사인 유사도
   - MOTR: track query hidden state의 프레임 간 코사인 유사도
2. 세 신호 사이의 상관을 측정
3. 상관이 확인되면 진행, 아니면 방향 조정

---

## 7. 실행 계획

### 7.1 목표 학회와 마감

**ICLR 2027** — abstract 마감 **9/18**, paper 마감 **9/25** (기준일 2026-07-30, 약 8주 남음)

교수님의 목표는 **투고 경험**이며 accept 여부나 점수가 아니다.
따라서 SOTA를 노리기보다 **정직하고 완결된 논문을 기한 내 제출**하는 것이 우선이다.
결과가 부분적이거나 예상과 달라도 그대로 서술한다.

### 7.2 단계

| 단계 | 내용 | 검증 지표 | 8주 내 |
|---|---|---|---|
| 0 | 환경 구축, ByteTrack 클린 추론 재현 | MOT17 MOTA 재현 | O |
| 1 | 사전 검증 실험 (6절) | 신호 간 상관 | O |
| 2 | ByteTrack 단독 공격 (ReID 없음) | MOTA, IDF1, IDSW | O |
| 3 | FairMOT 추가 — **ReID 유무 전이 차이 (핵심)** | 두 트래커 동시 하락 | O |
| 4 | MOTR 계열 — 패러다임 간 전이 | HOTA, IDF1 | **X → future work** |
| 5 | 최신 TBD 트래커(OC-SORT, BoT-SORT) 추가 | 일반성 보강 | 여력 시 |

**3단계가 논지의 핵심이다.** BBA는 여기서 FairMOT -55% / ByteTrack -4%로 갈렸다.
**트래커 2개(ByteTrack + FairMOT)만으로도 핵심 주장은 성립한다.**

### 7.3 범위 축소 결정 (2026-07-30)

- **4단계(MOTR/TBP)는 이번 투고에서 제외**하고 future work로 명시한다.
  8주 안에 새 코드베이스(MOTR 계열)를 붙이는 것은 비현실적이며, 3단계까지만으로도
  "아키텍처 결박 vs 중립" 주장은 증명된다.
- 5단계는 3단계가 예정보다 빨리 끝날 때만 착수한다.

### 7.4 주차별 뼈대

| 주 | 목표 |
|---|---|
| 8/1 | GPU(RunPod) 확보·세팅, MOT17, ByteTrack 클린 재현 |
| 8/8 | 사전 검증 실험 + 공격 구현 착수 |
| 8/15 | ByteTrack 공격 1차 결과 / **논문 뼈대 작성 시작(병렬)** |
| 8/22 | 결과 개선 + FairMOT 연결 |
| 8/29 | **FairMOT 전이 실험 완료 (핵심 증거)** |
| 9/5 | 논문 본문 작성 |
| 9/12 | 마무리 + **9/18 abstract 등록** |
| 9/19 | 최종 점검 → **9/25 제출** |

**최대 리스크 2개**
1. **GPU 확보 지연** — 1주 밀리면 전체가 1주 밀린다. 8월 첫 주 안에 반드시 해결.
2. **논문 작성 시간 과소평가** — 실험이 끝난 뒤 쓰기 시작하면 늦는다. 8월 중순부터 병렬로 착수.

### 7.5 필수 ablation

리뷰어의 예상 질문을 미리 막기 위해 아래는 반드시 포함한다.

| # | 비교 | 막으려는 질문 | 우선도 |
|---|---|---|---|
| A1 | `L_attack` 만  vs  `L_attack + gen_sim` | **"SCGA 부분이 실제로 기여하나?"** | ⭐⭐ 필수 |
| A2 | 외형 성분만 / 기하 성분만 / 둘 다 (`alpha`) | "박스 기하가 정말 필요한가?" (ByteTrack 도달 근거) | ⭐⭐ 필수 |
| A3 | `L_decorr` 만 / `L_confuse` 만 / 둘 다 (`lambda`) | "두 손실 중 뭐가 실제로 일하나?" | ⭐ |
| A4 | 백본 층 선택 (얕음/중간/깊음) | "왜 하필 그 층인가?" | ⭐ |
| A5 | teacher 초기화: student 복사 vs 독립 랜덤 | 구현 디테일 (아래 주의 참고) | 여력 시 |
| **A6** | **서술자 중심화 on/off** (`--no_center`) | **"중심화가 실제로 필요한가?"** | ⭐⭐ 필수 |
| **A7** | **`--roi_source` pred vs gt** | "ROI를 어디서 잡아야 하는가?" | ⭐ |

**A6 근거 (이미 실측 확보, CHANGES_v2.md 5절):**
중심화하지 않으면 서로 다른 객체의 코사인이 이미 +0.80 이라 `L_confuse` 가 포화된다.
중심화하면 -0.06 으로 떨어져 판별 격차가 **0.19 -> 1.04 로 5배**가 되고,
generator gradient norm 도 6.62 -> 15.96 으로 2.4배가 된다.
외형·기하 두 성분 모두에서 같은 현상이 관찰됐다(공통 평균 성분 때문).

**A1이 가장 중요하다.** SCGA의 EMA 일관성이 전이성을 올린다는 것은
**이미지 분류의 cross-domain 전이**에서 검증된 것이지, **MOT의 cross-tracker 전이**에서는
아직 검증되지 않았다. 우리가 물려받은 가정이므로 반드시 확인해야 한다.

**A5 주의 — SCGA 원본 코드의 초기화 순서 문제**

`reference/scga_train_reference.py` 159~163행:

```python
teacher = copy.deepcopy(student)      # 여기선 동일
...
student.apply(weights_init_normal)    # student를 새 난수로 덮어씀
teacher.apply(weights_init_normal)    # teacher도 별개 난수로 덮어씀  <- 다시 갈라짐
```

`weights_init_normal`은 `nn.init.normal_`을 시드 고정 없이 쓰므로 두 번 호출하면 다른 값이 나온다.
표준 관행(Mean Teacher, BYOL)은 **teacher = student 복사본**으로 시작하는 것이다.
우리 구현에서는 **초기화를 먼저 하고 그 뒤에 deepcopy** 하는 순서로 바로잡되,
원본과의 차이를 ablation으로 남겨둔다.

---

## 8. 비교 대상 처리

| 논문 | 코드 | 처리 방침 |
|---|---|---|
| **BBA** (Neural Networks 2024) | 미공개 | 재현하지 않고 원 논문 수치 + FADE의 독립 평가 수치를 인용 |
| **TraSA** (ASOC 2024) | 공개 (JHL-HUST/TraSA, DerryHub/ByteTrack-attack) | 재현 비교 |
| **BankTweak** (IJCAI 2025) | 확인 필요 (arXiv 2408.12727) | ⚠️ 아래 참고 |
| **FADE** (CVPR 2026) | 확인 필요 | MOTR 계열이라 이번 범위 밖 (future work) |

**BankTweak 비교의 특수성**

BankTweak은 ReID feature bank를 직접 조작하므로 **ByteTrack에는 적용 자체가 불가능**하다.
따라서 "같은 조건에서 성능 비교"가 성립하지 않는다. 논문에서는 성능 표에 나란히 놓기보다,
**"적용 가능 범위"라는 축에서 비교**하는 것이 정확하다.

| | 적용 가능 트래커 |
|---|---|
| BankTweak | ReID feature bank 보유 (DeepSORT, StrongSORT, MOTDT) |
| BBA | ReID 헤드 보유 (CenterNet 계열, FairMOT) |
| 제안 | **백본을 가진 모든 검출기 기반 트래커** |

---

## 참고

- 저장소 구조: `yolox/`, `tools/` = ByteTrack 원본 / `generator/` = SCGA / `attack/` = 본 연구 / `reference/` = SCGA 학습 스크립트 원본
- 공격 참고 구현: `~/ssu/ByteTrack-attack` (TraSw의 ByteTrack 공격)
- SCGA 원본: `~/ssu/scga`
