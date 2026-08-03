# 1차 구현 정리

> 저장소를 구성하고 뼈대 코드를 작성한 결과 정리
> 기준일 2026-07-31 · 작성 김영재

이 문서는 **무엇을, 어떻게, 왜** 만들었고, 각 코드가 **직접 작성한 것인지 SCGA에서 가져온 것인지**를
구분해 기록한다. 나중에 논문의 구현 절을 쓰거나, 코드 출처를 밝힐 때 근거가 된다.

---

## 1. 저장소 구성 방식

### 왜 ByteTrack을 뼈대로 삼았는가

두 코드베이스를 합쳐야 했다.

| | 제공하는 것 | 분량 |
|---|---|---|
| **ByteTrack** | 검출기(YOLOX), 트래커, MOT 데이터 파이프라인, 평가 | 파이썬 101개 파일 |
| **SCGA** | 섭동 생성기(generator), EMA 일관성 기법 | generator 4개 파일 |

SCGA를 뼈대로 삼으면 트래커·데이터·평가를 전부 새로 만들어야 한다.
반대로 ByteTrack을 뼈대로 삼으면 **SCGA에서 가져올 것은 generator뿐**이다.
따라서 ByteTrack을 기준으로 잡았다.

### 구성 절차

```
git clone --depth 1 https://github.com/FoundationVision/ByteTrack.git mot-scga-attack
cd mot-scga-attack
rm -rf .git && git init          # ByteTrack 히스토리 제거, 새 저장소로 시작
cp -r ~/ssu/scga/generator ./generator
mkdir attack docs reference
cp ~/ssu/scga/train_BIA_ours.py reference/scga_train_reference.py
mv README.md README_ByteTrack.md
```

**ByteTrack-attack(TraSw 구현)은 가져오지 않았다.** `~/ssu/ByteTrack-attack`에 별도로 두고
분석용으로만 참고했다. 코드는 한 줄도 복사하지 않았다.

### 최종 구조

```
mot-scga-attack/
├── yolox/ tools/ exps/ datasets/ deploy/ ...   ByteTrack 원본 (무수정)
├── generator/                                  SCGA 원본 (무수정)
├── reference/scga_train_reference.py           SCGA 원본 (참고용)
├── attack/                                     ★ 직접 작성
│   ├── dataset.py
│   ├── descriptor.py
│   ├── losses.py
│   └── train_generator.py
├── docs/DESIGN.md, CHANGES_v1.md               ★ 직접 작성
└── README.md                                   ★ 직접 작성
```

---

## 2. 파일별 상세

### 2.1 `attack/dataset.py` (203줄) — **100% 직접 작성**

**목적:** MOT17에서 연속 두 프레임 + 박스 + ID를 뽑아온다.

**왜 새로 만들었나:** SCGA의 데이터로더는 ImageNet 분류용(이미지 한 장 + 클래스 라벨)이라
프레임 쌍과 track ID 개념이 없다. ByteTrack의 `MOTDataset`은 COCO 형식 단일 프레임용이라
역시 맞지 않는다.

**주요 결정과 근거**

| 결정 | 내용 | 근거 |
|---|---|---|
| gt 필터링 | `conf<1 or class!=1 or visibility<0.3` 제외 | 무시 대상 박스, 비보행자, 심하게 가려진 객체는 외형 feature가 신뢰할 수 없다 |
| 전처리 | ByteTrack `preproc`과 동일하게 letterbox(114 패딩, 좌상단 정렬) | 학습·평가 조건을 일치시키기 위함 |
| **`[0,1]`까지만 반환** | mean/std 정규화는 학습 루프에서 수행 | generator와 eps 투영이 `[0,1]` 공간에서 동작해야 한다. SCGA와 동일한 구조 |
| 프레임 쌍 필터 | 공통 ID가 2개 이상인 쌍만 사용 | 1개면 `L_confuse`(다른 객체 간 혼동)를 계산할 수 없다 |
| `frame_gaps=(1,2,3)` | 간격을 섞음 | 간격이 클수록 난이도가 오르고 데이터 다양성이 늘어난다. BBA도 간격 2, 1, -1을 섞어 썼다 |
| `detector='FRCNN'` | 검출기 변형 1벌만 사용 | MOT17은 같은 영상이 DPM/FRCNN/SDP 3벌인데 이미지와 GT가 동일하다. 중복 학습 방지 |
| `half='first'/'second'` | 시퀀스를 반으로 분할 | MOT17 test에는 라벨이 없어 자체 평가가 불가능하다. ByteTrack 관례를 따름 |

**`pair_collate`를 따로 만든 이유**

프레임마다 사람 수가 달라 기본 collate로는 스택할 수 없다. 단순히 리스트로 두는 대신
`(batch_idx, x1, y1, x2, y2)` 형태 `[K, 5]`로 만들었는데, 이는 **`torchvision.ops.roi_align`이
요구하는 정확한 형식**이라 `descriptor.py`에 그대로 넣을 수 있다.

**ByteTrack 전처리 확인 경로**

`yolox/data/data_augment.py:189` `preproc()`, `exps/example/mot/yolox_x_mix_det.py:102`
에서 `ValTransform(rgb_means=(0.485,0.456,0.406), std=(0.229,0.224,0.225))` 확인.

---

### 2.2 `attack/descriptor.py` (71줄) — **100% 직접 작성**

**목적:** 백본 feature와 박스로부터 트래커 구조에 의존하지 않는 객체 서술자를 만든다.

```
d_k = [ L2norm(GAP(ROIAlign(F, b_k))) ;  alpha * (cx, cy, w, h) ]
```

**왜 기하 성분을 넣었나 (설계의 핵심)**

`yolox/tracker/matching.py`를 읽고 확인한 사실:

```python
def iou_distance(atracks, btracks):
    cost_matrix = 1 - ious(예측박스, 검출박스)      # 재료가 박스 좌표뿐
def fuse_score(cost_matrix, detections):
    fuse_sim = (1 - cost_matrix) * det_scores        # 최종 = 1 - IoU x score
```

| 매칭 단계 | 비용 | 임계값 | 통과 조건 |
|---|---|---|---|
| 1차 | `1 - IoU x score` | 0.8 | IoU x score >= 0.2 |
| 2차 | `1 - IoU` (**점수 무시**) | 0.5 | IoU >= 0.5 |

**ByteTrack에는 ReID가 없고, 2차 매칭은 점수를 아예 보지 않는다.**
따라서 외형 feature만으로는 ByteTrack에 닿을 수 없고, **박스 위치를 흔들어야 한다.**
기하 성분은 선택이 아니라 필수다.

(이는 BBA가 ByteTrack에서 MOTA를 4%만 떨어뜨린 이유이기도 하다.)

---

### 2.3 `attack/losses.py` (105줄) — **혼합**

| 구간 | 함수 | 출처 |
|---|---|---|
| 상단 ~60줄 | `_pair_by_id`, `temporal_decorrelation_loss`, `cross_object_confusion_loss`, `attack_loss` | ★ **직접 작성** |
| 하단 ~45줄 | `feature_map_similarity_hinge`, `generator_consistency_loss`, `update_ema`, `project` | **SCGA에서 가져옴** |

**직접 작성한 부분**

```
L_decorr  =   mean_k  cos( d_t[k], d_{t+1}[k] )          # 같은 ID를 시간축에서 분리
L_confuse = - mean_{k!=j} cos( d_t[k], d_{t+1}[j] )      # 다른 ID를 서로 근접
```

**SCGA에서 가져온 부분과 그 변형**

| 함수 | SCGA 원본 위치 | 변형 |
|---|---|---|
| `feature_map_similarity_hinge` | `train_BIA_ours.py:80-105` | 로직 동일, 중간 변수만 축약 |
| `update_ema` | `train_BIA_ours.py:108-121` | `@torch.no_grad()` 데코레이터 추가 |
| `project` | 학습 루프 227-230행에 인라인 | 함수로 분리 |
| `generator_consistency_loss` | 학습 루프 232-240행 `taus` 반복문 | 함수로 묶음 (⚠️ 아래 참고) |

⚠️ **`generator_consistency_loss`는 원본과 다를 수 있다.**
SCGA 원본은 블록 개수를 `len(feats_teacher[1:]) - 4`로 계산하는데,
이 값은 generator가 실제로 반환하는 feature 개수에 의존한다.
우리 구현은 `n_blocks=6` 인자로 일반화했으므로, **실제 feature 개수를 확인해 맞춰야 한다.**

---

### 2.4 `attack/train_generator.py` (약 320줄) — **혼합 + 빈칸**

| 구간 | 출처 |
|---|---|
| 인자 파서 | SCGA `utee/parser.py` 인자명 유지 + 본 연구용 추가 |
| 출력 폴더 생성, 스크립트 자기 복사 | SCGA `train_BIA_ours.py:34-48` |
| generator 구성, 옵티마이저 | SCGA `train_BIA_ours.py:151-164` |
| **학습 루프 전체** | SCGA `train_BIA_ours.py:203-279` |
| 체크포인트 저장 | SCGA (student가 아니라 **teacher**를 저장) |
| 데이터로더 연결, 우리 인자 | ★ 직접 작성 |
| `build_detector`, `forward_detector`, `preds_to_xyxy_scores` | ★ 직접 작성 (YOLOX 구조 확인 후) |
| `assign_ids`, `frame_descriptor` | ⬜ **미작성 (TODO 3, 4)** |

---

## 3. SCGA 원본에서 의도적으로 바꾼 것

### 3.1 teacher 초기화 순서 — **버그 교정**

**원본** (`reference/scga_train_reference.py:158-163`)

```python
student = torch.compile(student)
teacher = copy.deepcopy(student)      # 여기선 동일
student.apply(weights_init_normal)    # student를 새 난수로 덮어씀
teacher.apply(weights_init_normal)    # teacher도 별개 난수로 덮어씀  <- 다시 갈라짐
```

`weights_init_normal`은 `nn.init.normal_`을 시드 고정 없이 사용하므로 두 번 호출하면
서로 다른 값이 나온다. 즉 **teacher와 student가 완전히 다른 난수로 시작한다.**

표준 관행(Mean Teacher, BYOL)은 **teacher = student의 복사본**으로 시작하는 것이다.
momentum이 0.999라 teacher가 student를 따라잡는 데 수천 스텝이 걸리므로,
초반 일관성 손실이 의미 없는 신호에 맞추라는 압력이 될 수 있다.

**우리 구현**

```python
student.apply(weights_init_normal)    # 먼저 초기화
teacher = copy.deepcopy(student)      # 그 다음 복사 -> 완전히 동일
```

원본과의 차이는 ablation A5로 남긴다 (DESIGN.md 7.5절).

### 3.2 `normalize()`를 in-place가 아니게 변경

**원본** (`reference/scga_train_reference.py:124-148`)

```python
def normalize(t, mean=None, std=None):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]   # 원본 텐서를 덮어씀
    ...
    return t
```

이 때문에 호출부에서 반드시 `.clone()`을 붙여야 했다(`model(normalize(img.clone()))`).
빠뜨리면 원본 이미지가 오염된다.

**우리 구현** (`dataset.py :: normalize_for_detector`)

```python
return (x - m) / s      # 새 텐서 반환
```

`.clone()` 누락으로 인한 버그 가능성을 제거했다.

### 3.3 `torch.compile` 제거

원본은 `student = torch.compile(student)` 후에 `deepcopy`와 `apply`를 수행한다.
compile로 감싼 객체에 대한 `deepcopy`/`apply` 동작이 미묘해질 수 있어 우선 제외했다.
속도가 문제되면 나중에 다시 넣는다.

### 3.4 인자 교체

`--train_data`, `--surrogate`, `--RN`, `--DA`, ViT 관련 인자 등 분류 태스크 전용 인자를 제거하고
`--data_root`, `--detector_variant`, `--input_h/w`, `--frame_gaps`, `--alpha`, `--lam`,
`--fpn_level`, `--exp_file`, `--ckpt`를 추가했다.
`--eps`, `--gap`, `--nat`, `--save_freq` 등 SCGA 인자명은 그대로 유지해 대조가 쉽도록 했다.

---

## 4. YOLOX 구조 확인 결과 (TODO 1, 2의 근거)

| 확인 항목 | 결과 | 위치 |
|---|---|---|
| 모델 구조 | `backbone(YOLOPAFPN)` → `head(YOLOXHead)`로 분리 | `yolox/models/yolox.py:28-48` |
| 백본 반환 | `(pan_out2, pan_out1, pan_out0)` = stride 8, 16, 32 | `yolox/models/yolo_pafpn.py:116` |
| head eval 반환 | `[B, N, 5+num_classes]`, decode 후 절대 픽셀 cxcywh | `yolox/models/yolo_head.py:207-214` |
| decode | `(x+grid)*stride`, `exp(wh)*stride` | `yolo_head.py:238-253` |
| MOT 클래스 수 | `num_classes = 1` (보행자만) → 채널 `[cx,cy,w,h,obj,cls]` | `exps/example/mot/yolox_x_mix_det.py:14` |
| 모델 로드 | `get_exp()` → `get_model()` → `eval()` → `load_state_dict(ckpt['model'])` | `tools/track.py:163-190` |
| 입력 크기 | `(800, 1440)`, 모델은 YOLOX-X(depth 1.33, width 1.25) | `yolox_x_mix_det.py:15-21` |

**중요한 확인:** 우리가 만든 `preds_to_xyxy_scores`의 점수 계산

```python
scores = preds[..., 4] * preds[..., 5]      # obj x cls
```

는 ByteTrack이 실제로 쓰는 계산과 동일하다 (`byte_tracker.py:1908`).
즉 **우리가 공격하는 점수가 트래커가 보는 바로 그 점수**다.

**주의:** `decode_outputs`가 in-place 연산을 사용한다.
autograd가 처리하지만 문제가 생기면 `.clone()`으로 우회한다.
ByteTrack-attack이 같은 경로로 `imgs.grad`를 얻는 데 성공했으므로 가능성은 낮다.

---

## 5. 검증 상태

CPU 전용 가상환경(`~/.venv`, torch 2.13.0+cpu, torchvision 0.28.0+cpu)을 만들어
GPU 없이 검증 가능한 부분을 먼저 확인했다.

| 항목 | 상태 |
|---|---|
| 문법 검사 (`py_compile`) | 통과 |
| **CPU 스모크 테스트** (`attack/smoke_test.py`) | **통과** |
| ├ 서술자 형태 | 통과 — `feat(2,64,50,90)` + `boxes(7,5)` → `d(7,68)` |
| ├ **박스 좌표 gradient 도달** | **통과** — 설계의 전제 확인 |
| ├ 손실 최소화 방향 | 통과 — `L_decorr`, `L_confuse` 모두 의도대로 |
| └ SCGA 유틸 (hinge, project) | 통과 |
| **`dataset.py` 실물 실행** | **통과** — 아래 5.2절 |
| **박스 좌표 정렬 시각 검증** | **통과** — `attack/viz_dataset.py` |
| 검출기 연동 (`build_detector`, `forward_detector`) | **미실행** — 가중치 미보유 |
| VRAM 사용량 | 미측정 |

**환경**

```bash
source ~/.venv/bin/activate      # torch 2.13.0+cpu, torchvision 0.28.0+cpu, opencv 5.0.0
cd ~/ssu/mot-scga-attack
PYTHONPATH=. python attack/smoke_test.py
PYTHONPATH=. python attack/viz_dataset.py --n 3
```

**아직 필요한 패키지:** `pycocotools`, `cython_bbox` (ByteTrack 트래커 실행용)

### 5.2 MOT17 실물 검증 (2026-08-02)

**데이터셋 확보**

Kaggle `wenhoujinjust/mot-17`을 `kagglehub`으로 받았고, 구조가 공식 MOT17과 일치함을 확인했다.

```
MOT17/train/MOT17-02-FRCNN/
├── det/  gt/gt.txt  img1/(600장)  seqinfo.ini(seqLength=600)
gt.txt 예시: 1,1,17,150,77,191,1,1,1      # 9개 컬럼 정상
```

용량 5.7GB. 저장소 밖(`~/.cache/kagglehub/.../MOT17`)에 두고 심볼릭 링크만 걸었다.

```bash
ln -sfn ~/.cache/kagglehub/datasets/wenhoujinjust/mot-17/versions/1/MOT17 \
        ~/ssu/mot-scga-attack/datasets/mot
```

`.gitignore`에 `datasets/mot`, `experiments/`, `.venv/`를 추가해 5.7GB가 커밋되는 사고를 막았다.

**`dataset.py` 실행 결과**

| 항목 | 값 |
|---|---|
| 인덱싱 시간 | 0.4초 |
| **학습 쌍 개수** | **7,929** (train 전반부, FRCNN, gap 1~3, 공통 ID 2개 이상) |
| 이미지 | `[3, 800, 1440]`, float32, 범위 `[0.000, 1.000]` |
| 박스 예시 | `[991.1, 309.6, 1114.8, 590.4]` — 입력 해상도 범위 내 |
| collate 출력 | `boxes [30, 5]`, `ids [30]` — roi_align 형식 정상 |

**시각 검증 (`attack/viz_dataset.py`)**

두 프레임을 위아래로 붙이고 ID별 고정 색으로 박스를 그려 확인했다.

- 박스가 사람 위에 정확히 놓임
- letterbox 패딩(회색)이 **오른쪽에만** 생김 → 좌상단 정렬이 의도대로 동작
  (MOT17-02는 1920x1080, `r = min(800/1080, 1440/1920) = 0.74` → 세로는 꽉 차고 가로에 여백)
- 같은 색 박스가 두 프레임에서 같은 사람을 가리킴 → ID 대응 정상
- 프레임 간 사람의 이동이 눈에 보임 → 프레임 쌍 추출 정상

결과물: `output/viz/pair_*.jpg`

### 5.1 스모크 테스트에서 얻은 설계 정보

상세는 `DESIGN.md` 5.1절에 기록했다. 요약하면

- `roi_align`은 **박스 좌표에 대해서도 미분 가능**하다. 기하 성분이 실제로 박스를 움직일 수 있다.
- `alpha=1.0`에서 외형(norm 1.0)과 기하(norm 0.80)의 기여가 이미 균형에 가깝다.
  요소당 gradient 크기도 거의 동일(0.000752 vs 0.000778)하므로 극단적인 `alpha`는 불필요하다.
- ⚠️ gradient를 **합계로 비교하면 안 된다.** 텐서 크기 차이(144,000 vs 20) 때문에
  수천 배 차이로 오해하기 쉽다. 반드시 요소당 평균으로 볼 것.

---

## 6. 남은 작업

| # | 항목 | 비고 |
|---|---|---|
| TODO 3 | `assign_ids()` — 예측 박스와 GT를 IoU 매칭해 ID 부여 | **설계 결정 A** 확정 필요 |
| TODO 4 | `frame_descriptor()` — 위 조각들을 연결 | TODO 3 이후 |
| — | `generator_consistency_loss`의 블록 개수를 원본과 대조 | 실제 feature 개수 확인 필요 |
| — | 데이터로더 시각 검증 스크립트 | GPU 없이도 작성 가능 |
| — | 평가 스크립트 (공격 적용 후 트래커 실행 → TrackEval) | |

**설계 결정 A (DESIGN.md 참조)**

기하 성분에 gradient가 흐르려면 예측 박스를 써야 하지만, 공격이 성공할수록 검출이 사라져
gradient가 소실될 위험이 있다. 현재 후보는

> ROI 영역 = GT 박스(안정) + 기하 성분 = 매칭된 예측 박스(gradient 확보)

이며 아직 확정하지 않았다.
