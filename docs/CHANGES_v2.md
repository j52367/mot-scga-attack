# 2차 구현 정리

> 빈칸(TODO 3, 4)을 채우고 end-to-end 경로를 실제로 검증한 결과
> 기준일 2026-08-02 · 작성 김영재
> 1차 정리는 [CHANGES_v1.md](CHANGES_v1.md) 참조

---

## 0. 이번 라운드 요약

| # | 한 일 | 결과 |
|---|---|---|
| 1 | `assign_ids`, `frame_descriptor` 구현 | 완료, 단위 테스트 통과 |
| 2 | 검출기 가중치(YOLOX-S) 확보 | 완료 |
| 3 | **end-to-end 1스텝 실행** | **성공** — generator까지 gradient 도달 |
| 4 | ⚠️ **SCGA 공개 코드의 불일치 발견** | 래퍼로 우회 (원본 무수정) |
| 5 | ⚠️ **`L_confuse` 포화 문제 발견** | 중심화로 해결, 판별 격차 5배 개선 |
| 6 | **평가 파이프라인 작성** | 클린 기준값 확보 (MOTA 55.1 / IDF1 53.1) |
| 7 | **랜덤 노이즈 baseline 측정** | 하한선 확보 — 랜덤으로는 IDF1 변화 없음 |

4번과 5번은 **돌려보지 않았으면 발견할 수 없는 문제**였다.

이로써 학습과 평가 양쪽 경로가 모두 검증됐다.

```
학습:  generator -> 노이즈 -> 검출기 -> 서술자 -> 손실 -> generator   (6절)
평가:  generator -> 노이즈 -> 검출기 -> ByteTrack -> MOTA/IDF1/IDSW   (7절)
```

---

## 1. `assign_ids()` — 예측 박스에 GT ID 붙이기

### 설계 결정 A가 해소된 경위

1차 정리에서 미결로 남긴 문제는 다음과 같았다.

> 기하 성분에 gradient가 흐르려면 예측 박스를 써야 하는데,
> 공격이 성공하면 검출이 사라져 gradient가 소실될 수 있다.

**YOLOX head 출력을 확인하면서 이 걱정이 불필요함을 알았다.**

head는 신뢰도와 무관하게 **모든 그리드 셀에 대해 박스를 출력**한다.
800x1440 기준 `N = (100x180) + (50x90) + (25x45) ≈ 23,600`개다.
박스가 사라지는 것은 `postprocess`(NMS + 임계값) **이후**이지 head 출력 단계가 아니다.

따라서 **점수 필터 없이 매칭**하면 공격이 아무리 성공해도 매칭 대상이 남는다.
`iou_thr`을 0.1로 낮게 둔 것도 같은 이유다 — 검출 품질 판정이 아니라 매칭 유지가 목적이다.

결과적으로 GT/예측을 섞는 절충안 없이 **예측 박스만으로 안전하게 처리**할 수 있게 됐다.

### 구현 핵심

```python
with torch.no_grad():
    iou = box_iou(g[:, 1:], pred_boxes[b].detach())   # 매칭은 detach
    best_iou, best_idx = iou.max(dim=1)
    keep = best_iou >= iou_thr

picked = pred_boxes[b][best_idx[keep]]                # 원본 인덱싱 -> gradient 유지
```

**인덱스만 빌려오고 값은 미분 가능한 원본에서 가져온다.**
ByteTrack-attack이 칼만 예측 좌표를 상수로만 받아온 것과 같은 패턴이다.

**알려진 한계:** 서로 다른 GT가 같은 예측에 매칭될 수 있다(중복).
예측이 23,600개이고 사람들의 위치가 서로 달라 실제로는 드물지만,
문제가 되면 IoU 내림차순 그리디 유일 매칭으로 바꾼다.

### 단위 테스트 (`attack/smoke_test.py` [6], [7])

| 확인 | 결과 |
|---|---|
| 의도한 예측(3번, 41번)이 선택되는가 | 통과 |
| 매칭된 ID가 `[7, 9]`인가 | 통과 |
| **선택된 예측에만 gradient가 도달하는가** | 통과 — 미선택 예측은 `grad.sum() == 0` |
| 겹치는 예측이 없을 때 | 빈 결과 반환, 에러 없음 |
| GT가 없을 때 | 빈 결과 반환 |
| 배치 인덱스 유지 | `[0.0, 1.0]` 정상 |

---

## 2. `frame_descriptor()` — 조각 연결

```python
x_norm = normalize_for_detector(x01)
feat, preds = forward_detector(model, x_norm, args.fpn_level)
pred_boxes, _ = preds_to_xyxy_scores(preds)
m_pred, m_gt, m_ids = assign_ids(pred_boxes, boxes_gt, ids_gt, args.iou_thr)
d = object_descriptor(feat, m_pred if roi_source=='pred' else m_gt, ...)
```

`--roi_source` 를 플래그로 둔 이유

| 값 | ROI 영역 | 성격 |
|---|---|---|
| `pred` (기본) | 예측 박스 | 트래커가 실제로 보는 영역과 일치 |
| `gt` | GT 박스 | "그 사람이 실제로 있는 곳"의 외형 |

둘 다 정당해 ablation 항목으로 남긴다.
**어느 쪽이든 기하 성분은 항상 예측 박스에서 오므로 gradient는 유지된다.**

---

## 3. 검출기 가중치

ByteTrack 모델 zoo에서 **YOLOX-S**(`bytetrack_s_mot17`, 69MB)를 받았다.
CPU 검증이 목적이라 가장 큰 YOLOX-X(99M 파라미터) 대신 소형 모델을 택했다.

```bash
gdown 1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj -O pretrained/bytetrack_s_mot17.pth.tar
```

추가 설치 패키지: `loguru`, `thop`, `pycocotools`, `gdown`

---

## 4. ⚠️ SCGA 공개 코드의 불일치와 래퍼

### 문제

학습 스크립트(`train_BIA_ours.py:232-240`)는 generator가 **블록별 feature 리스트**를
반환한다고 가정한다.

```python
taus = [0.6 + 0.05 * i for i in range(6)]   # 주석: "thresholds for 6 blocks"
for j in range(len(feats_teacher[1:]) - 4):
    hinge(feats_teacher[1:][j], feats_student[1:][j], threshold=taus[j])
```

그런데 공개된 generator는 **어느 것도 리스트를 반환하지 않는다.**

| 파일 | `feat` 인자 | 반환 |
|---|---|---|
| `resnet.py` `GeneratorResnet` | 있음 | 텐서 하나 (`res_blocks` 출력) |
| `resnet_gap.py` `ResnetGenerator` | 없음 | 텐서 하나 |
| `resnet_nat.py` `StableGeneratorResnet` | 없음 | 텐서 하나 |
| `resnet_gap.py` `forward_features()` | — | 텐서 하나 + **`.detach()`** (gradient 끊김) |

`feats`가 `[B,C,H,W]` 텐서이면 `[1:]`은 **배치 차원**을 자르게 되어 의도와 다르게 동작한다.
우리 테스트(배치 1)에서는 `len(feats)=1` → `[1:]`이 비어 → 손실이 `0.0`(float)이 되면서
`.item()` 호출에서 에러가 났고, 그 덕에 문제를 발견했다.

**SCGA의 핵심 기여가 generator 내부 일관성인데, 공개 코드로는 그것이 계산되지 않는다.**

### 해결 — `attack/generator_wrap.py` (원본 무수정)

`generator/`를 고치지 않고 **forward hook으로 블록 출력을 모으는 래퍼**를 만들었다.

```python
class GeneratorWithFeatures(nn.Module):
    def forward(self, x, feat=False):
        handles = [getattr(self.netG, n).register_forward_hook(...) for n in self.block_names]
        try:    out = self.netG(x)
        finally: [h.remove() for h in handles]
        return out, feats
```

설계상 주의한 점

- **hook은 호출할 때마다 등록하고 즉시 제거**한다. 영속 hook을 두면 `deepcopy`(teacher 생성)나
  `apply`(초기화)에서 문제가 생길 수 있다.
- 모듈 객체가 아니라 **속성 이름**을 들고 있는다. 그래야 `deepcopy` 후에도 올바른 사본을 가리킨다.
- hook에서 `.detach()`를 하지 않는다. gradient가 흘러야 한다.

블록은 `GeneratorResnet` 기준 6개를 잡았고, 실행 결과 정확히 6개가 나왔다.

```
(64,320,576) (128,160,288) (256,80,144) (256,80,144) (128,160,288) (64,320,576)
 block1       block2        block3       res_blocks   upsample1     upsample2
```

`taus`의 6개와 1:1로 대응한다.

### 원본과의 해석 차이 (논문에 밝힐 것)

SCGA 원본의 `for j in range(len(feats[1:]) - 4)`는 리스트 길이에 의존하는데,
공개 구현에서는 그 리스트가 존재하지 않는다.
주석에 "thresholds for 6 blocks"라고 적혀 있고 `taus`가 6개이므로,
**블록 6개를 `taus[0..5]`와 1:1 대응**시키는 것으로 해석했다.
(`attack/losses.py :: generator_consistency_loss` 주석에 기록)

---

## 5. ⚠️ `L_confuse` 포화와 중심화

### 발견

end-to-end 첫 실행에서 손실 값이 이상했다.

```
decorr  = +0.9931     같은 ID가 프레임 간 거의 동일 (예상대로, 공격 여지 큼)
confuse = -0.8156     다른 객체끼리 이미 코사인 0.82로 유사   <- 문제
```

`L_confuse`의 최적값이 `-1.0`인데 시작이 `-0.82`면 **개선 여지가 0.18뿐**이라
학습을 거의 이끌지 못한다.

### 진단

원인이 외형인지 기하인지 확인하기 위해 성분별로 분리해 측정했다(프레임 쌍 3개).

| 구성 | 같은 ID | 다른 ID | 판별 격차 |
|---|---|---|---|
| 전체 `[외형; 기하]` | +0.997 | **+0.804** | 0.19 |
| 외형만 | +0.995 | +0.735 | 0.26 |
| 기하만 | +1.000 | +0.957 | 0.04 |
| **전체 (중심화)** | +0.980 | **-0.060** | **1.04** |
| 외형만 (중심화) | +0.978 | -0.061 | 1.04 |
| 기하만 (중심화) | +1.000 | -0.038 | 1.04 |

**두 성분 모두 공통 평균 성분 때문에 유사도가 부풀려져 있었다.**

- 외형: 사람 박스에서 뽑은 feature라 전부 "사람다움"을 공유
- 기하: `(cx,cy,w,h)/(W,H,W,H)`가 **전부 양수**라 구조적으로 코사인이 높다

세 샘플 모두 같은 패턴이었다.

### 해결 — `center_pair()`

두 프레임의 서술자를 **합친 평균으로 함께 중심화**한 뒤 코사인을 계산한다.

```python
def center_pair(d_t, d_t1):
    m = torch.cat([d_t, d_t1], dim=0).mean(dim=0, keepdim=True)
    return d_t - m, d_t1 - m
```

프레임마다 따로 중심화하지 않는 이유: 좌표계가 어긋나 프레임 간 비교가 왜곡된다.

`--no_center` 플래그로 끌 수 있게 해 ablation 대상으로 남겼다(기본은 켜짐).

### 효과

| | 중심화 전 | 중심화 후 |
|---|---|---|
| `L_decorr` | +0.9931 | +0.9739 (여전히 높음 — 공격 여지 유지) |
| `L_confuse` | -0.8156 (포화) | **+0.0553** (-1까지 여지 확보) |
| generator gradient norm | 6.62 | **15.96** (2.4배) |

`L_decorr`는 거의 그대로 유지되면서 `L_confuse`만 정상 범위로 돌아왔고,
gradient 신호도 2.4배 강해졌다.

> 이 문제는 CKA가 커널을 중심화하는 것과 같은 이유에서 발생한다.
> 공통 성분을 제거해야 상대적 차이를 볼 수 있다.

---

## 6. end-to-end 검증 결과

`attack/e2e_test.py` — 실제 MOT17 프레임 쌍 + 실제 검출기 가중치로 학습 한 스텝.

```bash
source ~/.venv/bin/activate
cd ~/ssu/mot-scga-attack
PYTHONPATH=. python attack/e2e_test.py
```

| 단계 | 결과 |
|---|---|
| 검출기 로드 | 1.1초, `num_classes=1` |
| generator 파라미터 | 7.8M |
| 블록 feature | **6개** (taus와 대응) |
| eps 투영 | 최대 편차 0.0392 = `10/255` 정확 |
| 서술자 | `d_t(15, 260)` = 256(FPN stride16) + 4(기하) |
| 매칭 | 15/16개, 공통 ID 15개 |
| 손실 | attack `+1.0293`, gen_sim `0.0000` |
| **generator gradient norm** | **15.96** |
| **gradient 받은 파라미터** | **53/53** |
| step 후 파라미터 변화 | 1.88 (실제로 학습됨) |

**`generator -> 검출기 -> 서술자 -> 손실 -> generator` 경로가 끊김 없이 이어진다.**

### `gen_sim = 0.0000`은 정상이다

우리가 1차에서 교정한 초기화 순서 때문이다.

```python
student.apply(weights_init_normal)
teacher = copy.deepcopy(student)      # teacher == student
```

두 feature가 동일 -> 코사인 1.0 -> `relu(0.85 - 1.0) = 0`.
학습이 진행돼 student가 teacher에서 벌어질 때만 벌점이 생긴다.

> 반대로 SCGA 원본(teacher와 student를 각각 랜덤 초기화)이었다면
> step 0부터 `gen_sim > 0`이 되어 student를 **무의미한 랜덤 teacher 쪽으로 끌어당겼을 것**이다.
> 1차에서 지적한 문제가 실측으로 확인된 셈이다.

### CPU 실행 시간 (320x576, YOLOX-S, 배치 1)

| 단계 | 시간 |
|---|---|
| generator forward | 24.3초 |
| 검출기 forward + 서술자 | 3.0~5.6초 |
| backward | 27.4초 |

GPU + 원본 해상도(800x1440)에서는 크게 빨라질 것으로 예상한다.

---

## 7. 평가 파이프라인 (`attack/evaluate.py`)

학습 경로는 검증됐지만 **공격이 잘 됐는지 잴 방법이 없었다.**
GPU가 없어도 작성·검증이 가능하고, 이것이 없으면 GPU가 와도 결과를 낼 수 없으므로 먼저 만들었다.

### 7.1 구조

```
프레임마다:
  letterbox -> [0,1]
    | (--attack_ckpt 있으면) generator 노이즈 + eps 투영
  normalize_for_detector -> 검출기 -> postprocess(NMS)
    |
  BYTETracker.update(out, 원본크기, 입력크기)
    |
  MOT 형식으로 기록
-> Evaluator.eval_file -> MOTA / IDF1 / IDSW
```

**클린과 공격을 같은 스크립트로 처리**한다. `--attack_ckpt` 유무로만 갈리므로
전처리·검출·추적·평가 조건이 정확히 일치한다.

기본 `--half second` — 학습에 `first` 를 썼으므로 평가 구간이 겹치지 않는다.

### 7.2 재사용한 ByteTrack 구성요소

| 대상 | 위치 | 확인한 규약 |
|---|---|---|
| `BYTETracker.update(output_results, img_info, img_size)` | `yolox/tracker/byte_tracker.py:159` | `img_info` = 원본 (h,w), `img_size` = 입력 (h,w). 내부에서 `scale = min(img_size/img_info)` 로 나눠 원본 좌표로 복원 |
| `postprocess(pred, num_classes, conf_thre, nms_thre)` | `yolox/utils/boxes.py:33` | 출력 `[N,7]` = (x1,y1,x2,y2,obj,cls,cls_pred) |
| `Evaluator(data_root, seq, 'mot')` + `eval_file` + `get_summary` | `yolox/tracking_utils/evaluation.py` | motmetrics 기반 |

트래커 파라미터는 `tools/track.py` 기본값을 그대로 썼다
(`track_thresh=0.6`, `track_buffer=30`, `match_thresh=0.9`).
박스 필터도 동일하다 (`min_box_area=100`, 종횡비 `w/h > 1.6` 제외).

### 7.3 `attack/compat.py` — numpy 호환 계층

ByteTrack(2021)과 motmetrics 1.4.0 이 numpy 2.x 에서 제거된 API 를 쓴다.

| API | 사용처 | 제거 시점 |
|---|---|---|
| `np.float`, `np.int`, `np.bool` | `byte_tracker.py:18`, `matching.py:61,66,67,121,124,127` | numpy 1.24 |
| `np.asfarray` | `motmetrics/distances.py:117` | numpy 2.0 |

**ByteTrack 원본을 수정하지 않기 위해** 우리 쪽에서 별칭을 되살렸다.
`yolox` import 보다 먼저 `import attack.compat` 해야 한다.
numpy 가 구버전이면 자동으로 무동작이므로 랩 PC 환경에도 안전하다.

### 7.4 환경 구축에서 걸린 것

로컬 WSL 에 개발 도구가 없어 순차적으로 막혔다. 랩 PC 에서는 대부분 이미 있을 것이다.

| 오류 | 원인 | 해결 |
|---|---|---|
| `cython_bbox` 휠 빌드 실패 (`gcc` 없음) | 컴파일러 미설치 | `sudo apt-get install build-essential` |
| `fatal error: Python.h` | Python 개발 헤더 없음 | `sudo apt-get install python3-dev` |
| `np.float` AttributeError | numpy 2.x | `attack/compat.py` |
| `np.asfarray` AttributeError | numpy 2.x | `attack/compat.py` |

추가 패키지: `motmetrics`, `lap`, `cython_bbox`

### 7.5 클린 기준값 (MOT17-02-FRCNN, val 절반 300프레임, 800x1440, YOLOX-S)

| 지표 | 값 |
|---|---|
| MOTA | 55.1% |
| IDF1 | 53.1% |
| IDSW | 66 |
| Precision / Recall | 89.8% / 62.9% |

소요 373초 (1.25초/프레임, CPU)

> 공식 ByteTrack 의 MOT17 MOTA(80%대)보다 낮은 것은 YOLOX-S 사용, 단일 시퀀스,
> 그중 절반만 평가하기 때문이다. **절대값이 아니라 공격 전후 차이**를 보는 것이 목적이지만,
> 논문에는 동일 조건의 클린 기준값을 반드시 명시해야 한다.

### 7.6 ⭐ 랜덤 노이즈 baseline

학습 전(랜덤 초기화) generator 로 평가해, **학습된 공격이 넘어야 할 하한선**을 확보했다.

MOT17-02-FRCNN, 320x576, 100프레임 (CPU 시간 제약)

| | MOTA | IDF1 | IDSW |
|---|---|---|---|
| 클린 | 33.3% | 46.1% | 6 |
| 랜덤 노이즈 (eps=10/255) | 32.1% | 47.0% | 7 |
| **차이** | **-1.2** | **+0.9** | +1 |

**랜덤 노이즈로는 사실상 아무 일도 일어나지 않는다.**

이 행은 논문에 반드시 포함해야 한다. 리뷰어의
*"단지 노이즈를 넣어서 성능이 떨어진 것 아닌가"* 를 막는 근거이며,
학습된 공격의 효과를 판정하는 기준선이 된다.

| 학습 후 결과 | 해석 |
|---|---|
| IDF1 이 이 수준(±1)만 움직임 | 공격이 작동하지 않음 |
| IDF1 이 유의하게 하락 | 학습이 실제로 의미 있음 |

### 7.7 CPU 속도 측정 (generator 가 병목)

| 해상도 | generator | 검출기 | 300프레임 환산 |
|---|---|---|---|
| 320x576 | 2.38초 | ~0.20초 | ~13분 |
| 448x800 | 5.19초 | ~0.39초 | ~28분 |
| 608x1088 | 11.92초 | ~0.72초 | ~64분 |
| **800x1440** | **30.36초** | 1.25초 | **~158분** |

CPU 에서 원본 해상도로 공격 평가를 하는 것은 비현실적이다.
**공정 비교를 위해 클린도 같은 해상도로 다시 측정해야 한다**는 점에 주의.

---

## 8. 파일 인벤토리 (2차 시점)

| 파일 | 출처 | 검증 |
|---|---|---|
| `attack/dataset.py` | 직접 작성 | 실물 실행 + 시각 검증 |
| `attack/descriptor.py` | 직접 작성 | 단위 테스트 |
| `attack/losses.py` | 혼합 (상단 신규 / 하단 SCGA) | 단위 테스트 |
| **`attack/generator_wrap.py`** | **직접 작성 (2차 신규)** | e2e |
| **`attack/compat.py`** | **직접 작성 (2차 신규)** | 평가 실행 |
| `attack/train_generator.py` | SCGA 골격 + 직접 작성 | e2e |
| **`attack/evaluate.py`** | **직접 작성 (2차 신규)** | 클린·공격 실행 |
| `attack/smoke_test.py` | 직접 작성 | 테스트 7개 통과 |
| `attack/viz_dataset.py` | 직접 작성 | 실행 확인 |
| **`attack/e2e_test.py`** | **직접 작성 (2차 신규)** | 통과 |

`generator/`, `yolox/`, `tools/`, `exps/` 는 여전히 **원본 무수정**이다.
numpy 호환 문제도 원본을 고치지 않고 `attack/compat.py` 로 처리했다.

---

## 9. 남은 작업

| 항목 | 비고 |
|---|---|
| **GPU에서 실제 학습** | 랩 PC(24GB) 확보 후 — 현재 최대 블로커 |
| VRAM 측정 | YOLOX-S + 320x576 부터 시작해 키우기 |
| ~~평가 스크립트~~ | ~~완료 (7절)~~ |
| 평가 시퀀스 확장 | 현재 MOT17-02 만. train 7개 시퀀스 전체로 |
| `alpha`, `lam` 탐색 | `alpha`는 1.0 부근(CHANGES_v1 5.1절) |
| ablation A1~A7 | DESIGN.md 7.5절 |
| FairMOT 연동 | Step 3, DCNv2 컴파일 확인 필요 |
| 논문 Related Work 초안 | BankTweak 포함. gap 문장 3개는 DESIGN.md 4.1.1절 |

### 이번에 추가된 ablation 항목

| # | 비교 | 막으려는 질문 |
|---|---|---|
| A6 | 중심화 on/off | "중심화가 실제로 필요한가?" — 이미 판별 격차 0.19 vs 1.04 로 근거 확보 |
| A7 | `roi_source` pred vs gt | "ROI를 어디서 잡아야 하는가?" |
