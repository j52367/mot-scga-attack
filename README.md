# Paradigm-Neutral Temporal Consistency Attack on MOT

SCGA(ICLR 2026)의 생성형 전이 공격을 다중객체추적(MOT)으로 확장하는 연구.
기존 MOT 공격이 특정 트래커의 구성요소에 결박되어 계열을 넘지 못하는 문제를,
**모든 검출기에 존재하는 백본 feature와 박스만으로 정의한 시간 일관성**을
교란하는 방식으로 접근한다.

학부연구생 김영재 · 숭실대학교

---

## 핵심 아이디어

기존 MOT 공격의 손실은 특정 트래커에만 있는 것을 표적으로 삼는다.

| 논문 | 표적 | ReID 없는 ByteTrack에서 |
|---|---|---|
| BBA (Neural Networks 2024) | CenterNet 계열 ReID 헤드 | MOTA -4% (사실상 실패) |
| TraSA (ASOC 2024) | re-ID 브랜치 + YOLOX grid | 절반만 적용 |
| BankTweak (IJCAI 2025) | ReID feature bank | 적용 불가 |
| FADE (CVPR 2026) | MOTR track query | 존재하지 않음 |

본 연구는 **백본 feature와 박스 기하**만 쓴다. 둘 다 모든 검출기에 존재한다.

```
d_k = [ L2norm(GAP(ROIAlign(F, b_k))) ; alpha * (cx, cy, w, h) ]

L_decorr  =  mean_k  cos( d_t[k],  d_{t+1}[k] )       # 같은 객체를 시간축에서 분리
L_confuse = -mean_{k!=j} cos( d_t[k], d_{t+1}[j] )    # 다른 객체를 서로 근접
```

두 손실은 계산 전 **중심화(공통 평균 제거)** 한다.
중심화하지 않으면 서로 다른 객체의 코사인이 이미 +0.80으로 포화해 학습을 이끌지 못한다.

SCGA의 generator와 EMA 일관성 구조는 그대로 재사용하고,
데이터(프레임 쌍)·surrogate(검출기)·손실만 교체한다.

---

## 저장소 구조

```
mot-scga-attack/
├── yolox/ tools/ exps/     ByteTrack 원본 (무수정)
├── generator/              SCGA 원본 (무수정)
├── reference/              SCGA 학습 스크립트 (참고용)
└── attack/                 ★ 본 연구
    ├── dataset.py            MOT17 프레임 쌍 + 박스 + ID
    ├── descriptor.py         객체 서술자
    ├── losses.py             L_decorr, L_confuse + SCGA 유틸
    ├── generator_wrap.py     generator 블록 feature 추출
    ├── compat.py             numpy 2.x 호환 계층
    ├── train_generator.py    학습
    ├── evaluate.py           공격 효과 평가 (MOTA/IDF1/IDSW)
    ├── diagnose.py           실패 원인 진단 (노이즈·검출분포·서술자)
    ├── smoke_test.py         단위 테스트
    ├── e2e_test.py           end-to-end 1스텝 검증
    └── viz_dataset.py        데이터로더 시각 검증
```

**ByteTrack과 SCGA 원본은 수정하지 않는다.**
numpy 호환 문제도 원본 대신 `attack/compat.py`에서 처리한다.

---

## 설치

```bash
conda create -n mot python=3.12 -y
conda activate mot

# PyTorch (GPU 아키텍처에 맞는 빌드를 쓸 것)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install opencv-python-headless motmetrics lap cython_bbox \
            loguru thop tabulate pycocotools gdown kagglehub tqdm
```

`cython_bbox` 빌드에는 컴파일러와 Python 헤더가 필요하다.

```bash
sudo apt-get install -y build-essential python3-dev
```

GPU 확인 (`capability`가 `arch list`에 포함돼야 한다)

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), \
torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
```

---

## 데이터와 가중치

**MOT17** — 저장소 밖에 두고 심볼릭 링크만 건다.

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('wenhoujinjust/mot-17'))"
ln -sfn <위 경로>/MOT17 datasets/mot
```

기대 구조

```
datasets/mot/train/MOT17-02-FRCNN/
├── img1/000001.jpg ...
├── gt/gt.txt        frame,id,x,y,w,h,conf,class,visibility
└── seqinfo.ini
```

**검출기 가중치** — ByteTrack 모델 zoo

```bash
mkdir -p pretrained
gdown 1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj -O pretrained/bytetrack_s_mot17.pth.tar  # YOLOX-S
gdown 1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5 -O pretrained/bytetrack_x_mot17.pth.tar  # YOLOX-X
```

---

## 검증

모든 명령은 저장소 루트에서 `PYTHONPATH=.` 를 붙여 실행한다.

```bash
# 단위 테스트 (GPU 불필요)
PYTHONPATH=. python attack/smoke_test.py

# 데이터로더 시각 검증 — output/viz/pair_*.jpg 확인
PYTHONPATH=. python attack/viz_dataset.py --n 3

# end-to-end 1스텝 (학습 경로 확인)
PYTHONPATH=. python attack/e2e_test.py
```

---

## 학습

```bash
PYTHONPATH=. python attack/train_generator.py \
    --data_root datasets/mot \
    --exp_file exps/example/mot/yolox_s_mix_det.py \
    --ckpt pretrained/bytetrack_s_mot17.pth.tar \
    --batch_size 4 --epochs 10
```

주요 인자

| 인자 | 기본 | 설명 |
|---|---|---|
| `--eps` | 10 | L-inf 예산 (x/255) |
| **`--delta_param`** | **additive** | 섭동 파라미터화. 아래 참고 |
| **`--w_iou`** | **0.0** | `L_iou` 가중치. 0 이면 끈다 |
| **`--w_keep`** | **1.0** | `L_keep` 가중치. 검출을 클린 상태로 붙잡아 둔다 |
| `--alpha` | 1.0 | 서술자의 기하 성분 가중치 |
| `--lam` | 1.0 | `L_confuse` 가중치 |
| `--fpn_level` | 1 | 백본 레벨 (0=stride8, 1=stride16, 2=stride32) |
| `--iou_thr` | 0.1 | 예측-GT 매칭 최소 IoU |
| `--roi_source` | pred | ROI를 예측 박스로 잡을지 GT로 잡을지 |
| `--no_center` | off | 서술자 중심화 해제 (ablation용) |
| `--w_consistency` | 0.7 | SCGA 일관성 손실 가중치 |

결과는 `experiments/<시각>_<method>_<generator>/` 에 저장된다.
저장되는 것은 student가 아니라 **teacher(EMA)** 이다.

### 오래 걸리는 학습은 백그라운드로

SSH 를 끊어도 계속 돌도록 `nohup` 으로 띄우고 로그를 파일에 남긴다.

```bash
mkdir -p logs
PYTHONPATH=. nohup python attack/train_generator.py \
    --data_root datasets/mot \
    --exp_file exps/example/mot/yolox_s_mix_det.py \
    --ckpt pretrained/bytetrack_s_mot17.pth.tar \
    --batch_size 4 --epochs 3 --w_iou 1.0 \
    > logs/train.log 2>&1 &
```

`PYTHONPATH=.` 를 빠뜨리면 `ModuleNotFoundError: No module named 'generator'` 로 즉시 죽는다.
저장소 루트에서 실행하는 다른 명령과 마찬가지다.

터미널이 아니면 tqdm 을 자동으로 끄고 `--log_every` (기본 50) 마다 평문 한 줄만 남긴다.
tqdm 은 `\r` 로 덮어쓰기 때문에 파일로 넘기면 읽을 수 없다.

```
[1/3] 50/5949 loss=1.8021 dec=0.9612 con=0.0488 iou=0.8433 gen=0.0000 skip=0 | 0.8분 경과, 남은 92분
```

같은 값이 `experiments/<폴더>/loss_log.csv` 에도 쌓인다 (손실 곡선용).

```bash
tail -f logs/train.log          # 진행 보기
nvidia-smi                       # 살아 있는지 확인
```

### 섭동 파라미터화 (`--delta_param`)

SCGA/BIA 원본(`image`)은 generator 가 `tanh` 로 **이미지 한 장**을 내고 그것을 eps 상자로 자른다.
eps 공 안에 들어가려면 generator 가 먼저 항등 사상(입력 복원)을 배워야 하는데,
ImageNet 128만장 x 10에폭이면 배우지만 MOT17 2천 스텝으로는 못 배운다.

Exp-001 실측: **픽셀의 98%가 eps 경계에서 잘렸다.** `clamp` 는 잘린 곳에 gradient 를
정확히 0 으로 흘리므로 **2%만 학습에 참여**했다. 노이즈 맵에 입력 장면이 그대로
비쳐 보이는 것도 같은 이유다 — generator 가 공격이 아니라 입력 복원 중이었다.

```
image      adv = clamp(G(x), x-eps, x+eps)      gradient 죽은 픽셀 98.7%
additive   adv = clamp(x + eps*G(x), 0, 1)      gradient 죽은 픽셀  0.0%
```

`G` 출력이 이미 `tanh` 라 `[-1,1]` 이므로 `eps` 를 곱하면 정확히 예산 안에 들어간다.
두 방식 모두 남겨 ablation 으로 쓴다. 수치는 `attack/smoke_test.py [5b]` 에서 재현된다.

**평가·진단에도 학습 때와 같은 값을 줘야 한다.** 다르면 공격이 성립하지 않는다.

### `L_iou` (`--w_iou`)

서술자 132차원 중 128차원이 외형인데, **ByteTrack 은 외형을 전혀 쓰지 않는다.**
(BYTE 1차 cost = `1 - IoU*score`, 2차 cost = `1 - IoU`)
외형 공격이 완벽히 성공해도 ByteTrack 의 IDF1 은 움직이지 않는다.

```
L_iou = mean_k IoU( b_t[k], b_{t+1}[k] )     # 같은 객체의 박스를 프레임 간에 벌린다
```

박스는 모든 트래커가 쓰므로 패러다임 중립성은 유지된다.
목표 수준: 2차 매칭 탈락은 `IoU < 0.5`, 1차 매칭 탈락은 `IoU * score < 0.1`.
인접 프레임 같은 사람의 클린 IoU 는 보통 0.8~0.9 다.

### `L_keep` (`--w_keep`) — 없으면 반드시 검출이 무너진다

서술자의 외형 성분은 `GAP(ROIAlign(F, b))` 이고, **검출 헤드도 같은 `F` 를 읽는다.**

```
F = backbone(x)
 ├→ head(F)              → 박스, 점수     (검출)
 └→ GAP(ROIAlign(F, b))  → 서술자 외형    (공격 대상)
```

따라서 `L_decorr` 을 낮추는 가장 쉬운 길은 `F` 를 난장판으로 만드는 것이고,
그러면 검출이 같이 죽는다. 최적화는 매번 그 지름길을 택한다.

| 실측 | Precision | Recall | 결과 |
|---|---|---|---|
| 클린 | 89.7 | 62.8 | — |
| Exp-002 (`w_iou 1.0`, `w_keep 0`) | 42.1 | **5.5** | 검출 억제 |
| Exp-003 (`w_iou 0`, `w_keep 0`) | **0.1** | 5.9 | 오탐 홍수 (프레임당 트랙 1,873개) |

`w_iou` 를 꺼도 같은 일이 났으므로 원인은 손실 항이 아니라 **분리 장치의 부재**다.

```
L_keep = mean | s_adv - s_clean |            모든 예측(약 23,600개)의 점수
       + mean_k ( 1 - IoU(b_adv_k, gt_k) )   매칭된 박스를 GT 에 고정
```

점수 항은 **양방향** 벌점이라 억제와 홍수를 모두 막는다.
전체 예측에 걸어야 GT 가 없는 위치의 오탐까지 잡힌다.

학습 로그의 `dscore`(클린 대비 점수 편차)와 `keepiou`(예측-GT 겹침)로 검출 건강도를
실시간 확인한다. `dscore` 가 커지거나 `keepiou` 가 떨어지면 검출을 부수는 중이다.

`--w_keep 0` 은 위 실패를 재현하는 ablation 이다.
검출기 forward 가 프레임당 1회 늘어 약 40% 느려진다.

---

## 평가

클린과 공격을 같은 스크립트로 처리한다. `--attack_ckpt` 유무로만 갈린다.

```bash
# 클린 기준값
PYTHONPATH=. python attack/evaluate.py --seqs MOT17-02-FRCNN --tag clean

# 공격 적용
PYTHONPATH=. python attack/evaluate.py --seqs MOT17-02-FRCNN \
    --attack_ckpt experiments/<폴더>/teacher_ptca_final.pth --tag attack
```

학습에 `--half first` 를 쓰므로 평가는 기본 `--half second` 로 겹치지 않는다.

**지표 해석**

| 지표 | 의미 | 주의 |
|---|---|---|
| MOTA | 검출 종합 | 오탐만 뿌려도 크게 떨어진다 |
| **IDF1** | **ID 유지 품질** | ★ 이것이 떨어져야 추적을 무너뜨린 것 |
| IDSW | ID 전환 횟수 | |

---

## 진단

평가 지표가 예상과 다르게 나왔을 때, 학습을 다시 돌리기 전에 원인을 좁힌다.
같은 프레임을 클린/공격 두 경로로 돌려 네 가지를 한 번에 본다.

```bash
PYTHONPATH=. python attack/diagnose.py \
    --attack_ckpt experiments/<폴더>/teacher_ptca_final.pth
```

| 항목 | 무엇을 가르는가 |
|---|---|
| 노이즈 통계 | **구조/상수 비**가 1 미만이면 공격이 아니라 전역 톤 이동을 배운 것 |
| 이미지 덤프 | `output/diagnose/<폴더>/` 에 클린·공격·δ 증폭 3장 |
| 검출 분포 | BYTE 1차(≥`track_thresh`)·2차(0.1~thr) 구간별 개수. **low 만 줄면 공격이 트래커를 청소해 준 것** |
| 서술자 코사인 | 평가 구간에서 같은-ID 코사인이 안 변하면 학습 실패, 내려갔는데 IDF1이 멀쩡하면 손실-실효 불일치 |

마지막에 판정 문구가 출력된다. 서술자·공격 인자는 **학습 때와 같은 값**을 줘야 한다.

---

## 현재 상태

| 항목 | 상태 |
|---|---|
| 단위 테스트 / e2e | 통과 (CPU) |
| 데이터로더 시각 검증 | 통과 |
| 랜덤 노이즈 baseline | IDF1 변화 없음 — 학습된 공격이 넘어야 할 하한선 |
| GPU 학습 (1 epoch) | 완료 — 30분 / 1,983 스텝 |
| Exp-001 클린 (MOT17-02-FRCNN, val 절반, 800x1440, YOLOX-S) | MOTA 54.9 / IDF1 52.7 / IDSW 68 |
| Exp-001 공격 | MOTA 55.4 / **IDF1 60.9** / IDSW 48 |

Exp-001 은 **공격이 추적을 오히려 개선했다**(IDF1 +8.2, IDSW -20).
Precision 은 오르고 Recall 만 떨어진 패턴이라, 노이즈가 저신뢰 검출을 지워
ByteTrack 의 2차 매칭을 정리해 준 것으로 의심된다. 원인 확인은 위 **진단** 절차로 한다.

---

## 관련 저장소

| 용도 | 위치 |
|---|---|
| ByteTrack | https://github.com/FoundationVision/ByteTrack |
| SCGA | https://github.com/andyj1/SCGA |
| TraSw의 ByteTrack 공격 (참고 구현) | https://github.com/DerryHub/ByteTrack-attack |
| 평가 툴킷 | https://github.com/JonathonLuiten/TrackEval |

원본 ByteTrack README는 [README_ByteTrack.md](README_ByteTrack.md) 참조.
