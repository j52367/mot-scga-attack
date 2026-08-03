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
            loguru thop pycocotools gdown tqdm
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
| `--alpha` | 1.0 | 서술자의 기하 성분 가중치 |
| `--lam` | 1.0 | `L_confuse` 가중치 |
| `--fpn_level` | 1 | 백본 레벨 (0=stride8, 1=stride16, 2=stride32) |
| `--iou_thr` | 0.1 | 예측-GT 매칭 최소 IoU |
| `--roi_source` | pred | ROI를 예측 박스로 잡을지 GT로 잡을지 |
| `--no_center` | off | 서술자 중심화 해제 (ablation용) |
| `--w_consistency` | 0.7 | SCGA 일관성 손실 가중치 |

결과는 `experiments/<시각>_<method>_<generator>/` 에 저장된다.
저장되는 것은 student가 아니라 **teacher(EMA)** 이다.

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

## 현재 상태

| 항목 | 상태 |
|---|---|
| 단위 테스트 / e2e | 통과 (CPU) |
| 데이터로더 시각 검증 | 통과 |
| 클린 기준값 (MOT17-02-FRCNN, val 절반, 800x1440, YOLOX-S) | MOTA 55.1 / IDF1 53.1 / IDSW 66 |
| 랜덤 노이즈 baseline | IDF1 변화 없음 — 학습된 공격이 넘어야 할 하한선 |
| GPU 학습 | 미실행 |

---

## 관련 저장소

| 용도 | 위치 |
|---|---|
| ByteTrack | https://github.com/FoundationVision/ByteTrack |
| SCGA | https://github.com/andyj1/SCGA |
| TraSw의 ByteTrack 공격 (참고 구현) | https://github.com/DerryHub/ByteTrack-attack |
| 평가 툴킷 | https://github.com/JonathonLuiten/TrackEval |

원본 ByteTrack README는 [README_ByteTrack.md](README_ByteTrack.md) 참조.
