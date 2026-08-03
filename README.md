# Paradigm-Neutral Temporal Consistency Attack on MOT

SCGA(ICLR 2026)의 생성형 전이 공격을 다중객체추적(MOT)으로 확장하는 연구.
기존 MOT 공격이 특정 트래커의 구성요소에 결박되어 계열을 넘지 못하는 문제를,
**모든 검출기에 존재하는 백본 feature와 박스만으로 정의한 시간 일관성**을
교란하는 방식으로 해결하려 한다.

학부연구생 김영재 · 숭실대학교

---

## 저장소 구조

```
mot-scga-attack/
├── yolox/          ByteTrack 원본 — 검출기(YOLOX), 트래커, 데이터, 평가
├── tools/          ByteTrack 원본 — 실행 스크립트
├── exps/           ByteTrack 원본 — 실험 설정
├── generator/      SCGA에서 가져옴 — 섭동 생성기 (ResNet 계열)
├── attack/         ★ 본 연구
│   ├── descriptor.py   객체 서술자 (백본 feature + 박스 기하)
│   └── losses.py       공격 손실 2개 + SCGA 유틸(EMA, hinge, projection)
├── docs/
│   └── DESIGN.md       ★ 설계 문서 (근거, 방법, 실행 계획)
└── reference/
    └── scga_train_reference.py   SCGA 원본 학습 스크립트 (참고용)
```

## 핵심 아이디어

기존 MOT 공격의 손실은 특정 트래커에만 있는 것을 표적으로 삼는다.

| 논문 | 표적 | 없으면 |
|---|---|---|
| BBA | CenterNet 계열 ReID 헤드 | ByteTrack에서 MOTA -4% (사실상 실패) |
| TraSA | re-ID 브랜치 + YOLOX grid 인덱스 | ReID 없는 트래커엔 절반만 적용 |
| FADE | MOTR의 track query | TBD 계열로 전이 불가 |

본 연구는 **백본 feature와 박스**만 쓴다. 둘 다 모든 검출기에 존재한다.

```
d_k = [ L2norm(GAP(ROIAlign(F, b_k))) ; alpha * (cx, cy, w, h) ]

L_decorr  = mean_k  cos( d_t[k],  d_{t+1}[k] )          # 같은 객체를 시간축에서 분리
L_confuse = -mean_{k!=j} cos( d_t[k], d_{t+1}[j] )      # 다른 객체를 서로 근접
```

SCGA의 generator와 EMA 일관성 구조는 그대로 재사용하고,
데이터(프레임 쌍)·surrogate(검출기)·손실만 교체한다.

자세한 근거와 계획은 [docs/DESIGN.md](docs/DESIGN.md) 참조.

## 관련 저장소

| 용도 | 위치 |
|---|---|
| SCGA 원본 | `~/ssu/scga` |
| TraSw의 ByteTrack 공격 (참고 구현) | `~/ssu/ByteTrack-attack` |
| 평가 툴킷 | [TrackEval](https://github.com/JonathonLuiten/TrackEval) |

## 진행 상황

- [x] SCGA 논문·코드 정독
- [x] MOT 도메인 및 선행 연구 조사
- [x] ByteTrack 트래커·매칭·공격 코드 분석
- [x] 설계 문서 작성
- [ ] 환경 구축 (GPU 확보 대기)
- [ ] 사전 검증 실험 (시간 일관성 신호 상관)
- [ ] ByteTrack 공격 구현
- [ ] FairMOT 추가 — ReID 유무 전이 차이 검증
- [ ] MOTR 계열 — 패러다임 간 전이

원본 ByteTrack README는 [README_ByteTrack.md](README_ByteTrack.md) 참조.
