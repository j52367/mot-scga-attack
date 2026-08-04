# -*- coding: utf-8 -*-
"""
손실 함수

본 연구의 공격 손실 2개 + SCGA에서 그대로 가져오는 유틸 2개.

공격 손실은 트래커의 내부 구조를 전혀 참조하지 않는다.
필요한 입력은 (프레임 쌍의 서술자, ID) 뿐이며 칼만/헝가리안/ReID 헤드/track query를
어느 것도 쓰지 않는다. 이것이 '패러다임 중립'의 실체다.
"""

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
#  본 연구의 공격 손실
# ══════════════════════════════════════════════════════════════════

def _pair_by_id(ids_t, ids_t1):
    """두 프레임에 공통으로 등장하는 ID의 인덱스 쌍을 반환."""
    common = [i for i in ids_t.tolist() if i in set(ids_t1.tolist())]
    if not common:
        return None, None
    idx_t = torch.tensor([ids_t.tolist().index(c) for c in common], device=ids_t.device)
    idx_t1 = torch.tensor([ids_t1.tolist().index(c) for c in common], device=ids_t.device)
    return idx_t, idx_t1


def center_pair(d_t, d_t1):
    """
    두 프레임의 서술자를 '함께' 중심화한다 (공통 평균 성분 제거).

    왜 필요한가 — 실측으로 확인된 문제:
        중심화하지 않으면 서로 다른 객체의 코사인 유사도가 이미 +0.75~0.80 이다.
        외형 성분은 모두 "사람"이라 공통 성분이 크고, 기하 성분은 (cx,cy,w,h)/(W,H,W,H)
        가 전부 양수라 구조적으로 유사도가 높다.
        L_confuse 의 최적값이 -1 인데 시작이 -0.8 이면 학습을 이끌 여지가 거의 없다.

        중심화하면 다른 객체가 -0.06 (거의 직교) 로 떨어지고,
        같은 객체는 +0.98 을 유지해 판별 격차가 0.19 -> 1.04 로 5배가 된다.

    두 프레임을 합친 평균을 쓰는 이유:
        프레임마다 따로 중심화하면 좌표계가 어긋나 프레임 간 비교가 왜곡된다.
    """
    m = torch.cat([d_t, d_t1], dim=0).mean(dim=0, keepdim=True)
    return d_t - m, d_t1 - m


def temporal_decorrelation_loss(d_t, d_t1, ids_t, ids_t1, center=True):
    """
    L_decorr — 같은 객체의 프레임 간 서술자를 분리한다.
    최소화하면 같은 사람이 프레임마다 다르게 보여 추적이 끊긴다.

    Returns: 스칼라 (코사인 유사도 평균). 낮을수록 공격 성공.
    """
    i_t, i_t1 = _pair_by_id(ids_t, ids_t1)
    if i_t is None:
        return d_t.sum() * 0.0
    if center:
        d_t, d_t1 = center_pair(d_t, d_t1)
    a = F.normalize(d_t[i_t], p=2, dim=1)
    b = F.normalize(d_t1[i_t1], p=2, dim=1)
    return (a * b).sum(dim=1).mean()


def cross_object_confusion_loss(d_t, d_t1, ids_t, ids_t1, center=True):
    """
    L_confuse — 서로 다른 객체의 프레임 간 서술자를 근접시킨다.
    최소화하면 다른 사람끼리 비슷해져 ID 교환이 유도된다.

    Returns: 스칼라 (-유사도 평균). 낮을수록 공격 성공.
    """
    same = ids_t.view(-1, 1) == ids_t1.view(1, -1)    # 같은 ID 위치 마스크
    if same.numel() == same.sum():
        return d_t.sum() * 0.0
    if center:
        d_t, d_t1 = center_pair(d_t, d_t1)
    a = F.normalize(d_t, p=2, dim=1)
    b = F.normalize(d_t1, p=2, dim=1)
    sim = a @ b.t()                                   # [K_t, K_t1]
    return -sim[~same].mean()


def attack_loss(d_t, d_t1, ids_t, ids_t1, lam=1.0, center=True):
    """L_attack = L_decorr + lam * L_confuse"""
    l_dec = temporal_decorrelation_loss(d_t, d_t1, ids_t, ids_t1, center=center)
    l_con = cross_object_confusion_loss(d_t, d_t1, ids_t, ids_t1, center=center)
    return l_dec + lam * l_con, {"decorr": l_dec.item(), "confuse": l_con.item()}


# ══════════════════════════════════════════════════════════════════
#  SCGA에서 그대로 가져오는 부분 (reference/scga_train_reference.py)
# ══════════════════════════════════════════════════════════════════

def feature_map_similarity_hinge(fmap_ref, fmap_tgt, threshold=0.6):
    """
    SCGA의 semantic consistency 손실.
    student generator의 중간 feature를 EMA teacher와 최소 threshold 만큼 닮게 유지.
    """
    B = fmap_ref.shape[0]
    ref_n = F.normalize(fmap_ref.view(B, -1), p=2, dim=1)
    tgt_n = F.normalize(fmap_tgt.view(B, -1), p=2, dim=1)
    cos_sim = (ref_n * tgt_n).sum(dim=1)
    return F.relu(threshold - cos_sim).mean()


def generator_consistency_loss(feats_teacher, feats_student, n_blocks=6, tau0=0.6, step=0.05):
    """
    generator 블록별 hinge 를 합산한다. SCGA 의 핵심 기여.

    threshold 가 블록마다 커지므로(0.60 -> 0.85) 깊은 블록일수록 더 강하게 일치시킨다.

    ⚠️ SCGA 원본과의 차이
        원본은 `for j in range(len(feats_teacher[1:]) - 4)` 로 순회하는데,
        공개된 generator 들이 블록 리스트가 아니라 텐서 하나를 반환하기 때문에
        이 인덱싱은 배치 차원을 자르게 되어 의도대로 동작하지 않는다.
        (attack/generator_wrap.py 주석 참고)

        주석에 "thresholds for 6 blocks" 라고 적혀 있고 taus 가 6개이므로,
        여기서는 **블록 6개를 taus[0..5] 와 1:1로 대응**시키는 것으로 해석했다.

    Returns:
        스칼라 텐서 (블록이 없으면 0 텐서)
    """
    n = min(len(feats_teacher), len(feats_student), n_blocks)
    if n == 0:
        ref = feats_teacher[0] if len(feats_teacher) else feats_student[0]
        return ref.sum() * 0.0

    taus = [tau0 + step * i for i in range(n_blocks)]
    loss = feature_map_similarity_hinge(feats_teacher[0], feats_student[0], taus[0])
    for j in range(1, n):
        loss = loss + feature_map_similarity_hinge(
            feats_teacher[j], feats_student[j], threshold=taus[j])
    return loss


@torch.no_grad()
def update_ema(student, teacher, momentum=0.999):
    """teacher = momentum * teacher + (1 - momentum) * student"""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def project(adv, img, eps):
    """L-infinity eps 상자 + [0, 1] 범위로 투영."""
    return torch.clamp(torch.min(torch.max(adv, img - eps), img + eps), 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════
#  섭동 파라미터화
# ══════════════════════════════════════════════════════════════════

def apply_perturbation(raw, img, eps, mode='additive'):
    """
    generator 출력 raw 를 적대 이미지로 만든다.

    'image' — SCGA/BIA 원본 방식.
        generator 가 tanh 로 [-1,1] '이미지 한 장'을 내고, 그것을 eps 상자로 자른다.
        eps 공 안에 들어가려면 generator 가 먼저 항등 사상(입력 복원)을 배워야 한다.
        ImageNet 128만장 x 10에폭이면 배우지만, MOT17 2천 스텝으로는 못 배운다.

        ★ 실측 (Exp-001 진단): 픽셀의 98.0% 가 eps 경계에 붙어 잘렸다.
          clamp/min/max 는 잘린 곳에 gradient 를 정확히 0 으로 흘리므로
          **픽셀의 2% 만 학습에 참여**했다. 노이즈 맵에 입력 장면이 그대로
          보이는 것도 같은 이유다 — 생성기가 공격이 아니라 입력 복원 중이었다.

    'additive' — 본 연구 기본값.
        generator 출력을 '섭동 자체'로 해석한다. tanh 라 이미 [-1,1] 이므로
        eps 를 곱하면 정확히 L-inf 예산 안에 들어간다.
        클리핑이 없으니 모든 픽셀에 gradient 가 흐른다.
        (경계 [0,1] clamp 는 남지만 실제로 걸리는 픽셀은 소수다)

    두 방식을 플래그로 남겨 ablation 으로 쓴다 (DESIGN.md A8).
    """
    if mode == 'additive':
        return torch.clamp(img + eps * raw, 0.0, 1.0)
    if mode == 'image':
        return project(raw, img, eps)
    raise ValueError(f'알 수 없는 delta_param: {mode}')


# ══════════════════════════════════════════════════════════════════
#  기하 손실 — IoU 기반 연관을 직접 겨냥
# ══════════════════════════════════════════════════════════════════

def _diag_iou(a, b, eps=1e-7):
    """짝지어진 박스끼리의 IoU. a, b : [M,4] xyxy. 미분 가능."""
    x1 = torch.max(a[:, 0], b[:, 0])
    y1 = torch.max(a[:, 1], b[:, 1])
    x2 = torch.min(a[:, 2], b[:, 2])
    y2 = torch.min(a[:, 3], b[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + eps)


def temporal_iou_loss(boxes_t, boxes_t1, ids_t, ids_t1):
    """
    L_iou — 같은 객체의 프레임 간 박스 겹침을 줄인다.

    왜 필요한가:
        서술자의 132차원 중 128차원이 외형인데, ByteTrack 은 외형을 전혀 쓰지 않는다.
        (BYTE 1차 cost = 1 - IoU*score, 2차 cost = 1 - IoU)
        외형 공격이 완벽히 성공해도 ByteTrack 의 IDF1 은 움직이지 않는다.
        박스는 모든 트래커가 쓰므로 패러다임 중립성도 유지된다.

    목표 수준 (evaluate.py 의 track_thresh 0.6 / match_thresh 0.9 기준):
        1차 매칭 탈락  IoU * score < 0.1
        2차 매칭 탈락  IoU < 0.5
        인접 프레임 같은 사람의 클린 IoU 는 보통 0.8~0.9 다.

    Args:
        boxes_t, boxes_t1 : [M,5] (batch_idx, x1, y1, x2, y2) — 미분 가능
        ids_t, ids_t1     : [M]

    Returns:
        스칼라 (같은 객체의 평균 IoU). 낮을수록 공격 성공.
    """
    i_t, i_t1 = _pair_by_id(ids_t, ids_t1)
    if i_t is None:
        return boxes_t.sum() * 0.0
    return _diag_iou(boxes_t[i_t][:, 1:], boxes_t1[i_t1][:, 1:]).mean()
