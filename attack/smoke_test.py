# -*- coding: utf-8 -*-
"""
CPU 스모크 테스트 — GPU/데이터 없이 형태와 gradient 흐름만 확인한다.

실행:
    source ~/.venv/bin/activate
    cd ~/ssu/mot-scga-attack
    python attack/smoke_test.py
"""

import torch

from attack.descriptor import object_descriptor
from attack.losses import (attack_loss, temporal_decorrelation_loss,
                           cross_object_confusion_loss,
                           feature_map_similarity_hinge, project)

torch.manual_seed(0)
OK, FAIL = "  [OK]", "  [FAIL]"


def make_boxes(batch_idx, n, H, W, device='cpu'):
    """겹치지 않게 대충 흩뿌린 박스 [n, 5]"""
    cx = torch.rand(n) * (W - 100) + 50
    cy = torch.rand(n) * (H - 100) + 50
    w = torch.rand(n) * 40 + 30
    h = torch.rand(n) * 80 + 60
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    idx = torch.full((n, 1), float(batch_idx))
    return torch.cat([idx, boxes], 1).to(device)


# ── 1. descriptor 형태 ────────────────────────────────────────────
def test_descriptor_shape():
    print("[1] object_descriptor 형태")
    B, C, H, W = 2, 64, 800, 1440
    feat = torch.randn(B, C, H // 16, W // 16)      # stride 16 가정
    boxes = torch.cat([make_boxes(0, 3, H, W), make_boxes(1, 4, H, W)])
    d = object_descriptor(feat, boxes, (H, W), alpha=1.0)
    assert d.shape == (7, C + 4), d.shape
    print(f"{OK} feat{tuple(feat.shape)} + boxes{tuple(boxes.shape)} -> d{tuple(d.shape)}")


# ── 2. gradient 흐름 (핵심) ───────────────────────────────────────
def test_gradient_flow():
    print("[2] gradient 흐름 — feature 와 박스 좌표 양쪽")
    B, C, H, W = 1, 32, 800, 1440
    feat = torch.randn(B, C, H // 16, W // 16, requires_grad=True)
    boxes = make_boxes(0, 4, H, W)
    boxes = boxes.clone().requires_grad_(True)      # 박스에도 gradient 필요

    d = object_descriptor(feat, boxes, (H, W), alpha=1.0)
    d.sum().backward()

    assert feat.grad is not None and feat.grad.abs().sum() > 0
    print(f"{OK} feature  gradient 도달 (|grad|={feat.grad.abs().sum():.3f})")

    assert boxes.grad is not None and boxes.grad.abs().sum() > 0
    print(f"{OK} 박스좌표 gradient 도달 (|grad|={boxes.grad.abs().sum():.3f})")
    print("       -> 기하 성분이 실제로 박스를 움직일 수 있다")


# ── 3. 손실 동작 ──────────────────────────────────────────────────
def test_losses():
    print("[3] 공격 손실")
    D = 36
    ids_t = torch.tensor([1, 2, 3])
    ids_u = torch.tensor([1, 2, 5])                 # 3은 사라지고 5가 등장

    d_t = torch.randn(3, D, requires_grad=True)
    d_u = torch.randn(3, D, requires_grad=True)

    l_dec = temporal_decorrelation_loss(d_t, d_u, ids_t, ids_u)
    l_con = cross_object_confusion_loss(d_t, d_u, ids_t, ids_u)
    total, parts = attack_loss(d_t, d_u, ids_t, ids_u, lam=1.0)

    print(f"{OK} L_decorr={l_dec.item():+.4f}  L_confuse={l_con.item():+.4f}  total={total.item():+.4f}")
    assert -1.01 <= l_dec.item() <= 1.01, "코사인 평균은 [-1,1]"
    total.backward()
    assert d_t.grad is not None
    print(f"{OK} 손실 -> 서술자 gradient 도달")


# ── 4. 손실 방향 검증 ─────────────────────────────────────────────
def test_loss_direction():
    print("[4] 손실 방향 — 최소화가 의도한 방향인가")
    D = 16
    ids = torch.tensor([1, 2])

    same = torch.randn(2, D)
    d_t, d_u_same = same.clone(), same.clone()          # 같은 ID 가 동일
    d_u_diff = torch.stack([-same[0], -same[1]])        # 정반대

    l_same = temporal_decorrelation_loss(d_t, d_u_same, ids, ids).item()
    l_diff = temporal_decorrelation_loss(d_t, d_u_diff, ids, ids).item()

    print(f"       동일할 때 L_decorr={l_same:+.3f}  /  반대일 때 {l_diff:+.3f}")
    assert l_diff < l_same, "분리될수록 손실이 낮아야 한다"
    print(f"{OK} L_decorr 는 '같은 객체가 벌어질수록' 낮아진다")

    # L_confuse: 다른 객체끼리 비슷할수록 낮아야 함
    a = torch.randn(2, D)
    far = torch.stack([a[1], a[0]])                     # 서로 바꿔 놓으면 교차쌍이 비슷
    l_far = cross_object_confusion_loss(a, far, ids, ids).item()
    l_rand = cross_object_confusion_loss(a, torch.randn(2, D), ids, ids).item()
    print(f"       교차쌍 유사할 때={l_far:+.3f}  /  무작위={l_rand:+.3f}")
    assert l_far < l_rand
    print(f"{OK} L_confuse 는 '다른 객체가 닮을수록' 낮아진다")


# ── 5. SCGA 유틸 ──────────────────────────────────────────────────
def test_scga_utils():
    print("[5] SCGA 유틸")
    a = torch.randn(2, 8, 4, 4)
    h_same = feature_map_similarity_hinge(a, a.clone(), threshold=0.6).item()
    h_diff = feature_map_similarity_hinge(a, torch.randn(2, 8, 4, 4), threshold=0.6).item()
    print(f"       동일={h_same:.4f}  무관={h_diff:.4f}")
    assert h_same == 0.0, "threshold 이상이면 벌점 0 (hinge)"
    assert h_diff > 0
    print(f"{OK} hinge: 충분히 닮으면 0, 멀어지면 벌점")

    img = torch.rand(1, 3, 32, 32)
    adv = img + torch.randn_like(img) * 0.5
    eps = 10 / 255.0
    p = project(adv, img, eps)
    dev = (p - img).abs().max().item()
    print(f"       eps={eps:.4f}  투영 후 최대 편차={dev:.4f}")
    assert dev <= eps + 1e-6 and p.min() >= 0 and p.max() <= 1
    print(f"{OK} project: eps 상자 + [0,1] 모두 만족")


# ── 6. assign_ids — 예측-GT 매칭 ──────────────────────────────────
def test_assign_ids():
    from attack.train_generator import assign_ids
    print("[6] assign_ids — 예측-GT 매칭")

    # GT 2개
    gt = torch.tensor([[0., 100, 100, 200, 300],
                       [0., 500, 150, 600, 400]])
    gt_ids = torch.tensor([7, 9])

    # 예측 50개: 대부분 엉뚱한 곳, 3번과 41번만 GT 와 거의 일치하게 심어둠
    torch.manual_seed(1)
    pred = torch.rand(1, 50, 4) * 50 + 900          # 겹치지 않는 영역
    pred[0, :, 2:] += pred[0, :, :2]
    pred[0, 3] = torch.tensor([102., 98., 198., 302.])    # GT0 과 거의 동일
    pred[0, 41] = torch.tensor([495., 155., 605., 395.])  # GT1 과 거의 동일
    pred = pred.clone().requires_grad_(True)

    m_pred, m_gt, m_ids = assign_ids(pred, gt, gt_ids, iou_thr=0.1)

    assert m_ids.tolist() == [7, 9], m_ids.tolist()
    print(f"{OK} 매칭된 ID: {m_ids.tolist()}  (기대 [7, 9])")
    assert m_pred.shape == (2, 5) and m_gt.shape == (2, 5)
    print(f"{OK} 형태: m_pred{tuple(m_pred.shape)}  m_gt{tuple(m_gt.shape)}")

    # 심어둔 예측이 실제로 선택됐는가
    assert torch.allclose(m_pred[0, 1:], pred[0, 3], atol=1e-4)
    assert torch.allclose(m_pred[1, 1:], pred[0, 41], atol=1e-4)
    print(f"{OK} 의도한 예측(3번, 41번)이 선택됨")

    # gradient 가 원본 pred 까지 도달하는가 (핵심)
    m_pred[:, 1:].sum().backward()
    g = pred.grad[0]
    assert g[3].abs().sum() > 0 and g[41].abs().sum() > 0
    assert g[0].abs().sum() == 0        # 선택 안 된 예측엔 gradient 없음
    print(f"{OK} 선택된 예측에만 gradient 도달 (매칭은 detach, 값은 미분 가능)")


def test_assign_ids_edge():
    from attack.train_generator import assign_ids
    print("[7] assign_ids — 예외 상황")

    gt = torch.tensor([[0., 100, 100, 200, 300]])
    gt_ids = torch.tensor([7])

    # 겹치는 예측이 하나도 없을 때
    far = torch.tensor([[[900., 900., 950., 950.]]])
    _, _, ids = assign_ids(far, gt, gt_ids, iou_thr=0.1)
    assert ids.numel() == 0
    print(f"{OK} 겹치는 예측 없음 -> 빈 결과 (에러 없이)")

    # GT 가 아예 없을 때
    _, _, ids = assign_ids(far, torch.zeros(0, 5), torch.zeros(0, dtype=torch.int64))
    assert ids.numel() == 0
    print(f"{OK} GT 없음 -> 빈 결과")

    # 배치 2개에 걸쳐 있을 때 batch_idx 가 유지되는가
    gt2 = torch.tensor([[0., 100, 100, 200, 300],
                        [1., 100, 100, 200, 300]])
    pred2 = torch.zeros(2, 1, 4)
    pred2[0, 0] = torch.tensor([100., 100., 200., 300.])
    pred2[1, 0] = torch.tensor([100., 100., 200., 300.])
    mp, _, mi = assign_ids(pred2, gt2, torch.tensor([7, 9]), iou_thr=0.1)
    assert mp[:, 0].tolist() == [0.0, 1.0] and mi.tolist() == [7, 9]
    print(f"{OK} 배치 인덱스 유지: {mp[:, 0].tolist()}")


if __name__ == '__main__':
    for fn in (test_descriptor_shape, test_gradient_flow, test_losses,
               test_loss_direction, test_scga_utils,
               test_assign_ids, test_assign_ids_edge):
        fn()
        print()
    print("전부 통과")
