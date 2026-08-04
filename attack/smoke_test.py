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


# ── 5b. 섭동 파라미터화 ───────────────────────────────────────────
def test_apply_perturbation():
    from attack.losses import apply_perturbation
    print("[5b] 섭동 파라미터화 — gradient 가 살아 있는가")

    eps = 10 / 255.0
    img = torch.rand(1, 3, 32, 32) * 0.6 + 0.2      # 경계 clamp 를 피해 중간값으로
    # generator 출력을 흉내 낸다: tanh 라 [-1,1] 이고 대부분 포화되어 있다
    raw = torch.tanh(torch.randn(1, 3, 32, 32) * 5).requires_grad_(True)

    dead = {}
    for mode in ('image', 'additive'):
        r = raw.detach().clone().requires_grad_(True)
        out = apply_perturbation(r, img, eps, mode)
        dev = (out - img).abs().max().item()
        assert dev <= eps + 1e-6, f'{mode}: eps 예산 초과'
        assert out.min() >= 0 and out.max() <= 1

        out.sum().backward()
        dead[mode] = (r.grad.abs() == 0).float().mean().item()

    print(f"       gradient 가 0 인 픽셀 비율   "
          f"image={dead['image']*100:.1f}%   additive={dead['additive']*100:.1f}%")
    assert dead['image'] > 0.5, "image 방식은 대부분 잘려 gradient 가 죽어야 한다"
    assert dead['additive'] < 0.01, "additive 방식은 거의 모든 픽셀에 gradient 가 흘러야 한다"
    print(f"{OK} image 는 클리핑으로 gradient 가 죽고, additive 는 살아 있다")
    print(f"{OK} 두 방식 모두 eps 예산과 [0,1] 을 지킨다")

    # zero_dc — 전역 톤 이동(자명해)을 표현할 수 없어야 한다
    const = torch.full((1, 3, 32, 32), 0.9)               # 균일한 상수 출력
    d_plain = apply_perturbation(const, img, eps, 'additive') - img
    d_nodc = apply_perturbation(const, img, eps, 'additive', zero_dc=True) - img
    c_plain = d_plain.mean(dim=(2, 3)).abs().mean().item()
    c_nodc = d_nodc.mean(dim=(2, 3)).abs().mean().item()
    print(f"       상수 출력의 DC 성분   기본={c_plain:.4f}  zero_dc={c_nodc:.4f}")
    assert c_plain > 0.5 * eps, "기본 방식은 톤 이동을 그대로 통과시킨다"
    assert c_nodc < 0.01 * eps, "zero_dc 는 상수 성분을 제거해야 한다"
    print(f"{OK} zero_dc 는 전역 톤 이동을 표현할 수 없다")

    # 구조가 있는 출력은 예산을 그대로 쓰면서 gradient 도 살아야 한다
    r = torch.randn(1, 3, 32, 32).requires_grad_(True)
    out = apply_perturbation(r, img, eps, 'additive', zero_dc=True)
    dev = (out - img).abs()
    assert dev.max().item() <= eps + 1e-6, "zero_dc 여도 eps 예산은 지켜야 한다"
    out.sum().backward()
    assert (r.grad.abs() == 0).float().mean().item() < 0.01
    print(f"       구조 있는 출력  평균 |delta|={dev.mean().item()/eps*100:.0f}% of eps")
    print(f"{OK} 예산은 유지되고 gradient 도 살아 있다")


# ── 5c. L_iou ────────────────────────────────────────────────────
def test_temporal_iou():
    from attack.losses import temporal_iou_loss
    print("[5c] L_iou — 같은 객체의 프레임 간 박스 겹침")

    ids = torch.tensor([7, 9])
    b_t = torch.tensor([[0., 100, 100, 200, 300],
                        [0., 500, 150, 600, 400]])
    # 거의 안 움직인 경우 -> IoU 높음
    near = b_t.clone(); near[:, 1:] += 5
    # 크게 벗어난 경우 -> IoU 낮음
    far = b_t.clone(); far[:, 1:] += 300

    l_near = temporal_iou_loss(b_t, near, ids, ids).item()
    l_far = temporal_iou_loss(b_t, far, ids, ids).item()
    print(f"       거의 정지={l_near:.3f}  /  크게 이탈={l_far:.3f}")
    assert l_near > 0.8, "인접 프레임 클린 IoU 는 0.8 이상이어야 한다"
    assert l_far < 0.5, "2차 매칭 임계값 0.5 아래로 내려가야 공격이 성립한다"
    print(f"{OK} 박스가 벌어질수록 낮아진다 (최소화 방향이 맞다)")

    # 박스 좌표에 gradient 가 흐르는가
    b = b_t.clone().requires_grad_(True)
    temporal_iou_loss(b, near, ids, ids).backward()
    assert b.grad is not None and b.grad.abs().sum() > 0
    print(f"{OK} 박스 좌표에 gradient 도달 (검출기까지 이어진다)")

    # ID 가 하나도 안 겹치면 0 을 돌려주고 죽지 않아야 한다
    z = temporal_iou_loss(b_t, near, torch.tensor([1, 2]), torch.tensor([3, 4]))
    assert z.item() == 0.0
    print(f"{OK} 공통 ID 없음 -> 0 (에러 없이)")


# ── 5d. L_keep ───────────────────────────────────────────────────
def test_detection_keep():
    from attack.losses import detection_keep_loss
    print("[5d] L_keep — 검출 보존")

    torch.manual_seed(0)
    s_clean = torch.rand(1, 200)
    gt = torch.tensor([[0., 100, 100, 200, 300],
                       [0., 500, 150, 600, 400]])

    # 검출이 그대로인 경우
    l_same, p_same = detection_keep_loss(s_clean.clone(), s_clean,
                                         gt.clone(), gt)
    # 점수를 흔들고 박스도 밀어낸 경우
    moved = gt.clone(); moved[:, 1:] += 150
    l_bad, p_bad = detection_keep_loss(torch.rand(1, 200), s_clean, moved, gt)

    print(f"       그대로={l_same.item():.4f} (dscore={p_same['dscore']:.4f}, "
          f"keepiou={p_same['keepiou']:.3f})")
    print(f"       망가짐={l_bad.item():.4f} (dscore={p_bad['dscore']:.4f}, "
          f"keepiou={p_bad['keepiou']:.3f})")
    assert l_same.item() < 1e-6, "검출이 그대로면 벌점 0"
    assert l_bad.item() > l_same.item()
    print(f"{OK} 검출이 유지되면 0, 흐트러지면 벌점")

    # 보이는 검출(s > tau)은 양방향 벌점이어야 한다
    tau = 0.1
    hi = s_clean > tau
    s_up = s_clean.clone(); s_up[hi] += 0.3
    s_dn = s_clean.clone(); s_dn[hi] = (s_dn[hi] - 0.3).clamp(min=0)
    up = detection_keep_loss(s_up, s_clean, gt.clone(), gt, tau)[1]['dscore']
    dn = detection_keep_loss(s_dn, s_clean, gt.clone(), gt, tau)[1]['dscore']
    print(f"       보이는 검출  상승={up:.4f}  하락={dn:.4f}")
    assert up > 0.2 and dn > 0.1
    print(f"{OK} 보이는 검출은 양방향 — 억제도 증폭도 막는다")

    # 배경(s <= tau)은 '올라갈 때만' 벌점이어야 한다.
    # 전부 양방향으로 묶으면 백본이 통째로 고정돼 공격이 불가능해진다 (Exp-004)
    lo = ~hi
    bg_up = s_clean.clone(); bg_up[lo] = 0.9          # 배경이 치솟음 (오탐 홍수)
    bg_dn = s_clean.clone(); bg_dn[lo] = 0.0          # 배경이 더 내려감 (무해)
    p_up = detection_keep_loss(bg_up, s_clean, gt.clone(), gt, tau)[1]
    p_dn = detection_keep_loss(bg_dn, s_clean, gt.clone(), gt, tau)[1]
    print(f"       배경 상승 bgup={p_up['bgup']:.4f}  "
          f"배경 하락 bgup={p_dn['bgup']:.4f}")
    assert p_up['bgup'] > 0.5, "배경이 tau 를 넘으면 벌점 (Exp-003 오탐 홍수 차단)"
    assert p_dn['bgup'] == 0.0, "배경이 내려가는 것은 트래커가 못 보므로 무벌점"
    print(f"{OK} 배경은 단방향 — 홍수는 막고 자유도는 남긴다")

    # 매칭이 하나도 없어도 죽지 않아야 한다
    empty = torch.zeros(0, 5)
    l0, _ = detection_keep_loss(s_clean.clone(), s_clean, empty, empty, tau)
    assert l0.item() < 1e-6
    print(f"{OK} 매칭 없음 -> 점수 항만 (에러 없이)")


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
               test_apply_perturbation, test_temporal_iou, test_detection_keep,
               test_assign_ids, test_assign_ids_edge):
        fn()
        print()
    print("전부 통과")
