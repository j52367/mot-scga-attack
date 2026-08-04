# -*- coding: utf-8 -*-
"""
generator 학습 스크립트

SCGA(train_BIA_ours.py)의 골격을 그대로 가져오고,
surrogate(분류 CNN -> 검출기)와 손실(분류 feature -> 시간 일관성) 부분만 교체한다.

  [SCGA 그대로]  generator 구성, EMA teacher, eps 투영, 일관성 손실, 학습 루프, 저장
  [새로 작성]    검출기에서 백본 feature + 예측 박스 꺼내기, GT 매칭, 서술자, 공격 손실
"""

import argparse
import copy
import csv
import datetime
import os
import shutil
import sys
import time
import warnings

# MOT 는 프레임마다 객체 수가 달라 중간 텐서 크기가 계속 바뀐다.
# 기본 할당자는 그 때문에 조각이 심하게 나므로 expandable_segments 를 켠다.
# torch import 전에 설정해야 적용된다. 사용자가 지정했으면 그대로 둔다.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.optim as optim
from tqdm import tqdm

from generator import build_generator
from generator.resnet import weights_init_normal

from attack.dataset import (MOT17PairDataset, pair_collate,
                            normalize_for_detector)
from attack.descriptor import object_descriptor
from attack.generator_wrap import GeneratorWithFeatures
from attack.losses import (attack_loss, apply_perturbation, temporal_iou_loss,
                           detection_keep_loss, generator_consistency_loss,
                           update_ema)

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# ══════════════════════════════════════════════════════════════════
#  인자  (SCGA utee/parser.py 기반 + 본 연구용 추가)
# ══════════════════════════════════════════════════════════════════

def get_parser():
    p = argparse.ArgumentParser(description="Paradigm-neutral temporal attack on MOT")

    # ── SCGA 원본 인자 ──────────────────────────────────────────
    p.add_argument('--generator', type=str, default='resnet')
    p.add_argument('--gap', action='store_true', help='GAP generator')
    p.add_argument('--nat', action='store_true', help='NAT generator')
    p.add_argument('-b', '--batch_size', type=int, default=1)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--eps', type=float, default=10, help='L-inf 예산 (x/255)')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_freq', type=int, default=200)
    p.add_argument('--resume', type=str, default=None,
                   help='ckpt_last.pth 경로. 에폭 단위로 이어받는다')
    p.add_argument('--log_every', type=int, default=50,
                   help='이 간격마다 손실을 한 줄로 찍고 loss_log.csv 에 기록한다. '
                        'nohup 으로 파일에 남길 때 tqdm 대신 이것을 읽는다')
    p.add_argument('--output_dir', type=str, default='./experiments')
    p.add_argument('--method', type=str, default='ptca')
    p.add_argument('--ema_momentum', type=float, default=0.999)
    p.add_argument('--w_consistency', type=float, default=0.7,
                   help='SCGA 일관성 손실 가중치')

    # ── 본 연구 추가 인자 ───────────────────────────────────────
    p.add_argument('--data_root', type=str, required=True, help='MOT17 루트')
    p.add_argument('--detector_variant', type=str, default='FRCNN')
    p.add_argument('--input_h', type=int, default=800)
    p.add_argument('--input_w', type=int, default=1440)
    p.add_argument('--frame_gaps', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--alpha', type=float, default=1.0, help='기하 성분 가중치')
    p.add_argument('--lam', type=float, default=1.0, help='L_confuse 가중치')
    p.add_argument('--fpn_level', type=int, default=1,
                   help='백본 FPN 레벨 (0=stride8, 1=stride16, 2=stride32)')
    p.add_argument('--iou_thr', type=float, default=0.1,
                   help='예측-GT 매칭 최소 IoU. 검출 품질 판정이 아니라 매칭 유지가 목적이라 낮게 둔다')
    p.add_argument('--roi_source', type=str, default='pred', choices=['pred', 'gt'],
                   help='ROI 영역을 예측 박스로 잡을지 GT 로 잡을지 (ablation)')
    p.add_argument('--delta_param', type=str, default='additive',
                   choices=['additive', 'image'],
                   help="섭동 파라미터화. 'image' 는 SCGA 원본(생성기가 이미지를 내고 "
                        "eps 로 자름) 이지만 Exp-001 에서 픽셀의 98%%가 잘려 "
                        "gradient 가 죽었다. 'additive' 는 생성기 출력을 섭동으로 "
                        "직접 해석한다 (attack/losses.py apply_perturbation 참고)")
    p.add_argument('--w_iou', type=float, default=0.0,
                   help='L_iou 가중치. 같은 객체의 프레임 간 박스 겹침을 줄인다. '
                        'ByteTrack 처럼 외형을 쓰지 않는 트래커를 겨냥하려면 필요하다. '
                        '0 이면 끈다')
    p.add_argument('--zero_dc', action='store_true',
                   help='섭동에서 상수 성분(전역 톤 이동)을 제거한다. '
                        'L_keep 을 걸면 최적화가 톤 이동으로 붕괴하는데 '
                        '(실측: 예산의 92.6% 를 쓰면서 잔차는 eps 의 10%, '
                        '같은-ID 코사인 변화 +0.0003) 그 자명해를 금지한다. '
                        'additive 방식에서만 쓴다')
    p.add_argument('--w_keep', type=float, default=1.0,
                   help='L_keep 가중치. 검출(점수·박스)을 클린 상태로 붙잡아 둔다. '
                        '0 으로 두면 최적화가 백본 feature 를 부수는 지름길을 택해 '
                        '연관이 아니라 검출을 무너뜨린다 (Exp-002/003 실측). '
                        '0 은 그 상태를 재현하는 ablation 용이다. '
                        '검출기 forward 가 프레임당 1회 늘어 약 40% 느려진다')
    p.add_argument('--keep_tau', type=float, default=0.1,
                   help='L_keep 이 "트래커가 본다"고 볼 최소 점수. ByteTrack 2차 매칭 하한. '
                        '이 아래 배경 앵커는 tau 를 넘을 때만 벌점을 받는다. '
                        '전부 양방향으로 고정하면 백본을 통째로 묶어 공격이 불가능해진다 '
                        '(Exp-004 실측)')
    p.add_argument('--no_center', action='store_true',
                   help='서술자 중심화를 끈다. 켜두는 것이 기본이며, 끄면 다른 객체 간 '
                        '코사인이 +0.8 로 포화해 L_confuse 가 거의 학습을 이끌지 못한다')

    # ── 검출기 ──────────────────────────────────────────────────
    p.add_argument('--exp_file', type=str, default=None, help='ByteTrack exp 파일')
    p.add_argument('--ckpt', type=str, default=None, help='검출기 가중치')

    return p


# ══════════════════════════════════════════════════════════════════
#  검출기 준비  (SCGA 의 surrogate 로딩에 대응)
# ══════════════════════════════════════════════════════════════════

def build_detector(args, device):
    """
    ByteTrack 검출기(YOLOX)를 로드하고 freeze 한다.

    SCGA 대응:
        model = surrogate.Vgg16(); model.cuda(); model.eval()
        for p in model.parameters(): p.requires_grad_(False)

    주의:
        - requires_grad_(False) 는 '가중치를 학습하지 않는다'는 뜻일 뿐,
          입력에 대한 gradient 는 그대로 흐른다. 우리에게 필요한 것이 바로 그것이다.
        - torch.no_grad() 로 감싸면 안 된다.
        - fuse_model(conv+bn 병합)은 쓰지 않는다. 추론 속도용이며 여기서는 불필요.
    """
    from yolox.exp import get_exp

    exp = get_exp(args.exp_file, None)
    model = exp.get_model()
    model.eval()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device)

    for p in model.parameters():
        p.requires_grad_(False)

    print(f'detector loaded: {args.exp_file} | num_classes={exp.num_classes}')
    return model


# ══════════════════════════════════════════════════════════════════
#  백본 feature + 예측 박스 꺼내기
# ══════════════════════════════════════════════════════════════════

def forward_detector(model, x_norm, fpn_level):
    """
    정규화된 입력에서 (백본 FPN feature, head 예측) 을 얻는다.

    SCGA 대응:
        img_out_slices = model(normalize(img.clone()))
        img_out_slice  = img_out_slices[layer_idx]
        -> SCGA 는 층별 리스트에서 layer_idx 로 하나를 꺼냈다.
           YOLOX 는 backbone 과 head 가 분리되어 있어 backbone 을 직접 호출하면 된다.

    YOLOX 구조 (yolox/models/yolox.py):
        fpn_outs = self.backbone(x)      # YOLOPAFPN -> (pan_out2, pan_out1, pan_out0)
        outputs  = self.head(fpn_outs)   # eval 모드

    YOLOPAFPN 반환 (yolo_pafpn.py:116):
        (stride 8, stride 16, stride 32) 순서   -> fpn_level 0 / 1 / 2

    YOLOXHead eval 반환 (yolo_head.py:207-214):
        [B, N, 5 + num_classes],  N = sum(H/s * W/s)
        decode_outputs 를 거쳐 좌표는 이미 절대 픽셀(cxcywh), 입력 해상도 기준.
        채널: [cx, cy, w, h, obj, cls...]   MOT 는 num_classes=1 이므로 [.., obj, cls]

    Returns:
        feat  : [B, C, Hf, Wf]     선택한 FPN 레벨
        preds : [B, N, 5+ncls]     미분 가능
    """
    fpn_feats = model.backbone(x_norm)     # 튜플 (s8, s16, s32)
    preds = model.head(fpn_feats)          # eval 모드 -> decode 된 예측
    return fpn_feats[fpn_level], preds


def preds_to_xyxy_scores(preds):
    """
    head 출력을 서술자가 쓰는 형태로 변환.

    ByteTrack update() 의 점수 계산과 동일:
        scores = output_results[:, 4] * output_results[:, 5]   # obj * cls

    Returns:
        boxes  : [B, N, 4]  (x1, y1, x2, y2), 미분 가능
        scores : [B, N]
    """
    cx, cy, w, h = preds[..., 0], preds[..., 1], preds[..., 2], preds[..., 3]
    boxes = torch.stack([cx - w * 0.5, cy - h * 0.5,
                         cx + w * 0.5, cy + h * 0.5], dim=-1)
    scores = preds[..., 4] * preds[..., 5]
    return boxes, scores


# ══════════════════════════════════════════════════════════════════
#  예측 박스에 GT ID 붙이기
# ══════════════════════════════════════════════════════════════════

def assign_ids(pred_boxes, gt_boxes, gt_ids, iou_thr=0.1):
    """
    검출기 예측 박스를 GT 와 IoU 로 매칭해 ID 를 부여한다.

    왜 필요한가:
        기하 성분 g_k 에 gradient 가 흐르려면 '예측 박스'를 써야 한다.
        GT 박스는 상수라 gradient 가 없다.
        매칭 자체는 미분 불가하지만, 어느 예측을 쓸지 고르는 용도이므로 문제없다.
        (ByteTrack-attack 이 칼만 예측 좌표를 상수로만 받아온 것과 같은 패턴)

    gradient 소실이 문제되지 않는 이유:
        YOLOX head 는 신뢰도와 무관하게 모든 그리드 셀에 대해 박스를 출력한다.
        (N = sum over 3 levels of H/s * W/s, 800x1440 기준 약 23,600개)
        박스가 사라지는 것은 postprocess(NMS + 임계값) 이후이지 head 출력 단계가 아니다.
        따라서 '점수 필터를 하지 않고' 매칭하면 공격이 아무리 성공해도 매칭 대상이 남는다.
        iou_thr 을 0.1 로 낮게 두는 것도 같은 이유다 — 검출 품질 판정이 아니라
        매칭 유지가 목적이다.

    Args:
        pred_boxes : [B, N, 4]  xyxy, 미분 가능
        gt_boxes   : [K, 5]     (batch_idx, x1, y1, x2, y2)
        gt_ids     : [K]

    Returns:
        matched_pred : [M, 5]  (batch_idx, xyxy) 미분 가능 — 기하 성분/ROI 용
        matched_gt   : [M, 5]  대응하는 GT 박스   — roi_source='gt' 일 때 사용
        matched_ids  : [M]
    """
    from torchvision.ops import box_iou

    device = pred_boxes.device
    empty = (torch.zeros(0, 5, device=device),
             torch.zeros(0, 5, device=device),
             torch.zeros(0, dtype=torch.int64, device=device))
    if gt_boxes.numel() == 0:
        return empty

    out_pred, out_gt, out_ids = [], [], []
    for b in range(pred_boxes.shape[0]):
        sel = gt_boxes[:, 0] == b
        if sel.sum() == 0:
            continue
        g, gid = gt_boxes[sel], gt_ids[sel]

        # 매칭은 detach 한 값으로 — 인덱스만 얻으면 된다
        with torch.no_grad():
            iou = box_iou(g[:, 1:], pred_boxes[b].detach())      # [Kb, N]
            best_iou, best_idx = iou.max(dim=1)
            keep = best_iou >= iou_thr
        if keep.sum() == 0:
            continue

        # 원본(미분 가능) 텐서를 인덱싱 -> gradient 유지
        picked = pred_boxes[b][best_idx[keep]]                    # [M, 4]
        bi = torch.full((picked.shape[0], 1), float(b), device=device)
        out_pred.append(torch.cat([bi, picked], dim=1))
        out_gt.append(g[keep])
        out_ids.append(gid[keep])

    if not out_pred:
        return empty
    return torch.cat(out_pred), torch.cat(out_gt), torch.cat(out_ids)

    # 참고: 서로 다른 GT 가 같은 예측에 매칭될 수 있으나(중복),
    #       예측이 23,600개이고 사람들의 위치가 서로 다르므로 실제로는 드물다.
    #       문제가 되면 IoU 내림차순 그리디 유일 매칭으로 바꾼다.


# ══════════════════════════════════════════════════════════════════
#  한 프레임의 서술자 만들기  (본 연구의 핵심)
# ══════════════════════════════════════════════════════════════════

def frame_descriptor(model, x01, boxes_gt, ids_gt, args):
    """
    [0,1] 이미지 한 프레임 -> 객체 서술자

    흐름:
        x01 -> normalize_for_detector -> forward_detector
            -> preds_to_xyxy_scores -> assign_ids -> object_descriptor

    roi_source:
        'pred' — ROI 를 예측 박스로 잡는다. 트래커가 실제로 보는 영역과 일치.
        'gt'   — ROI 를 GT 박스로 잡는다. "그 사람이 실제로 있는 곳"의 외형.
        둘 다 정당하므로 ablation 항목으로 둔다 (기본 pred).
        어느 쪽이든 기하 성분은 항상 예측 박스에서 오므로 gradient 는 유지된다.

    Returns:
        dict
            d      : [M, C+4]  서술자. 매칭 실패 시 [0, C+4]
            ids    : [M]
            box    : [M, 5] (batch_idx, xyxy) 매칭된 예측 박스. L_iou 가 쓴다
            gt     : [M, 5] 대응하는 GT 박스.   L_keep 이 쓴다
            scores : [B, N]  전체 예측 점수.    L_keep 이 쓴다
                     (GT 가 없는 위치의 오탐까지 잡으려면 전체가 필요하다)
    """
    H, W = x01.shape[-2:]

    x_norm = normalize_for_detector(x01)
    feat, preds = forward_detector(model, x_norm, args.fpn_level)
    pred_boxes, scores = preds_to_xyxy_scores(preds)

    m_pred, m_gt, m_ids = assign_ids(pred_boxes, boxes_gt, ids_gt,
                                     iou_thr=args.iou_thr)
    if m_ids.numel() == 0:
        return {'d': torch.zeros(0, feat.shape[1] + 4, device=x01.device),
                'ids': m_ids, 'box': m_pred, 'gt': m_gt, 'scores': scores}

    if args.roi_source == 'gt':
        # ROI 는 GT 로 뽑고, 기하 성분만 예측 박스에서 가져온다
        d_app = object_descriptor(feat, m_gt, (H, W), alpha=0.0)[:, :-4]
        g = m_pred[:, 1:]
        geo = torch.stack([(g[:, 0] + g[:, 2]) * 0.5 / W,
                           (g[:, 1] + g[:, 3]) * 0.5 / H,
                           (g[:, 2] - g[:, 0]) / W,
                           (g[:, 3] - g[:, 1]) / H], dim=1)
        d = torch.cat([d_app, args.alpha * geo], dim=1)
    else:
        d = object_descriptor(feat, m_pred, (H, W), alpha=args.alpha)

    return {'d': d, 'ids': m_ids, 'box': m_pred, 'gt': m_gt, 'scores': scores}


def clean_scores(model, x01, args):
    """클린 입력의 예측 점수. L_keep 의 기준값이라 gradient 가 필요 없다."""
    with torch.no_grad():
        _, preds = forward_detector(model, normalize_for_detector(x01),
                                    args.fpn_level)
        _, s = preds_to_xyxy_scores(preds)
    return s


# ══════════════════════════════════════════════════════════════════
#  메인  (SCGA train_BIA_ours.py 골격 그대로)
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
#  한 스텝
# ══════════════════════════════════════════════════════════════════

def train_step(batch, student, teacher, detector, optimG, args, eps, device):
    """
    학습 한 스텝. 반환값은 float 뿐이라 그래프 참조가 함수 밖으로 새지 않는다.

    ★ 왜 함수로 뺐나 — OOM 때문이다.

        루프 본문에 그대로 두면 adv_t / fs_t / d_t / box_t 같은 그래프 참조가
        **다음 반복이 새 forward 를 다 만들 때까지 살아 있다.**
        파이썬은 재대입 시점에 이전 값을 버리는데, 그 시점이 새 activation 을
        이미 다 할당한 뒤다. 즉 한때 두 스텝분 활성값이 동시에 잡힌다.

        더 나쁜 것은 공통 ID 부족으로 건너뛰는 경로다. backward 를 부르지 않으므로
        그래프가 통째로 살아남아 다음 스텝까지 넘어간다.

        실측: batch 4 / 800x1440 에서 950 스텝에 94.7GB 로 터졌다.
        함수로 빼면 return 시점에 지역 변수가 전부 사라져 한 스텝분만 잡힌다.

    Returns:
        dict — 손실 값들. 공통 ID 가 부족해 건너뛰면 None.
    """
    img_t = batch['img_t'].to(device, non_blocking=True)
    img_u = batch['img_u'].to(device, non_blocking=True)

    student.train()
    optimG.zero_grad(set_to_none=True)

    # 1) teacher forward (clean path)
    with torch.no_grad():
        _, ft_t = teacher(img_t, feat=True)
        _, ft_u = teacher(img_u, feat=True)

    # 2) student forward (adversarial)
    adv_t, fs_t = student(img_t, feat=True)
    adv_u, fs_u = student(img_u, feat=True)

    # 3) eps 예산 적용
    adv_t = apply_perturbation(adv_t, img_t, eps, args.delta_param, args.zero_dc)
    adv_u = apply_perturbation(adv_u, img_u, eps, args.delta_param, args.zero_dc)

    # 4) SCGA 일관성 손실
    gen_sim = (generator_consistency_loss(ft_t, fs_t)
               + generator_consistency_loss(ft_u, fs_u))

    # 5) 공격 손실
    ft = frame_descriptor(detector, adv_t, batch['boxes_t'].to(device),
                          batch['ids_t'].to(device), args)
    fu = frame_descriptor(detector, adv_u, batch['boxes_u'].to(device),
                          batch['ids_u'].to(device), args)
    ids_t, ids_u = ft['ids'], fu['ids']

    # 두 프레임에 공통 ID 가 없으면 공격 손실을 계산할 수 없다.
    # (데이터셋에서 공통 ID 2개 이상인 쌍만 뽑았으므로 드물지만,
    #  매칭 실패로 남지 않을 수 있다)
    common = set(ids_t.tolist()) & set(ids_u.tolist())
    if len(common) < 2:
        return None

    l_att, parts = attack_loss(ft['d'], fu['d'], ids_t, ids_u, lam=args.lam,
                               center=not args.no_center)

    # 6) 합산 + 갱신
    loss = l_att + args.w_consistency * gen_sim
    if args.w_iou > 0:
        l_iou = temporal_iou_loss(ft['box'], fu['box'], ids_t, ids_u)
        loss = loss + args.w_iou * l_iou
        parts['iou'] = l_iou.item()

    # L_keep — 검출을 클린 상태로 붙잡아 둔다.
    # 이것이 없으면 최적화가 백본 feature 를 부수는 지름길을 택한다 (Exp-002/003).
    if args.w_keep > 0:
        l_keep = 0.0
        acc = {'dscore': 0.0, 'bgup': 0.0, 'keepiou': 0.0}
        for f, x_clean in ((ft, img_t), (fu, img_u)):
            lk, kp = detection_keep_loss(f['scores'],
                                         clean_scores(detector, x_clean, args),
                                         f['box'], f['gt'], tau=args.keep_tau)
            l_keep = l_keep + lk
            for k in acc:
                acc[k] += kp[k] / 2
        loss = loss + args.w_keep * l_keep
        parts.update(acc)

    loss.backward()
    optimG.step()
    update_ema(student, teacher, momentum=args.ema_momentum)

    parts['loss'] = loss.item()
    parts['gen'] = gen_sim.item()
    return parts


# ══════════════════════════════════════════════════════════════════
#  체크포인트
# ══════════════════════════════════════════════════════════════════

def save_ckpt(path, student, teacher, optimG, epoch, gstep, args):
    """중단된 학습을 이어받을 수 있도록 전부 담는다."""
    torch.save({'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optim': optimG.state_dict(),
                'epoch': epoch, 'gstep': gstep,
                'args': vars(args)}, path)


def main():
    args = get_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # ── 출력 폴더 + 스크립트 백업 (SCGA 그대로) ────────────────
    now = datetime.datetime.now().strftime("%Y%m%d_T%H%M%S")[2:]
    args.output_dir = f'{args.output_dir}/{now}_{args.method}_{args.generator}'
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(__file__, os.path.join(args.output_dir, os.path.basename(__file__)))
    print(vars(args))

    # ── 검출기 (surrogate) ──────────────────────────────────────
    detector = build_detector(args, device)

    # ── generator: student / teacher (SCGA 그대로) ──────────────
    # 주의: SCGA 원본은 deepcopy 후 student/teacher 를 각각 초기화해
    #       서로 다른 난수로 시작한다(DESIGN.md 7.5 A5 참고).
    #       여기서는 초기화 -> deepcopy 순서로 바로잡는다.
    # GeneratorWithFeatures 로 감싸는 이유는 attack/generator_wrap.py 주석 참고
    # (공개된 generator 는 블록별 feature 리스트를 반환하지 않는다)
    student = GeneratorWithFeatures(
        build_generator(generator=args.generator, gap=args.gap,
                        inception=False, nat=args.nat)).to(device)
    student.apply(weights_init_normal)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False

    optimG = optim.Adam(student.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # ── 이어받기 ────────────────────────────────────────────────
    # 에폭 단위로만 이어받는다. DataLoader 의 셔플 상태까지 복원하지는 않으므로
    # 에폭 중간에서 끊긴 경우 그 에폭은 처음부터 다시 돈다.
    start_epoch, gstep = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        student.load_state_dict(ck['student'])
        teacher.load_state_dict(ck['teacher'])
        optimG.load_state_dict(ck['optim'])
        start_epoch, gstep = ck['epoch'] + 1, ck['gstep']
        print(f'이어받기: {args.resume} | epoch {start_epoch} 부터, '
              f'누적 {gstep} 스텝', flush=True)

    # ── 데이터 ──────────────────────────────────────────────────
    train_set = MOT17PairDataset(
        data_root=args.data_root,
        split='train',
        detector=args.detector_variant,
        half='first',
        input_size=(args.input_h, args.input_w),
        frame_gaps=tuple(args.frame_gaps),
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=pair_collate,
    )
    print(f'train pairs: {len(train_set)}')

    eps = args.eps / 255.0

    # ── 손실 기록 ───────────────────────────────────────────────
    # nohup 으로 파일에 남길 때 tqdm 은 \r 로 덮어써서 읽을 수 없다.
    # 터미널이 아니면 tqdm 을 끄고 log_every 마다 평문 한 줄만 남긴다.
    is_tty = sys.stderr.isatty()
    csv_path = os.path.join(args.output_dir, 'loss_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_w = csv.writer(csv_file)
    csv_w.writerow(['step', 'epoch', 'loss', 'decorr', 'confuse', 'iou',
                    'dscore', 'bgup', 'keepiou', 'gen', 'skip',
                    'elapsed_s', 'vram_gb'])
    t_start = time.time()
    total_steps = len(train_loader) * args.epochs
    print(f'총 {total_steps} 스텝 | 배치 {args.batch_size} | 로그 {csv_path}',
          flush=True)

    # ── 학습 루프 (SCGA 그대로) ────────────────────────────────
    n_skip = 0
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', ncols=90,
                    disable=not is_tty)
        for it, batch in enumerate(pbar):
            parts = train_step(batch, student, teacher, detector,
                               optimG, args, eps, device)
            if parts is None:
                n_skip += 1
                continue

            post = dict(loss=f'{parts["loss"]:.3f}',
                        dec=f'{parts["decorr"]:.3f}',
                        con=f'{parts["confuse"]:.3f}',
                        gen=f'{parts["gen"]:.3f}',
                        skip=n_skip)
            if 'iou' in parts:
                post['iou'] = f'{parts["iou"]:.3f}'
            pbar.set_postfix(**post)

            # 7) 기록
            gstep += 1
            if gstep % args.log_every == 0 or gstep == 1:
                el = time.time() - t_start
                peak = (torch.cuda.max_memory_allocated() / 1024**3
                        if device.type == 'cuda' else 0.0)
                nan = float('nan')
                csv_w.writerow([gstep, epoch + 1, round(parts['loss'], 5),
                                round(parts['decorr'], 5),
                                round(parts['confuse'], 5),
                                round(parts.get('iou', nan), 5),
                                round(parts.get('dscore', nan), 5),
                                round(parts.get('bgup', nan), 5),
                                round(parts.get('keepiou', nan), 5),
                                round(parts['gen'], 5), n_skip,
                                round(el, 1), round(peak, 2)])
                csv_file.flush()
                if not is_tty:
                    eta = el / gstep * (total_steps - gstep) / 60
                    line = (f'[{epoch+1}/{args.epochs}] {gstep}/{total_steps} '
                            f'loss={parts["loss"]:.4f} dec={parts["decorr"]:.4f} '
                            f'con={parts["confuse"]:.4f}')
                    if 'iou' in parts:
                        line += f' iou={parts["iou"]:.4f}'
                    if 'dscore' in parts:
                        # 검출 건강도. dscore/bgup 이 커지거나 keepiou 가 떨어지면
                        # 연관이 아니라 검출을 부수고 있는 것이다
                        line += (f' dscore={parts["dscore"]:.4f}'
                                 f' bgup={parts["bgup"]:.4f}'
                                 f' keepiou={parts["keepiou"]:.4f}')
                    print(f'{line} gen={parts["gen"]:.4f} skip={n_skip} '
                          f'| VRAM {peak:.1f}GB | {el/60:.1f}분 경과, '
                          f'남은 {eta:.0f}분', flush=True)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

            # 8) 저장 (SCGA 는 student 가 아니라 teacher 를 저장한다)
            if it > 0 and it % args.save_freq == 0:
                torch.save(teacher.state_dict(),
                           os.path.join(args.output_dir,
                                        f'teacher_{args.method}_{epoch}.pth'))

        # 에폭 끝 — 중단되어도 이어받을 수 있게 전체 상태를 남긴다
        save_ckpt(os.path.join(args.output_dir, 'ckpt_last.pth'),
                  student, teacher, optimG, epoch, gstep, args)

    csv_file.close()
    torch.save(teacher.state_dict(),
               os.path.join(args.output_dir, f'teacher_{args.method}_final.pth'))
    print(f'saved: {args.output_dir}  ({(time.time()-t_start)/60:.1f}분)', flush=True)


if __name__ == '__main__':
    main()
