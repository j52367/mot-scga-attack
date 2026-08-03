# -*- coding: utf-8 -*-
"""
객체 서술자 (object descriptor)

검출기 백본의 중간 feature와 예측 박스로부터, 트래커 구조에 의존하지 않는
객체 단위 서술자를 만든다. 이 서술자의 시간적 일관성을 교란하는 것이 본 공격의 목표.

  d_k = [ L2norm(GAP(ROIAlign(F, b_k))) ;  alpha * (cx, cy, w, h) ]

- 외형 성분 f_k : ReID 헤드가 있든 없든 모든 검출기의 백본에 존재
- 기하 성분 g_k : ByteTrack처럼 IoU만 쓰는 트래커에 닿기 위해 필요

주의: boxes 는 반드시 검출기가 예측한 '미분 가능한 텐서'여야 한다.
      numpy 로 변환된 뒤에는 gradient 가 흐르지 않는다.
"""

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


def object_descriptor(feat_map, boxes, img_hw, alpha=1.0, out_size=7):
    """
    Args:
        feat_map : [B, C, Hf, Wf]  백본 중간 feature
        boxes    : [K, 5]  (batch_idx, x1, y1, x2, y2), 원본 이미지 좌표계, 미분 가능
        img_hw   : (H, W)  원본 입력 크기
        alpha    : 기하 성분 가중치
    Returns:
        d : [K, C + 4]
    """
    H, W = img_hw
    spatial_scale = feat_map.shape[-1] / float(W)

    # ── 외형 성분 ────────────────────────────────────────────────
    pooled = roi_align(
        feat_map, boxes,
        output_size=out_size,
        spatial_scale=spatial_scale,
        aligned=True,
    )                                   # [K, C, out, out]
    f = pooled.mean(dim=(2, 3))         # GAP -> [K, C]
    f = F.normalize(f, p=2, dim=1)

    # ── 기하 성분 ────────────────────────────────────────────────
    x1, y1, x2, y2 = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    cx = ((x1 + x2) * 0.5) / W
    cy = ((y1 + y2) * 0.5) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    g = torch.stack([cx, cy, bw, bh], dim=1)    # [K, 4]

    return torch.cat([f, alpha * g], dim=1)


def multi_level_descriptor(feat_maps, boxes, img_hw, alpha=1.0, out_size=7):
    """
    FPN 여러 레벨을 함께 쓰는 변형. 층 선택은 ablation 대상이므로
    단일 레벨(object_descriptor)로 먼저 실험한 뒤 필요하면 사용한다.
    """
    ds = [object_descriptor(fm, boxes, img_hw, alpha=0.0, out_size=out_size)[:, :-4]
          for fm in feat_maps]
    f = F.normalize(torch.cat(ds, dim=1), p=2, dim=1)

    H, W = img_hw
    x1, y1, x2, y2 = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    g = torch.stack([((x1 + x2) * 0.5) / W,
                     ((y1 + y2) * 0.5) / H,
                     (x2 - x1) / W,
                     (y2 - y1) / H], dim=1)
    return torch.cat([f, alpha * g], dim=1)
