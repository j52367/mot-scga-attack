# -*- coding: utf-8 -*-
"""
MOT17 프레임 쌍 데이터셋

generator 학습에 필요한 것: 연속된 두 프레임 + 각 프레임의 박스 + ID
ID 는 "같은 객체"를 정의하기 위해 필요하며 학습 시점에만 쓰인다(공격 시에는 불필요).

전처리는 ByteTrack 의 preproc 과 동일하게 맞춘다.
    BGR -> letterbox(114 패딩, 좌상단 정렬) -> RGB -> /255
반환값은 [0,1] 공간이며, mean/std 정규화는 학습 루프에서 검출기 입력 직전에 적용한다.
(generator 와 eps 투영이 [0,1] 공간에서 동작해야 하기 때문. SCGA 와 동일한 구조)

MOT17 gt.txt 형식:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    - conf  : 0 이면 무시 대상
    - class : 1 = pedestrian
"""

import configparser
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ── gt 파싱 ──────────────────────────────────────────────────────────

def load_gt(gt_path, min_visibility=0.3):
    """gt.txt -> {frame_id: [(track_id, x1, y1, x2, y2), ...]}"""
    per_frame = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 9:
                continue
            frame, tid = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            conf, cls, vis = float(p[6]), int(float(p[7])), float(p[8])
            if conf < 1 or cls != 1 or vis < min_visibility:
                continue
            per_frame[frame].append((tid, x, y, x + w, y + h))
    return per_frame


def read_seq_length(seq_dir):
    ini = os.path.join(seq_dir, 'seqinfo.ini')
    cfg = configparser.ConfigParser()
    cfg.read(ini)
    return int(cfg['Sequence']['seqLength'])


# ── 전처리 (ByteTrack preproc 과 동일) ───────────────────────────────

def letterbox(img_bgr, input_size):
    """
    Returns:
        img : [3, H, W] float32, RGB, [0,1]
        r   : 박스 좌표에 곱할 배율
    """
    ih, iw = img_bgr.shape[:2]
    r = min(input_size[0] / ih, input_size[1] / iw)
    nh, nw = int(ih * r), int(iw * r)

    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.float32) * 114.0
    padded[:nh, :nw] = cv2.resize(img_bgr, (nw, nh),
                                  interpolation=cv2.INTER_LINEAR).astype(np.float32)

    padded = padded[:, :, ::-1]          # BGR -> RGB
    padded = padded / 255.0              # [0,1]  <- 노이즈는 이 공간에서 더해진다
    img = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return img, r


# ── 데이터셋 ─────────────────────────────────────────────────────────

class MOT17PairDataset(Dataset):
    """
    한 샘플 = (프레임 t, 프레임 t+gap) 와 각각의 박스·ID

    Args:
        data_root      : MOT17 루트 (예: .../MOT17)
        split          : 'train'
        detector       : 'FRCNN' | 'DPM' | 'SDP'  (이미지·GT 동일, 하나만 쓰면 됨)
        half           : None | 'first' | 'second'
                         ByteTrack 관례대로 시퀀스를 반으로 갈라 train/val 로 사용
        input_size     : (H, W)
        frame_gaps     : 프레임 간격 후보
        min_common_ids : 두 프레임에 공통으로 있는 ID 최소 개수
                         L_confuse 를 계산하려면 2 이상이어야 한다
    """

    def __init__(self,
                 data_root,
                 split='train',
                 detector='FRCNN',
                 half=None,
                 input_size=(800, 1440),
                 frame_gaps=(1, 2, 3),
                 min_visibility=0.3,
                 min_common_ids=2):
        self.input_size = input_size
        self.samples = []          # (seq_dir, frame_a, frame_b)
        self.gt_cache = {}         # seq_dir -> per_frame dict

        split_dir = os.path.join(data_root, split)
        seqs = sorted(s for s in os.listdir(split_dir) if s.endswith(detector))
        if not seqs:
            raise FileNotFoundError(
                f'{split_dir} 에서 {detector} 시퀀스를 찾지 못했습니다.')

        for seq in seqs:
            seq_dir = os.path.join(split_dir, seq)
            gt = load_gt(os.path.join(seq_dir, 'gt', 'gt.txt'), min_visibility)
            self.gt_cache[seq_dir] = gt

            n = read_seq_length(seq_dir)
            lo, hi = 1, n
            if half == 'first':
                hi = n // 2
            elif half == 'second':
                lo = n // 2 + 1

            for t in range(lo, hi + 1):
                for gap in frame_gaps:
                    u = t + gap
                    if u > hi:
                        continue
                    ids_t = {b[0] for b in gt.get(t, [])}
                    ids_u = {b[0] for b in gt.get(u, [])}
                    if len(ids_t & ids_u) >= min_common_ids:
                        self.samples.append((seq_dir, t, u))

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, seq_dir, frame):
        path = os.path.join(seq_dir, 'img1', f'{frame:06d}.jpg')
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(path)
        img, r = letterbox(img_bgr, self.input_size)

        anns = self.gt_cache[seq_dir].get(frame, [])
        ids = np.array([a[0] for a in anns], dtype=np.int64)
        boxes = np.array([a[1:] for a in anns], dtype=np.float32).reshape(-1, 4) * r
        return torch.from_numpy(img), torch.from_numpy(boxes), torch.from_numpy(ids)

    def __getitem__(self, i):
        seq_dir, t, u = self.samples[i]
        img_t, box_t, id_t = self._load_frame(seq_dir, t)
        img_u, box_u, id_u = self._load_frame(seq_dir, u)
        return {
            'img_t': img_t, 'img_u': img_u,
            'boxes_t': box_t, 'boxes_u': box_u,
            'ids_t': id_t, 'ids_u': id_u,
        }


# ── collate ──────────────────────────────────────────────────────────

def pair_collate(batch):
    """
    박스 개수가 샘플마다 달라 기본 collate 를 쓸 수 없다.
    박스는 roi_align 이 요구하는 [K, 5] = (batch_idx, x1, y1, x2, y2) 형태로 만든다.
    """
    out = {
        'img_t': torch.stack([b['img_t'] for b in batch]),
        'img_u': torch.stack([b['img_u'] for b in batch]),
    }
    for side in ('t', 'u'):
        boxes, ids = [], []
        for bi, b in enumerate(batch):
            bx = b[f'boxes_{side}']
            if bx.numel() == 0:
                continue
            idx = torch.full((bx.shape[0], 1), float(bi))
            boxes.append(torch.cat([idx, bx], dim=1))
            ids.append(b[f'ids_{side}'])
        out[f'boxes_{side}'] = (torch.cat(boxes) if boxes
                                else torch.zeros(0, 5))
        out[f'ids_{side}'] = (torch.cat(ids) if ids
                              else torch.zeros(0, dtype=torch.int64))
    return out


# ── 검출기 입력용 정규화 ─────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_for_detector(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    [0,1] 이미지를 ByteTrack 검출기가 기대하는 입력으로 변환.
    ByteTrack 의 ValTransform(rgb_means=..., std=...) 과 동일.
    in-place 가 아니므로 원본이 오염되지 않는다.
    """
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s
