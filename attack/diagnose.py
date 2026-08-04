# -*- coding: utf-8 -*-
"""
공격 진단 (EXPERIMENTS.md 가설 1 / 4 확인용)

Exp-001 에서 공격이 추적을 오히려 개선했다(IDF1 52.7 -> 60.9).
학습을 다시 돌리기 전에 '무엇이 잘못됐는지'를 먼저 특정한다.

같은 프레임을 클린/공격 두 경로로 돌려 네 가지를 한 번에 본다.

    A-1  노이즈 자체가 유의미한가        (가설 1)
         - 절대값, +-eps 포화율
         - ★ 상수 성분 vs 잔차 표준편차 : delta 가 거의 균일하면
           공격이 아니라 '전역 톤 이동'을 배운 것이고, 그것이 정규화처럼
           작용해 IDF1 이 올랐다는 설명이 그대로 성립한다
         - 레터박스 패딩 영역에 예산을 낭비했는가

    A-2  눈으로 확인
         클린 / 공격 / delta 증폭 이미지를 저장

    A-3  검출이 어떻게 변했는가          (가설 4)
         ByteTrack 의 2단계 매칭 기준으로 나눠서 센다.
             score >= track_thresh   1차 매칭 대상 (high)
             0.1 <= score < thresh   2차 매칭 대상 (low)  <- 여기가 핵심
         Precision 상승 / Recall 하락 패턴이 사실이라면 low 가 줄었을 것이다.

    B    손실이 평가 구간에서도 작동하는가   (진단표에 없던 항목)
         학습 손실은 train half 에서 잰 값이다. 평가는 second half 다.
         평가 구간 프레임 쌍에서 중심화된 코사인을 클린/공격 각각 재서
         '학습 실패'와 '손실-실효 불일치'를 가른다.

실행
    PYTHONPATH=. python attack/diagnose.py \
        --attack_ckpt experiments/260804_T011034_ptca_resnet/teacher_ptca_final.pth
"""

import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch

import attack.compat  # noqa: F401  yolox import 보다 먼저

from yolox.utils import postprocess

from attack.dataset import (MOT17PairDataset, letterbox, normalize_for_detector,
                            pair_collate, read_seq_length)
from attack.evaluate import load_attacker
from attack.losses import center_pair, project, _pair_by_id
from attack.train_generator import build_detector, frame_descriptor


# ══════════════════════════════════════════════════════════════════
#  A-1  노이즈 통계
# ══════════════════════════════════════════════════════════════════

class NoiseStats:
    """프레임마다 누적하고 마지막에 평균을 낸다."""

    def __init__(self):
        self.absmax, self.absmean, self.sat = [], [], []
        self.const, self.resid_std = [], []
        self.pad_ratio = []

    def add(self, delta, eps, valid_hw=None):
        """
        delta : [1,3,H,W]  (adv - img, eps 투영 이후)
        valid_hw : (nh, nw) 레터박스에서 실제 영상이 있는 영역
        """
        d = delta[0]                                   # [3,H,W]
        self.absmax.append(d.abs().max().item())
        self.absmean.append(d.abs().mean().item())
        self.sat.append((d.abs() >= 0.99 * eps).float().mean().item())

        # 상수 성분(채널별 평균) 과 잔차로 분해
        c = d.mean(dim=(1, 2))                         # [3]
        r = d - c.view(3, 1, 1)
        self.const.append(c.abs().mean().item())
        self.resid_std.append(r.std().item())

        if valid_hw is not None:
            nh, nw = valid_hw
            total = d.abs().sum().item()
            inside = d[:, :nh, :nw].abs().sum().item()
            self.pad_ratio.append(1.0 - inside / max(total, 1e-12))

    def report(self, eps):
        m = lambda v: float(np.mean(v)) if v else float('nan')  # noqa: E731
        c, rs = m(self.const), m(self.resid_std)
        print('  [A-1] 노이즈 통계')
        print(f'    최대 |delta|      {m(self.absmax):.4f}   (eps = {eps:.4f})')
        print(f'    평균 |delta|      {m(self.absmean):.4f}   '
              f'(eps 대비 {m(self.absmean)/eps*100:.1f}%)')
        print(f'    +-eps 포화 비율   {m(self.sat)*100:.1f}%')
        print(f'    상수 성분 |c|     {c:.4f}      <- 전역 톤 이동 크기')
        print(f'    잔차 표준편차     {rs:.4f}      <- 공간 구조 크기')
        print(f'    구조/상수 비      {rs/max(c,1e-12):.2f}   '
              f'(1 미만이면 사실상 톤 이동)')
        if self.pad_ratio:
            print(f'    패딩 영역 낭비    {m(self.pad_ratio)*100:.1f}% '
                  f'(예산이 회색 패딩에 쓰인 비율)')
        return {'absmean': m(self.absmean), 'const': c, 'resid_std': rs}


# ══════════════════════════════════════════════════════════════════
#  A-3  검출 통계
# ══════════════════════════════════════════════════════════════════

class DetStats:
    def __init__(self, track_thresh):
        self.track_thresh = track_thresh
        self.n_all, self.n_high, self.n_low = [], [], []
        self.scores = []

    def add(self, out):
        """out : postprocess 결과 [N,7] 또는 None"""
        if out is None or len(out) == 0:
            self.n_all.append(0); self.n_high.append(0); self.n_low.append(0)
            return
        s = (out[:, 4] * out[:, 5]).detach().cpu().numpy()
        self.n_all.append(len(s))
        self.n_high.append(int((s >= self.track_thresh).sum()))
        self.n_low.append(int(((s >= 0.1) & (s < self.track_thresh)).sum()))
        self.scores.append(s)

    def summary(self):
        s = np.concatenate(self.scores) if self.scores else np.zeros(1)
        return {
            'all': float(np.mean(self.n_all)),
            'high': float(np.mean(self.n_high)),
            'low': float(np.mean(self.n_low)),
            'p50': float(np.percentile(s, 50)),
            'p90': float(np.percentile(s, 90)),
            'p99': float(np.percentile(s, 99)),
        }


def report_det(clean, adv, track_thresh):
    c, a = clean.summary(), adv.summary()
    print('  [A-3] 검출 분포 (프레임당 평균)')
    print(f'    {"":16s} {"클린":>10s} {"공격":>10s} {"차이":>10s}')
    rows = [
        ('전체 검출 수', 'all'),
        (f'high (>={track_thresh})', 'high'),
        ('low  (0.1~thr)', 'low'),
        ('점수 중앙값', 'p50'),
        ('점수 90분위', 'p90'),
        ('점수 99분위', 'p99'),
    ]
    for label, k in rows:
        d = a[k] - c[k]
        print(f'    {label:16s} {c[k]:10.2f} {a[k]:10.2f} {d:+10.2f}')
    return c, a


# ══════════════════════════════════════════════════════════════════
#  A-2  이미지 덤프
# ══════════════════════════════════════════════════════════════════

def to_bgr(x01):
    """[1,3,H,W] RGB [0,1] -> BGR uint8"""
    img = x01[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img[:, :, ::-1].copy()


def dump_images(out_dir, frame_id, x, adv, eps):
    delta = adv - x
    # delta 를 [-eps, eps] -> [0, 255] 로 펼쳐 증폭
    amp = (delta[0].detach().cpu().numpy().transpose(1, 2, 0) / (2 * eps) + 0.5)
    amp = np.clip(amp * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1].copy()

    cv2.imwrite(osp.join(out_dir, f'{frame_id:06d}_1clean.jpg'), to_bgr(x))
    cv2.imwrite(osp.join(out_dir, f'{frame_id:06d}_2adv.jpg'), to_bgr(adv))
    cv2.imwrite(osp.join(out_dir, f'{frame_id:06d}_3noise.jpg'), amp)


# ══════════════════════════════════════════════════════════════════
#  B  서술자 코사인 (평가 구간)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def descriptor_cosines(d_t, d_u, ids_t, ids_u, center=True):
    """
    Returns:
        same : 같은 ID 의 프레임 간 코사인 평균  (낮을수록 공격 성공)
        diff : 다른 ID 의 프레임 간 코사인 평균  (높을수록 공격 성공)

    손실이 아니라 '원래 부호의 코사인'을 돌려준다. 부호 혼동을 피하기 위해서다.
    (L_confuse = -diff 이므로 손실값과 헷갈리기 쉽다)
    """
    if d_t.numel() == 0 or d_u.numel() == 0:
        return None, None
    i_t, i_u = _pair_by_id(ids_t, ids_u)
    if i_t is None or len(i_t) < 2:
        return None, None

    if center:
        d_t, d_u = center_pair(d_t, d_u)
    a = torch.nn.functional.normalize(d_t, p=2, dim=1)
    b = torch.nn.functional.normalize(d_u, p=2, dim=1)

    same = (a[i_t] * b[i_u]).sum(dim=1).mean().item()
    sim = a @ b.t()
    mask = ids_t.view(-1, 1) == ids_u.view(1, -1)
    diff = sim[~mask].mean().item() if (~mask).any() else float('nan')
    return same, diff


@torch.no_grad()
def run_descriptor_check(args, model, attacker, device):
    ds = MOT17PairDataset(
        data_root=args.pair_root, split='train',
        detector=args.detector_variant, half=args.half,
        input_size=(args.input_h, args.input_w),
        frame_gaps=tuple(args.frame_gaps))
    # 진단 대상 시퀀스만 남긴다
    ds.samples = [s for s in ds.samples if args.seq in s[0]]
    if not ds.samples:
        print(f'  [B] {args.seq} 에 해당하는 프레임 쌍이 없다 — 건너뜀')
        return None

    # 한 구간에 몰리지 않도록 고르게 뽑는다
    step = max(1, len(ds.samples) // args.n_pairs)
    picks = list(range(0, len(ds.samples), step))[:args.n_pairs]
    eps = args.eps / 255.0

    acc = {'clean': ([], []), 'attack': ([], [])}
    n_used = 0

    for idx in picks:
        batch = pair_collate([ds[idx]])
        img_t = batch['img_t'].to(device)
        img_u = batch['img_u'].to(device)
        bt, bu = batch['boxes_t'].to(device), batch['boxes_u'].to(device)
        it, iu = batch['ids_t'].to(device), batch['ids_u'].to(device)

        for mode, (xt, xu) in (
                ('clean', (img_t, img_u)),
                ('attack', (project(_g(attacker, img_t), img_t, eps),
                            project(_g(attacker, img_u), img_u, eps)))):
            d_t, ids_t = frame_descriptor(model, xt, bt, it, args)
            d_u, ids_u = frame_descriptor(model, xu, bu, iu, args)
            same, diff = descriptor_cosines(d_t, d_u, ids_t, ids_u,
                                            center=not args.no_center)
            if same is not None:
                acc[mode][0].append(same)
                acc[mode][1].append(diff)
        n_used += 1

    if not acc['clean'][0]:
        print('  [B] 유효한 쌍이 없다 — 건너뜀')
        return None

    m = lambda v: float(np.mean(v))  # noqa: E731
    cs, cd = m(acc['clean'][0]), m(acc['clean'][1])
    as_, ad = m(acc['attack'][0]), m(acc['attack'][1])

    print(f'  [B] 서술자 코사인 (평가 구간 {n_used}쌍, 중심화 '
          f'{"on" if not args.no_center else "off"})')
    print(f'    {"":16s} {"클린":>10s} {"공격":>10s} {"차이":>10s}')
    print(f'    {"같은 ID":16s} {cs:10.4f} {as_:10.4f} {as_-cs:+10.4f}   '
          f'<- 내려가야 성공')
    print(f'    {"다른 ID":16s} {cd:10.4f} {ad:10.4f} {ad-cd:+10.4f}   '
          f'<- 올라가야 성공')
    print(f'    {"판별 격차":16s} {cs-cd:10.4f} {as_-ad:10.4f} '
          f'{(as_-ad)-(cs-cd):+10.4f}   <- 줄어야 성공')
    return {'same_clean': cs, 'same_attack': as_,
            'diff_clean': cd, 'diff_attack': ad}


def _g(attacker, x):
    out = attacker(x)
    return out[0] if isinstance(out, tuple) else out


# ══════════════════════════════════════════════════════════════════
#  A  프레임 순차 진단
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_frames(args, model, attacker, device, dump_dir):
    seq_dir = osp.join(args.data_root, args.seq)
    n_frames = read_seq_length(seq_dir)
    lo, hi = 1, n_frames
    if args.half == 'first':
        hi = n_frames // 2
    elif args.half == 'second':
        lo = n_frames // 2 + 1
    hi = min(hi, lo + args.n_frames - 1)

    eps = args.eps / 255.0
    ns = NoiseStats()
    det_clean = DetStats(args.track_thresh)
    det_adv = DetStats(args.track_thresh)

    for frame_id in range(lo, hi + 1):
        img_bgr = cv2.imread(osp.join(seq_dir, 'img1', f'{frame_id:06d}.jpg'))
        ih, iw = img_bgr.shape[:2]
        img, r = letterbox(img_bgr, (args.input_h, args.input_w))
        x = torch.from_numpy(img).unsqueeze(0).to(device)

        adv = project(_g(attacker, x), x, eps)
        ns.add(adv - x, eps, valid_hw=(int(ih * r), int(iw * r)))

        for xx, st in ((x, det_clean), (adv, det_adv)):
            out = model(normalize_for_detector(xx))
            out = postprocess(out, args.num_classes,
                              args.conf_thre, args.nms_thre)[0]
            st.add(out)

        if frame_id - lo < args.n_dump:
            dump_images(dump_dir, frame_id, x, adv, eps)

    print(f'  프레임 {lo}~{hi} ({hi-lo+1}장)\n')
    noise = ns.report(eps)
    print()
    c, a = report_det(det_clean, det_adv, args.track_thresh)
    return noise, c, a


# ══════════════════════════════════════════════════════════════════
#  판정
# ══════════════════════════════════════════════════════════════════

def verdict(noise, det_clean, det_adv, desc, eps):
    print('\n' + '=' * 66)
    print('  판정')
    print('=' * 66)
    lines = []

    # 가설 1
    ratio = noise['resid_std'] / max(noise['const'], 1e-12)
    if noise['absmean'] < 0.1 * eps:
        lines.append('가설 1 성립 — 노이즈가 eps 예산의 10% 도 쓰지 않는다. '
                     'generator 가 사실상 학습되지 않았다.')
    elif ratio < 1.0:
        lines.append('가설 1 성립 — delta 에 공간 구조가 거의 없다(전역 톤 이동). '
                     '공격이 아니라 색보정을 배운 것이며, 이것이 IDF1 상승을 설명한다.')
    else:
        lines.append('가설 1 기각 — delta 에 공간 구조가 있다. 노이즈 자체는 유의미하다.')

    # 가설 4
    dl = det_adv['low'] - det_clean['low']
    dh = det_adv['high'] - det_clean['high']
    if dl < -0.1 * max(det_clean['low'], 1e-9) and dh >= -0.05 * max(det_clean['high'], 1e-9):
        lines.append(f'가설 4 성립 — low 검출이 {dl:+.1f} 줄고 high 는 {dh:+.1f} 로 유지됐다. '
                     '공격이 2차 매칭 후보를 청소해 ByteTrack 을 도와준 것이다.')
    elif dh < -0.1 * max(det_clean['high'], 1e-9):
        lines.append(f'가설 4 기각 — high 검출이 {dh:+.1f} 로 실제 감소했다. '
                     '검출은 무너뜨렸으나 추적이 버틴 경우다.')
    else:
        lines.append('가설 4 판단 보류 — 검출 분포가 거의 변하지 않았다.')

    # B
    if desc is not None:
        d_same = desc['same_attack'] - desc['same_clean']
        if abs(d_same) < 0.02:
            lines.append(f'★ 평가 구간에서 같은-ID 코사인이 {d_same:+.4f} 로 거의 변하지 않았다. '
                         '학습된 공격이 평가 구간에 일반화되지 않았다 -> 가설 2(학습 부족)로.')
        elif d_same < -0.05:
            lines.append(f'★ 같은-ID 코사인이 {d_same:+.4f} 로 실제 내려갔다. '
                         '서술자는 무너뜨렸는데 추적은 멀쩡하다 -> 가설 3(손실-실효 불일치) 확정. '
                         '이 자체가 논문에 쓸 수 있는 발견이다.')
        else:
            lines.append(f'★ 같은-ID 코사인이 {d_same:+.4f} 로 오히려 올라갔다. '
                         '손실 부호나 최적화 방향을 다시 볼 것 -> 가설 5.')

    for i, s in enumerate(lines, 1):
        print(f'  {i}. {s}')
    print()


# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    # 데이터
    ap.add_argument('--data_root', default='datasets/mot/train')
    ap.add_argument('--pair_root', default='datasets/mot',
                    help='MOT17PairDataset 용 루트 (내부에서 split 을 붙인다)')
    ap.add_argument('--seq', default='MOT17-02-FRCNN')
    ap.add_argument('--half', default='second', choices=['first', 'second', 'all'])
    ap.add_argument('--detector_variant', default='FRCNN')
    ap.add_argument('--frame_gaps', type=int, nargs='+', default=[1, 2, 3])
    # 검출기
    ap.add_argument('--exp_file', default='exps/example/mot/yolox_s_mix_det.py')
    ap.add_argument('--ckpt', default='pretrained/bytetrack_s_mot17.pth.tar')
    ap.add_argument('--input_h', type=int, default=800)
    ap.add_argument('--input_w', type=int, default=1440)
    ap.add_argument('--conf_thre', type=float, default=0.001)
    ap.add_argument('--nms_thre', type=float, default=0.7)
    ap.add_argument('--num_classes', type=int, default=1)
    ap.add_argument('--track_thresh', type=float, default=0.6,
                    help='ByteTrack 1차 매칭 임계값. 이 위가 high, 0.1~이 값이 low')
    # 공격 (학습 때와 같은 값을 줘야 한다)
    ap.add_argument('--attack_ckpt', required=True)
    ap.add_argument('--eps', type=float, default=10)
    ap.add_argument('--generator', default='resnet')
    ap.add_argument('--gap', action='store_true')
    ap.add_argument('--nat', action='store_true')
    # 서술자 (학습 때와 같은 값)
    ap.add_argument('--fpn_level', type=int, default=1)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--iou_thr', type=float, default=0.1)
    ap.add_argument('--roi_source', default='pred', choices=['pred', 'gt'])
    ap.add_argument('--no_center', action='store_true')
    # 진단 규모
    ap.add_argument('--n_frames', type=int, default=50, help='A부 프레임 수')
    ap.add_argument('--n_pairs', type=int, default=30, help='B부 프레임 쌍 수')
    ap.add_argument('--n_dump', type=int, default=3, help='저장할 이미지 세트 수')
    ap.add_argument('--out', default='output/diagnose')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--skip_desc', action='store_true', help='B부 건너뛰기')
    args = ap.parse_args()

    device = torch.device(args.device)
    dump_dir = osp.join(args.out, osp.basename(osp.dirname(args.attack_ckpt)))
    os.makedirs(dump_dir, exist_ok=True)

    model = build_detector(args, device)
    attacker = load_attacker(args, device)
    print(f'\n진단 대상 {args.seq} | {args.half} half | eps {args.eps}/255 | {device}\n')

    noise, det_c, det_a = run_frames(args, model, attacker, device, dump_dir)

    desc = None
    if not args.skip_desc:
        print()
        desc = run_descriptor_check(args, model, attacker, device)

    verdict(noise, det_c, det_a, desc, args.eps / 255.0)
    print(f'이미지 저장: {dump_dir}')
    print('  *_1clean.jpg / *_2adv.jpg / *_3noise.jpg (delta 를 회색 128 기준으로 증폭)')


if __name__ == '__main__':
    main()
