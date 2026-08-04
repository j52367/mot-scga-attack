# -*- coding: utf-8 -*-
"""
공격 효과 평가

한 시퀀스를 프레임 순서대로 돌리면서
    (선택) generator 로 노이즈 주입 -> 검출기 -> ByteTrack -> 결과 기록
을 수행하고, GT 와 비교해 MOTA / IDF1 / IDSW 를 낸다.

지표 해석 (DESIGN.md)
    MOTA  검출 종합. 오탐(FP)만 뿌려도 크게 떨어지므로 이것만 보면 안 된다.
    IDF1  ID 유지 품질. ★ 이것이 떨어져야 '추적을 무너뜨렸다'고 할 수 있다.
    IDSW  ID 가 바뀐 횟수.

실행 예
    # 클린 기준값
    PYTHONPATH=. python attack/evaluate.py --seqs MOT17-02-FRCNN

    # 공격 적용
    PYTHONPATH=. python attack/evaluate.py --seqs MOT17-02-FRCNN \
        --attack_ckpt experiments/xxx/teacher_ptca_final.pth
"""

import argparse
import os
import os.path as osp
import time

import cv2
import numpy as np
import torch

import attack.compat  # noqa: F401  yolox import 보다 먼저 (np.float 등 별칭 복원)

from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.evaluation import Evaluator
from yolox.utils import postprocess

from attack.dataset import letterbox, normalize_for_detector, read_seq_length
from attack.generator_wrap import GeneratorWithFeatures
from attack.losses import apply_perturbation
from attack.train_generator import build_detector
from generator import build_generator


# ── 트래커 인자 (tools/track.py 기본값) ──────────────────────────────
class TrackerArgs:
    track_thresh = 0.6
    track_buffer = 30
    match_thresh = 0.9
    mot20 = False


def load_attacker(args, device):
    """학습된 generator 를 불러온다. 없으면 None (클린 평가)."""
    if not args.attack_ckpt:
        return None
    netG = GeneratorWithFeatures(
        build_generator(generator=args.generator, gap=args.gap,
                        inception=False, nat=args.nat))
    state = torch.load(args.attack_ckpt, map_location='cpu')
    netG.load_state_dict(state)
    netG.to(device).eval()
    for p in netG.parameters():
        p.requires_grad_(False)
    print(f'attacker loaded: {args.attack_ckpt}')
    return netG


@torch.no_grad()
def run_sequence(seq_dir, model, attacker, args, device):
    """한 시퀀스를 추적하고 MOT 형식 결과 문자열 리스트를 반환한다."""
    n_frames = read_seq_length(seq_dir)
    lo, hi = 1, n_frames
    if args.half == 'first':
        hi = n_frames // 2
    elif args.half == 'second':
        lo = n_frames // 2 + 1
    if args.max_frames:
        hi = min(hi, lo + args.max_frames - 1)

    tracker = BYTETracker(TrackerArgs(), frame_rate=30)
    results = []
    eps = args.eps / 255.0
    t0 = time.time()

    for frame_id in range(lo, hi + 1):
        path = osp.join(seq_dir, 'img1', f'{frame_id:06d}.jpg')
        img_bgr = cv2.imread(path)
        orig_h, orig_w = img_bgr.shape[:2]

        img, _ = letterbox(img_bgr, (args.input_h, args.input_w))
        x = torch.from_numpy(img).unsqueeze(0).to(device)      # [1,3,H,W] in [0,1]

        # ── 공격 주입 ────────────────────────────────────────
        if attacker is not None:
            adv = attacker(x)
            if isinstance(adv, tuple):
                adv = adv[0]
            # 학습 때와 같은 파라미터화를 써야 한다. 다르면 공격이 성립하지 않는다.
            x = apply_perturbation(adv, x, eps, args.delta_param)

        # ── 검출 ─────────────────────────────────────────────
        out = model(normalize_for_detector(x))
        out = postprocess(out, args.num_classes, args.conf_thre, args.nms_thre)[0]

        # ── 추적 ─────────────────────────────────────────────
        if out is not None and len(out):
            online = tracker.update(out, (orig_h, orig_w),
                                    (args.input_h, args.input_w))
            for t in online:
                x1, y1, w, h = t.tlwh
                if w * h < args.min_box_area or w / max(h, 1e-6) > 1.6:
                    continue
                results.append(
                    f'{frame_id},{t.track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},'
                    f'{t.score:.2f},-1,-1,-1\n')

        if args.verbose and (frame_id - lo) % 50 == 0:
            el = time.time() - t0
            done = frame_id - lo + 1
            print(f'    {done}/{hi-lo+1}  {el:.0f}초  ({el/done:.2f}초/프레임)')

    return results


def main():
    ap = argparse.ArgumentParser()
    # 데이터
    ap.add_argument('--data_root', default='datasets/mot/train')
    ap.add_argument('--seqs', nargs='+', default=['MOT17-02-FRCNN'])
    ap.add_argument('--half', default='second', choices=['first', 'second', 'all'],
                    help='학습에 first 를 썼으므로 평가는 second 가 기본')
    # 검출기
    ap.add_argument('--exp_file', default='exps/example/mot/yolox_s_mix_det.py')
    ap.add_argument('--ckpt', default='pretrained/bytetrack_s_mot17.pth.tar')
    ap.add_argument('--input_h', type=int, default=800)
    ap.add_argument('--input_w', type=int, default=1440)
    ap.add_argument('--conf_thre', type=float, default=0.001)
    ap.add_argument('--nms_thre', type=float, default=0.7)
    ap.add_argument('--num_classes', type=int, default=1)
    ap.add_argument('--min_box_area', type=float, default=100)
    # 공격
    ap.add_argument('--attack_ckpt', default=None, help='없으면 클린 평가')
    ap.add_argument('--eps', type=float, default=10)
    ap.add_argument('--delta_param', default='additive',
                    choices=['additive', 'image'],
                    help='★ 학습 때 쓴 값과 반드시 같아야 한다')
    ap.add_argument('--generator', default='resnet')
    ap.add_argument('--gap', action='store_true')
    ap.add_argument('--nat', action='store_true')
    # 기타
    ap.add_argument('--out', default='output/eval')
    ap.add_argument('--tag', default=None, help='결과 폴더 이름. 기본은 clean/attack')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--max_frames', type=int, default=0,
                    help='디버깅용. 0이면 전체')
    args = ap.parse_args()

    device = torch.device(args.device)
    tag = args.tag or ('attack' if args.attack_ckpt else 'clean')
    out_dir = osp.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    model = build_detector(args, device)
    attacker = load_attacker(args, device)
    print(f'모드: {tag} | 장치: {device} | 입력 {args.input_h}x{args.input_w}\n')

    accs, names = [], []
    for seq in args.seqs:
        seq_dir = osp.join(args.data_root, seq)
        print(f'[{seq}]')
        t0 = time.time()
        results = run_sequence(seq_dir, model, attacker, args, device)
        res_file = osp.join(out_dir, f'{seq}.txt')
        with open(res_file, 'w') as f:
            f.writelines(results)
        print(f'  추적 완료 {time.time()-t0:.0f}초, 결과 {len(results)}줄 -> {res_file}')

        ev = Evaluator(args.data_root, seq, 'mot')
        accs.append(ev.eval_file(res_file))
        names.append(seq)

    import motmetrics as mm
    summary = Evaluator.get_summary(accs, names)
    print()
    print(mm.io.render_summary(
        summary, formatters=mm.metrics.create().formatters,
        namemap=mm.io.motchallenge_metric_names))

    csv = osp.join(out_dir, 'summary.csv')
    summary.to_csv(csv)
    print(f'\n저장: {csv}')


if __name__ == '__main__':
    main()
