# -*- coding: utf-8 -*-
"""
end-to-end 1스텝 테스트 (CPU)

실제 MOT17 프레임 쌍 + 실제 검출기 가중치로 학습 루프 한 스텝을 돌려,
generator -> 검출기 -> 서술자 -> 손실 -> generator gradient 경로가
끊김 없이 이어지는지 확인한다.

실행:
    source ~/.venv/bin/activate
    cd ~/ssu/mot-scga-attack
    PYTHONPATH=. python attack/e2e_test.py
"""

import argparse
import copy
import time

import torch

from generator import build_generator
from generator.resnet import weights_init_normal

from attack.dataset import MOT17PairDataset, pair_collate
from attack.descriptor import object_descriptor  # noqa: F401  (frame_descriptor 내부에서 사용)
from attack.generator_wrap import GeneratorWithFeatures
from attack.losses import (attack_loss, apply_perturbation, temporal_iou_loss,
                           detection_keep_loss, generator_consistency_loss,
                           update_ema)
from attack.train_generator import build_detector, clean_scores, frame_descriptor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_file', default='exps/example/mot/yolox_s_mix_det.py')
    ap.add_argument('--ckpt', default='pretrained/bytetrack_s_mot17.pth.tar')
    ap.add_argument('--data_root', default='datasets/mot')
    ap.add_argument('--detector_variant', default='FRCNN')
    # CPU 라 해상도를 줄여서 확인한다 (32의 배수여야 함)
    ap.add_argument('--input_h', type=int, default=320)
    ap.add_argument('--input_w', type=int, default=576)
    ap.add_argument('--fpn_level', type=int, default=1)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--iou_thr', type=float, default=0.1)
    ap.add_argument('--roi_source', default='pred', choices=['pred', 'gt'])
    ap.add_argument('--no_center', action='store_true')
    ap.add_argument('--eps', type=float, default=10)
    ap.add_argument('--delta_param', default='additive',
                    choices=['additive', 'image'])
    ap.add_argument('--w_iou', type=float, default=1.0)
    ap.add_argument('--w_keep', type=float, default=1.0)
    ap.add_argument('--zero_dc', action='store_true')
    ap.add_argument('--generator', default='resnet')
    ap.add_argument('--gap', action='store_true')
    ap.add_argument('--nat', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f'장치 {device} | 입력 크기 {args.input_h}x{args.input_w}')
    if device.type == 'cuda':
        print(f'  {torch.cuda.get_device_name(0)}')
        torch.cuda.reset_peak_memory_stats()
    print()

    # ── 검출기 ───────────────────────────────────────────────────
    t0 = time.time()
    detector = build_detector(args, device)
    print(f'  로드 {time.time()-t0:.1f}초\n')

    # ── generator ────────────────────────────────────────────────
    student = GeneratorWithFeatures(
        build_generator(generator=args.generator, gap=args.gap,
                        inception=False, nat=args.nat)).to(device)
    student.apply(weights_init_normal)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    n_par = sum(p.numel() for p in student.parameters())
    print(f'generator 파라미터 {n_par/1e6:.1f}M\n')

    # ── 데이터 한 배치 ───────────────────────────────────────────
    ds = MOT17PairDataset(data_root=args.data_root, split='train',
                          detector=args.detector_variant, half='first',
                          input_size=(args.input_h, args.input_w))
    batch = {k: v.to(device) for k, v in pair_collate([ds[0]]).items()}
    img_t, img_u = batch['img_t'], batch['img_u']
    print(f'배치: img{tuple(img_t.shape)}  GT박스 {batch["boxes_t"].shape[0]}개\n')

    # ── 한 스텝 ──────────────────────────────────────────────────
    eps = args.eps / 255.0
    opt = torch.optim.Adam(student.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt.zero_grad()

    t0 = time.time()
    with torch.no_grad():
        _, ft_t = teacher(img_t, feat=True)
        _, ft_u = teacher(img_u, feat=True)
    adv_t, fs_t = student(img_t, feat=True)
    adv_u, fs_u = student(img_u, feat=True)
    print(f'generator forward {time.time()-t0:.1f}초')
    print(f'  블록 feature {len(ft_t)}개: ' +
          ', '.join(f'{tuple(f.shape[1:])}' for f in ft_t))

    adv_t = apply_perturbation(adv_t, img_t, eps, args.delta_param, args.zero_dc)
    adv_u = apply_perturbation(adv_u, img_u, eps, args.delta_param, args.zero_dc)
    d = (adv_t - img_t)
    print(f'  eps 적용 후 최대 편차 {d.abs().max():.4f}  (eps={eps:.4f})')
    print(f'  포화 비율 {(d.abs() >= 0.99*eps).float().mean()*100:.1f}% '
          f'(image 방식이면 여기서 gradient 가 죽는다)\n')

    gen_sim = (generator_consistency_loss(ft_t, fs_t)
               + generator_consistency_loss(ft_u, fs_u))

    t0 = time.time()
    ft = frame_descriptor(detector, adv_t, batch['boxes_t'], batch['ids_t'], args)
    fu = frame_descriptor(detector, adv_u, batch['boxes_u'], batch['ids_u'], args)
    d_t, ids_t, box_t = ft['d'], ft['ids'], ft['box']
    d_u, ids_u, box_u = fu['d'], fu['ids'], fu['box']
    print(f'검출기 forward + 서술자 {time.time()-t0:.1f}초')
    print(f'  서술자 d_t{tuple(d_t.shape)}  매칭 ID {len(ids_t)}개')
    print(f'  서술자 d_u{tuple(d_u.shape)}  매칭 ID {len(ids_u)}개')
    common = set(ids_t.tolist()) & set(ids_u.tolist())
    print(f'  공통 ID {len(common)}개  {sorted(common)[:8]}\n')

    if len(common) < 2:
        print('공통 ID 가 2개 미만이라 손실 계산 불가')
        return

    l_att, parts = attack_loss(d_t, d_u, ids_t, ids_u, lam=args.lam,
                               center=not args.no_center)
    loss = l_att + 0.7 * gen_sim
    print(f'손실  attack={l_att.item():+.4f}  (decorr={parts["decorr"]:+.4f}, '
          f'confuse={parts["confuse"]:+.4f})')

    l_iou = temporal_iou_loss(box_t, box_u, ids_t, ids_u)
    print(f'      iou={l_iou.item():.4f}  '
          f'<- 같은 객체의 프레임 간 박스 IoU (클린은 0.8~0.9)')
    if args.w_iou > 0:
        loss = loss + args.w_iou * l_iou

    l_keep, kp = detection_keep_loss(ft['scores'],
                                     clean_scores(detector, img_t, args),
                                     ft['box'], ft['gt'])
    print(f'      keep={l_keep.item():.4f}  (dscore={kp["dscore"]:.4f}, '
          f'keepiou={kp["keepiou"]:.4f})')
    print(f'        dscore 는 클린 대비 점수 편차, keepiou 는 예측-GT 겹침.')
    print(f'        학습 중 dscore 가 커지거나 keepiou 가 떨어지면 '
          f'검출을 부수고 있는 것이다.')
    if args.w_keep > 0:
        loss = loss + args.w_keep * l_keep
    print(f'      gen_sim={gen_sim.item():.4f}   total={loss.item():+.4f}\n')

    t0 = time.time()
    loss.backward()
    print(f'backward {time.time()-t0:.1f}초')

    gnorm = sum((p.grad**2).sum().item() for p in student.parameters()
                if p.grad is not None) ** 0.5
    n_with = sum(1 for p in student.parameters() if p.grad is not None
                 and p.grad.abs().sum() > 0)
    n_all = sum(1 for _ in student.parameters())
    print(f'  generator gradient norm {gnorm:.4f}')
    print(f'  gradient 받은 파라미터 {n_with}/{n_all}')
    assert gnorm > 0, 'generator 까지 gradient 가 도달하지 않았다'

    before = next(student.parameters()).clone()
    opt.step()
    update_ema(student, teacher)
    moved = (next(student.parameters()) - before).abs().sum().item()
    print(f'  step 후 파라미터 변화 {moved:.6f}')
    assert moved > 0

    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'\nVRAM 최대 사용 {peak:.2f} GB / {total:.0f} GB')
        print(f'  배치 1 기준이므로, 여유를 보고 --batch_size 를 올릴 수 있다')

    print('\nend-to-end 경로 확인 완료')


if __name__ == '__main__':
    main()
