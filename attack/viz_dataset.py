# -*- coding: utf-8 -*-
"""
데이터로더 시각 검증

letterbox 배율 r 이 박스에 제대로 적용됐는지 눈으로 확인한다.
박스가 어긋난 채로 학습하면 아무것도 되지 않으므로, 실험 전 필수 점검이다.

실행:
    source ~/.venv/bin/activate
    cd ~/ssu/mot-scga-attack
    PYTHONPATH=. python attack/viz_dataset.py --n 3
"""

import argparse
import os

import cv2
import numpy as np

from attack.dataset import MOT17PairDataset


def draw(img_chw01, boxes, ids, title):
    """[3,H,W] [0,1] RGB -> 박스가 그려진 BGR uint8"""
    img = (img_chw01.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for b, i in zip(boxes.tolist(), ids.tolist()):
        x1, y1, x2, y2 = [int(v) for v in b]
        # ID 마다 색을 고정해 두 프레임에서 같은 사람이 같은 색이 되게 한다
        rng = np.random.RandomState(int(i))
        color = tuple(int(c) for c in rng.randint(80, 255, 3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, str(i), (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(img, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='datasets/mot')
    ap.add_argument('--detector', default='FRCNN')
    ap.add_argument('--input_h', type=int, default=800)
    ap.add_argument('--input_w', type=int, default=1440)
    ap.add_argument('--n', type=int, default=3, help='저장할 샘플 수')
    ap.add_argument('--out', default='output/viz')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ds = MOT17PairDataset(
        data_root=args.data_root, split='train', detector=args.detector,
        half='first', input_size=(args.input_h, args.input_w),
    )
    print(f'학습 쌍: {len(ds)}')

    step = max(1, len(ds) // args.n)
    for k in range(args.n):
        s = ds[k * step]
        a = draw(s['img_t'], s['boxes_t'], s['ids_t'], f'frame t   (n={len(s["ids_t"])})')
        b = draw(s['img_u'], s['boxes_u'], s['ids_u'], f'frame t+d (n={len(s["ids_u"])})')
        pair = np.concatenate([a, b], axis=0)          # 위아래로 붙임
        path = os.path.join(args.out, f'pair_{k:02d}.jpg')
        cv2.imwrite(path, pair, [cv2.IMWRITE_JPEG_QUALITY, 85])

        common = set(s['ids_t'].tolist()) & set(s['ids_u'].tolist())
        print(f'  {path}  공통ID {len(common)}개')

    print('\n확인할 것')
    print('  1) 박스가 사람 위에 정확히 놓였는가')
    print('  2) 아래쪽(패딩) 회색 영역에 박스가 걸치지 않는가')
    print('  3) 같은 색 박스가 두 프레임에서 같은 사람인가')


if __name__ == '__main__':
    main()
