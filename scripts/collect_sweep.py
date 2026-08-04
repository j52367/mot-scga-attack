# -*- coding: utf-8 -*-
"""
스윕 한 지점의 결과를 CSV 한 줄로 뽑는다.

    python scripts/collect_sweep.py <w_keep> <실험폴더> <평가 summary.csv>

학습 손실은 마지막 10개 기록의 평균을 쓴다.
배치마다 객체 수가 달라 한 스텝만 보면 편차가 크기 때문이다.
"""

import csv
import os.path as osp
import sys

NAN = float('nan')

# motmetrics 내부 이름 -> 우리가 쓸 이름.
# summary.csv 는 비율(0.55)로 저장되므로 논의에 쓰는 % 로 바꾼다. IDSW 는 개수라 그대로.
METRICS = {
    'MOTA': (('mota',), 100.0),
    'IDF1': (('idf1',), 100.0),
    'IDSW': (('num_switches', 'idsw'), 1.0),
    'Prcn': (('precision', 'prcn'), 100.0),
    'Rcll': (('recall', 'rcll'), 100.0),
}


def tail_mean(path, cols, n=10):
    """loss_log.csv 의 마지막 n개 기록 평균."""
    out = {c: NAN for c in cols}
    if not osp.exists(path):
        return out
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return out
    rows = rows[-n:]
    for c in cols:
        vals = []
        for r in rows:
            try:
                v = float(r.get(c, 'nan'))
            except ValueError:
                continue
            if v == v:                      # NaN 제외
                vals.append(v)
        out[c] = sum(vals) / len(vals) if vals else NAN
    return out


def read_eval(path):
    """평가 summary.csv 에서 지표를 꺼낸다. 이름이 바뀌어도 죽지 않게 한다."""
    out = {k: NAN for k in METRICS}
    if not osp.exists(path):
        return out
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return out
    row = rows[-1]                          # OVERALL 행
    lower = {k.lower(): v for k, v in row.items() if k}
    for name, (cands, scale) in METRICS.items():
        for c in cands:
            if c in lower:
                try:
                    out[name] = float(lower[c]) * scale
                except (TypeError, ValueError):
                    pass
                break
    return out


def main():
    w_keep, exp_dir, eval_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    loss = tail_mean(osp.join(exp_dir, 'loss_log.csv'),
                     ['decorr', 'iou', 'dscore', 'bgup', 'keepiou'])
    ev = read_eval(eval_csv)

    fmt = lambda v: ('' if v != v else f'{v:.4f}')  # noqa: E731
    print(','.join([
        w_keep, exp_dir.rstrip('/').split('/')[-1],
        fmt(loss['decorr']), fmt(loss['iou']),
        fmt(loss['dscore']), fmt(loss['bgup']), fmt(loss['keepiou']),
        fmt(ev['MOTA']), fmt(ev['IDF1']), fmt(ev['IDSW']),
        fmt(ev['Prcn']), fmt(ev['Rcll']),
    ]))


if __name__ == '__main__':
    main()
