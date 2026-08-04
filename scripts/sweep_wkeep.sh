#!/usr/bin/env bash
#
# w_keep 트레이드오프 곡선
#
#   w_keep 이 크면  검출은 보존되지만 공격이 학습되지 않는다 (Exp-004/005)
#   w_keep 이 0 이면 공격은 학습되지만 검출이 무너진다      (Exp-002/003)
#
# 중간에 둘 다 되는 지점이 있는지를 본다. 없다면 그 곡선 자체가 결과다.
#
# 다른 조건은 전부 고정한다 (w_iou 1.0, 1 epoch, batch 4, seed 42).
# seed 가 같으므로 배치 순서도 같아 스텝 단위로 비교할 수 있다.
#
# 실행:
#   nohup bash scripts/sweep_wkeep.sh > logs/sweep.log 2>&1 &
#   tail -f logs/sweep.log
#
# 한 지점당 학습 31분 + 평가 1분, 총 2시간 남짓.
# 한 지점이 실패해도 나머지는 계속 돈다.

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

COMMON="--data_root datasets/mot \
        --exp_file exps/example/mot/yolox_s_mix_det.py \
        --ckpt pretrained/bytetrack_s_mot17.pth.tar \
        --batch_size 4 --epochs 1 --w_iou 1.0"

mkdir -p logs
OUT=logs/sweep_wkeep.csv
echo "w_keep,exp_dir,decorr,iou,dscore,bgup,keepiou,MOTA,IDF1,IDSW,Prcn,Rcll" > "$OUT"

# 1.0 은 Exp-005 에서 이미 확보했으나, 조건을 완전히 맞추기 위해 다시 돈다
for WK in 0 0.1 0.3 1.0 3.0; do
    echo ""
    echo "=============================================================="
    echo "  w_keep=$WK  학습 시작  $(date '+%m-%d %H:%M')"
    echo "=============================================================="
    python attack/train_generator.py $COMMON --w_keep "$WK"

    DIR=$(ls -dt experiments/*/ | head -1)
    CKPT="${DIR}teacher_ptca_final.pth"
    if [ ! -f "$CKPT" ]; then
        echo "  [건너뜀] 체크포인트가 없다: $CKPT"
        continue
    fi

    echo "--- w_keep=$WK 평가 ---"
    TAG="wk${WK}"
    python attack/evaluate.py --seqs MOT17-02-FRCNN \
        --attack_ckpt "$CKPT" --tag "$TAG"

    python scripts/collect_sweep.py "$WK" "$DIR" "output/eval/${TAG}/summary.csv" \
        >> "$OUT"
    echo "--- 누적 결과 ---"
    cat "$OUT"
done

echo ""
echo "=============================================================="
echo "  스윕 완료  $(date '+%m-%d %H:%M')   ->  $OUT"
echo "=============================================================="
cat "$OUT"
