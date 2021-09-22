CKPTS=("loss26" "loss28" "loss30" "loss32" "loss34")
#CKPTS=("loss26")
for CKPT in ${CKPTS[@]}; do
    python ./expand_nbars.py \
        --ckpt-path ./trained-model/$CKPT.ckpt \
        --result-dir ./expand_result/$CKPT \
        --target-file ./expand_target_list_test.txt
done
