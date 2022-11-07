python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 5000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model ProML --eval-mix-rate 0.5 --seed 0 --name EvalGUM --proj-dim 128 --with-dropout --only_test --eval-with-finetune --load_ckpt checkpoint/RUN_ALL_ProMLJSMixedPlusAlter_onlyB1+Awithsplit_OntoNotes_DomainTransferPretrained_1shot_seed=0_mix-rate=0.5.pth.tar --full-test --use-gum --totalN 11



# python3 train_demo.py  --mode intra \
# --lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 1 --Q 1 \
# --train_iter 5000 --val_iter 500 --test_iter 50 --val_step 1000 \
# --max_length 64 --model ProML --tau 0.05 --seed 0 --name ProMLvis --proj-dim 128 --with-dropout --only_test --load_ckpt checkpoint/RUN_ALL_ProMLJSMixedPlusAlter_onlyB1+Awithsplit_FewNERD_INTRA5way1shot_seed=0_mix-rate=0.5.pth.tar --visualize