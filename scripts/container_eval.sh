# python3 train_demo.py  --mode intra \
# --lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 1 --Q 1 \
# --train_iter 5000 --val_iter 500 --test_iter 500 --val_step 1000 \
# --max_length 64 --model container --tau 0.05 --seed 0 --name container__OntoNotesToI2B2noviterbiFullTestEval --proj-dim 128 --with-dropout --only_test --load_ckpt checkpoint/container__Ontonotesnoviterbi.pth.tar --full-test --use-i2b2 --totalN 19



python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 5000 --val_iter 500 --test_iter 50 --val_step 1000 \
--max_length 64 --model container --tau 0.05 --seed 0 --name CONTAINERvis --proj-dim 128 --with-dropout --only_test --load_ckpt checkpoint/RUN_ALL_CONTAINER_FewNERD_INTRA5way1shot_seed=0.pth.tar --visualize



# python3 train_demo.py  --mode intra \
# --lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 1 --Q 5 \
# --train_iter 5000 --val_iter 500 --test_iter 500 --val_step 1000 \
# --max_length 32 --model container --tau 0.05 --seed 0 --name EvalWNUT --proj-dim 128 --with-dropout --only_test --load_ckpt checkpoint/RUN_ALL_CONTAINER_OntoNotes_DomainTransferPretrained_1shot_seed=0.pth.tar --full-test --use-wnut --totalN 6 --output-file wnut-test-container
