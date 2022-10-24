python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 100 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --use_sampled_data --seed 0 --name NCAl2DropTest --proj-dim 768 --with-dropout --eval-with-finetune --only_test --load_ckpt checkpoint/NCAl2DropTest.pth.tar