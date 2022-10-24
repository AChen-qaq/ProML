CUDA_VISIBLE_DEVICES=1 python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed 1 --name WeightedNCA1 --normalize