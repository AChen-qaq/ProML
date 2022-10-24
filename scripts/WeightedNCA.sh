# python3 train_demo.py  --mode intra \
# --lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
# --train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
# --max_length 32 --model WeightedNCA --tau 0.318 --norm 32 --name WeightedNCA

# python3 train_demo.py  --mode intra \
# --lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
# --train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
# --max_length 32 --model WeightedNCA --tau 0.318 --normalize --norm 4 --name WeightedNCAn

python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --norm 4 --name WeightedNCA4_768