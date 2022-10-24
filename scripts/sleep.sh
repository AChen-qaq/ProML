# python3 train_demo.py  --mode intra \
# --lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
# --train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
# --max_length 32 --model container --tau 0.318 >KLNCE

python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model KLnnshot --tau 0.318 >KLCE

# python3 train_demo.py  --mode intra \
# --lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
# --train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
# --max_length 32 --model nnshot --tau 0.318 >L2MCE

# python3 train_demo.py  --mode intra \
# --lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
# --train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
# --max_length 32 --model contrast_miaom --tau 0.318 >L2NCE


