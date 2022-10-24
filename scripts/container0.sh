python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 5000 --val_iter 500 --test_iter 500 --val_step 1000 \
--max_length 32 --model container --tau 0.05 --seed 0 --name container__Ontonotesviterbi --proj-dim 128 --use-ontonotes