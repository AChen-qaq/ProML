python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 8 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --model nnshot --tau 0.318 --seed 0 --name nnshot_conllA --use-conll A