# python3 train_demo.py  --mode intra \
# --lr 1e-4 --batch_size 4 --trainN 5 --N 5 --K 1 --Q 1 \
# --train_iter 10000 --val_iter 500 --test_iter 50 --val_step 1000 \
# --max_length 64 --model transfer --seed 0 --name transfer_test --no-shuffle --eval-with-finetune --train-classes 12 --val-classes 6 --test-classes 6 \
# --only_test --load_ckpt checkpoint/transfer_test.pth.tar


python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 4 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 5 --test_iter 50 --val_step 1000 \
--max_length 64 --model transfer --seed 0 --name transfer_test --no-shuffle --eval-with-finetune