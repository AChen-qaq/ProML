python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 4 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 500 --val_step 1000 \
--max_length 64 --model PromptNCA --tau 0.05 --seed 0 --name casedPROMPTNCAOntoNotesB --proj-dim 128 --with-dropout --trainK 1 --use-conll B