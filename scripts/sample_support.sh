python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name ontoA --use-conll A --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/ontoA/1shot --totalN 6


python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name ontoB --use-conll B --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/ontoB/1shot --totalN 6

python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name ontoC --use-conll C --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/ontoC/1shot --totalN 6

python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name conll --use-conll2003 --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/conll/1shot --totalN 4

python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name wnut --use-wnut --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/wnut/1shot --totalN 6


python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name i2b2 --use-i2b2 --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/i2b2/1shot --totalN 19


python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name gum --use-gum --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/gum/1shot --totalN 11


python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name ontoA --use-conll A --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/ontoA/5shot --totalN 6


python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name ontoB --use-conll B --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/ontoB/5shot --totalN 6

python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name ontoC --use-conll C --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/ontoC/5shot --totalN 6

python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name conll --use-conll2003 --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/conll/5shot --totalN 4

python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name wnut --use-wnut --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/wnut/5shot --totalN 6


python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name i2b2 --use-i2b2 --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/i2b2/5shot --totalN 19


python3 train_demo.py  --mode intra \
--lr 1e-4 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --seed 0 --name gum --use-gum --full-test --no-shuffle --sample-support-with-dir ../EntLM/dataset/gum/5shot --totalN 11
