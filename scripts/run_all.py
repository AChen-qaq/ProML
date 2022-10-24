import numpy as np
import re
import subprocess
print("start")

models_intra = {'NCAl2': '(CUDA_VISIBLE_DEVICES=6 python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed {1} --name NCAl2_{0}_INTRA-{1} --proj-dim {0}) > outputs/NCAl2_{0}_INTRA-{1}',
                'NCAl2Drop': '(CUDA_VISIBLE_DEVICES=6 python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed {1} --name NCAl2Drop_{0}_INTRA-{1} --proj-dim {0} --with-dropout) > outputs/NCAl2Drop_{0}_INTRA-{1}',
                'container': '(CUDA_VISIBLE_DEVICES=6 python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model container --tau 0.318 --seed {1} --name container_{0}_INTRA-{1} --proj-dim {0})> outputs/container_{0}_INTRA-{1}', 'NCAl2_no_proj': '(CUDA_VISIBLE_DEVICES=3 python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed {1} --name NCAl2_no_proj_{0}_INTRA-{1} --proj-dim {0} --train-without-proj --normalize) > outputs/NCAl2_no_proj_{0}_INTRA-{1}',
                'NCAl2Drop_no_proj': '(CUDA_VISIBLE_DEVICES=3 python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed {1} --name NCAl2Drop_no_proj_{0}_INTRA-{1} --proj-dim {0} --with-dropout --train-without-proj --normalize) > outputs/NCAl2Drop_no_proj_{0}_INTRA-{1}',
                'NCAl2_eval_proj': '(CUDA_VISIBLE_DEVICES=3 python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed {1} --name NCAl2_eval_proj_{0}_INTRA-{1} --proj-dim {0} --eval-with-proj) > outputs/NCAl2_eval_proj_{0}_INTRA-{1}',
                'NCAl2Drop_eval_proj': '(CUDA_VISIBLE_DEVICES=3 python3 train_demo.py  --mode intra \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed {1} --name NCAl2Drop_eval_proj_{0}_INTRA-{1} --proj-dim {0} --with-dropout --eval-with-proj) > outputs/NCAl2Drop_eval_proj_{0}_INTRA-{1}'
                }

models_inter = {'NCAl2': '(CUDA_VISIBLE_DEVICES=0 python3 train_demo.py  --mode inter \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed {1} --name NCAl2_{0}_INTER-{1} --proj-dim {0}) > outputs/NCAl2_{0}_INTER-{1}',
                'NCAl2Drop': '(CUDA_VISIBLE_DEVICES=0 python3 train_demo.py  --mode inter \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed {1} --name NCAl2Drop_{0}_INTER-{1} --proj-dim {0} --with-dropout) > outputs/NCAl2Drop_{0}_INTER-{1}',
                'container': '(CUDA_VISIBLE_DEVICES=0 python3 train_demo.py  --mode inter \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model container --tau 0.318 --seed {1} --name container_{0}_INTER-{1} --proj-dim {0})> outputs/container_{0}_INTER-{1}',
                'NCAl2_no_proj': '(CUDA_VISIBLE_DEVICES=0 python3 train_demo.py  --mode inter \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed {1} --name NCAl2_no_proj_{0}_INTER-{1} --proj-dim {0} --train-without-proj --normalize) > outputs/NCAl2_no_proj_{0}_INTER-{1}',
                'NCAl2Drop_no_proj': '(CUDA_VISIBLE_DEVICES=0 python3 train_demo.py  --mode inter \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed {1} --name NCAl2Drop_no_proj_{0}_INTER-{1} --proj-dim {0} --with-dropout --train-without-proj --normalize) > outputs/NCAl2Drop_no_proj_{0}_INTER-{1}',
                'NCAl2_eval_proj': '(CUDA_VISIBLE_DEVICES=0 python3 train_demo.py  --mode inter \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed {1} --name NCAl2_eval_proj_{0}_INTER-{1} --proj-dim {0} --eval-with-proj) > outputs/NCAl2_eval_proj_{0}_INTER-{1}',
                'NCAl2Drop_eval_proj': '(CUDA_VISIBLE_DEVICES=0 python3 train_demo.py  --mode inter \
--lr 3e-5 --batch_size 1 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 32 --model WeightedNCA --tau 0.318 --seed {1} --name NCAl2Drop_eval_proj_{0}_INTER-{1} --proj-dim {0} --with-dropout --eval-with-proj) > outputs/NCAl2Drop_eval_proj_{0}_INTER-{1}'
                }


# for seed in range(1, 2):
#     for model in models_inter:
#         for dim in [32, 128, 768]:
#             print(seed, model, dim)
#             subprocess.call(models_inter[model].format(dim, seed), shell=True)


for model in models_inter:
    for dim in [32, 128, 768]:
        f1s = []
        for seed in range(3):
            # print(seed, model, dim)
            filename = models_inter[model].format(dim, seed).split()[-1]
            with open(filename) as f:
                logs = f.read()
                ret = re.search('RESULT.*f1:.*\n', logs)
                f1 = 0 if ret is None else float(ret.group().split(':')[-1])
                f1s.append(f1)
        exp_name = models_inter[model].format(dim, seed).split()[-1]
        exp_name = exp_name.split('/')[-1].split('-')[0]
        print(exp_name, 'mean={}, std={}'.format(np.mean(f1s), np.std(f1s)))


# for seed in range(3):
#     for model in models_:
#         for dim in [32, 128, 768]:
#             print(seed, model, dim)
#             subprocess.call(models_[model].format(dim, seed), shell=True)
# print("end")
