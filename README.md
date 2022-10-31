# ProML: Prompt-Based Metric Learning for Few-shot NER

This repo contains the source code for our paper:  [**Prompt-Based Metric Learning for Few-shot NER**](https://openreview.net/pdf?id=wHt8UumYfGT).


## Requirements

Run the following script to install the remaining dependencies,

```shell
pip install -r requirements.txt
```



## Run ProML

Run `scripts/prompt_nca_run_all.py` to run all experiments mentioned in our paper (You may need to place all datasets in the correct path).

Or run `train_demo.py` with specified arguments to run a single experiment.

### Arguments

TODO

### For Few-NERD Dataset
An example of script for running experiment for Few-NERD INTRA 5-way 1-shot episode evaluation is given as below.

```shell
python3 train_demo.py  --mode intra --lr 3e-5 --batch_size 4 --trainN 5 --N 5 --trainK 1 --K 1 --Q 1 --train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 --max_length 64 --model PromptNCA --seed 0 --mix-rate 0.7 --use_sampled_data --name RUN_ALL_ProML_FewNERD_INTRA5way1shot_seed=0_mix-rate=0.7 --proj-dim 128
```

Few-NERD dataset will be automatically downloaded and placed at `data/`. In the case that the links in `data/download.sh` are expired, you can visit their official website to access the dataset.

You can refer to `run_all_FewNERD()` in `scripts/prompt_nca_run_all.py` for more examples.

### For Ontonotes-ABC Splits
We didn't provide the data for Ontonotes dataset since Ontonotes is publicly not available. You can request the data from their official website.

Suppose the train/valid/test data (in conll format) is placed in `data/ontoNotes/train.sd.conllx`, `data/ontoNotes/dev.sd.conllx`, `data/ontoNotes/test.sd.conllx`, you can run `process_conll.py` to preprocess the dataset and get Ontonotes-ABC data splits.

Then you can run the following script.

```shell
python3 train_demo.py \
--lr 3e-5 --batch_size 4 --trainN 5 --N 5 --trainK 1 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --model PromptNCA --seed 0 --mix-rate 0.7 --name RUN_ALL_ProML_OntoNotes_TagExtensionA1shot_seed=0_mix-rate=0.7 --use-onto-split A --proj-dim 128
```

This script will run an experiment with 5-way 1-shot episode evaluation setting for Ontonotes-A data split. 

You can refer to `run_all_OntoNotes()` in `scripts/prompt_nca_run_all.py` for more examples.

### For Domain Transfer
For domain transfer tasks, the model need to first trained on a source domain. We use Ontonotes as the training source domain. The required preprocessings are also integrated in `process_conll.py`.

Suppose the preprocessed train/valid/test data is at `data/ontoNotes/train.txt`,`data/ontoNotes/dev.txt`,`data/ontoNotes/test.txt`, then you can run the following script to train ProML on source domain.

```shell
python3 train_demo.py \
--lr 3e-5 --batch_size 4 --trainN 5 --N 5 --trainK 1 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --model PromptNCA --seed 0 --mix-rate 0.7 --name RUN_ALL_ProML_OntoNotes_DomainTransferPretrained_1shot_seed=0_mix-rate=0.7 --use-ontonotes --proj-dim 128
```

You can refer to `run_all_DomainTransfer()` in `scripts/prompt_nca_run_all.py` for more examples.

After training on source domain, you can evaluate ProML on target domain. 

```shell
python3 train_demo.py \
--lr 3e-5 --batch_size 1 --totalN 4 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --model PromptNCA --seed 0 --mix-rate 0.7 --eval-mix-rate 0.7 --name RUN_ALL_ProML_OntoNotes_DomainTransferPretrained_1shot_seed=0_mix-rate=0.7 --proj-dim 128 --only_test --full-test --use-conll2003 --load_ckpt=checkpoint/RUN_ALL_ProML_OntoNotes_DomainTransferPretrained_1shot_seed=0_mix-rate=0.7.pth.tar
```

The above script will evaluate ProML on CoNLL2003 dataset (`data/conll-test.txt`). It will produce support sets according to arguments `--totalN, --K`, and use the whole dataset as query set. You can refer to `DomainTransferEval()` in `scripts/prompt_nca_run_all.py` for more examples.



You can also perform a low-resource evaluation for Ontonotes-ABC split from a pretrained checkpoint.

```shell
python3 train_demo.py \
--lr 3e-5 --batch_size 1 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --model PromptNCA --seed 0 --mix-rate 0.7 --name RUN_ALL_ProML_OntoNotes_TagExtensionA1shot_seed=0_mix-rate=0.7 --only_test --full-test --load_ckpt checkpoint/RUN_ALL_ProML_OntoNotes_TagExtensionA1shot_seed=0_mix-rate=0.7.pth.tar --use-onto-split A --proj-dim 128
```

The above script will evaluate ProML on Ontonotes-A data split (`data/ontoNotes/__test_A.txt`) for 5-way 1-shot setting. You may need to change the `--load_ckpt` argument with an appropriate checkpoint path. You can refer to `OntoNotesEval()` in `scripts/prompt_nca_run_all.py` for more examples.