# ProML: Prompt-Based Metric Learning for Few-shot NER

This repo contains the source code for our paper:  [**Prompt-Based Metric Learning for Few-shot NER**](https://arxiv.org/abs/2211.04337). 


## Requirements

Run the following script to install all dependencies

```shell
pip install -r requirements.txt
```



## Run ProML

Run `scripts/proml_run_all.py` to run all experiments mentioned in our paper (You may need to place all datasets in the correct path).

Or run `train_demo.py` with specified arguments to run a single experiment.

### Arguments

```shell
-- mode                 specify the training mode for few-nerd dataset, must be inter or intra
-- trainN               N-way in train
-- N                    N-way in val and test
-- trainK               K-shot in train
-- K                    K-shot in val and test
-- Q                    num of query per class
-- batch_size           batch size
-- train_iter           num of iters in training
-- val_iter             num of iters in validation
-- test_iter            num of iters in testing
-- val_step             val after training how many iters
-- model                model name, must be ProML, proto, nnshot, container, transfer or structshot
-- max_length           max length of tokenized sentence
-- lr                   learning rate
-- weight_decay         weight decay
-- load_ckpt            path to load model
-- save_ckpt            path to save model
-- only_test            no training process, only test
-- name                 experiment name
-- seed                 random seed
-- pretrain_ckpt        bert pre-trained checkpoint
-- use_sampled_data     whether to use sampled data
-- use-support          path to support set for low-resource evaluation, must be used together with -- use_sampled_data
-- use-query            path to test set for low-resource evaluation, must be used together with -- use_sampled_data
-- full-test            run test in low-resource evaluation mode
-- totalN               total N in low-resource evaluation
-- proj-dim             dim of gaussian embedding
-- mix-rate             the weighted averaging hyperparameter for ProML
-- eval-mix-rate        the weighted averaging hyperparameter for ProML used in evalution, overrides the --mix-rate in ckpt
-- topk                 inference with KNN
-- eval-with-finetune   finetune on support set
-- no-sep               no separator in prompts
```

### For Few-NERD Dataset
An example of script for running experiment for Few-NERD INTRA 5-way 1-shot episode evaluation is given as below.

```shell
python3 train_demo.py  --mode intra --lr 3e-5 --batch_size 4 --trainN 5 --N 5 --trainK 1 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --model ProML --seed 0 --mix-rate 0.7 --use_sampled_data --name RUN_ALL_ProML_FewNERD_INTRA5way1shot_seed=0_mix-rate=0.7 --proj-dim 128
```

Few-NERD dataset will be automatically downloaded and placed at `data/`. In the case that the links in `data/download.sh` are expired, you can visit their official website to access the dataset.

You can refer to `run_all_FewNERD()` in `scripts/proml_run_all.py` for more examples.

### For Ontonotes-ABC Splits
We didn't provide the data for Ontonotes dataset since Ontonotes is publicly not available. You can request the data from their official website.

Suppose the train/valid/test data (in conll format) is placed in `data/ontoNotes/train.sd.conllx`, `data/ontoNotes/dev.sd.conllx`, `data/ontoNotes/test.sd.conllx`, you can run `process_conll.py` to preprocess the dataset and get Ontonotes-ABC data splits.

Then you can run the following script.

```shell
python3 train_demo.py \
--lr 3e-5 --batch_size 4 --trainN 5 --N 5 --trainK 1 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --model ProML --seed 0 --mix-rate 0.7 --name RUN_ALL_ProML_OntoNotes_TagExtensionA1shot_seed=0_mix-rate=0.7 --use-onto-split A --proj-dim 128
```

This script will run an experiment with 5-way 1-shot episode evaluation setting for Ontonotes-A data split. 

You can refer to `run_all_OntoNotes()` in `scripts/proml_run_all.py` for more examples.

### For Domain Transfer
For domain transfer tasks, the model needs to be first trained on a source domain. We use Ontonotes as the training source domain. The required preprocessings are also integrated in `process_conll.py`.

Suppose the preprocessed train/valid/test data is at `data/ontoNotes/train.txt`,`data/ontoNotes/dev.txt`,`data/ontoNotes/test.txt`, then you can run the following script to train ProML on source domain.

```shell
python3 train_demo.py \
--lr 3e-5 --batch_size 4 --trainN 5 --N 5 --trainK 1 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --model ProML --seed 0 --mix-rate 0.7 --name RUN_ALL_ProML_OntoNotes_DomainTransferPretrained_1shot_seed=0_mix-rate=0.7 --use-ontonotes --proj-dim 128
```

You can refer to `run_all_DomainTransfer()` in `scripts/proml_run_all.py` for more examples.

After training on source domain, you can evaluate ProML on target domain. 

```shell
python3 train_demo.py \
--lr 3e-5 --batch_size 1 --totalN 4 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --model ProML --seed 0 --eval-mix-rate 0.7 --name RUN_ALL_ProML_OntoNotes_DomainTransferPretrained_1shot_seed=0_mix-rate=0.7 --proj-dim 128 --only_test --full-test --use-conll2003 --load_ckpt=checkpoint/RUN_ALL_ProML_OntoNotes_DomainTransferPretrained_1shot_seed=0_mix-rate=0.7.pth.tar
```

The above script will evaluate ProML on CoNLL2003 dataset (`data/conll-test.txt`). It will produce support sets according to arguments `--totalN, --K`, and use the whole dataset as query set. You can refer to `DomainTransferEval()` in `scripts/proml_run_all.py` for more examples.

You can also perform a low-resource evaluation for Ontonotes-ABC split from a pretrained checkpoint.

```shell
python3 train_demo.py \
--lr 3e-5 --batch_size 1 --totalN 6 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --model ProML --seed 0 --eval-mix-rate 0.7 --name RUN_ALL_ProML_OntoNotes_TagExtensionA1shot_seed=0_mix-rate=0.7 --only_test --full-test --load_ckpt checkpoint/RUN_ALL_ProML_OntoNotes_TagExtensionA1shot_seed=0_mix-rate=0.7.pth.tar --use-onto-split A --proj-dim 128
```

The above script will evaluate ProML on Ontonotes-A data split (`data/ontoNotes/__test_A.txt`) for 5-way 1-shot setting. You may need to change the `--load_ckpt` argument with an appropriate checkpoint path. You can refer to `OntoNotesEval()` in `scripts/proml_run_all.py` for more examples.

If you want to test the model on a customized target domain, you have to specify the test dataset together with a support set by \texttt{--use-query} and \texttt{--use-support} and also add \texttt{--use_sampled_data} flag. Currently we only support low-resource evaluation with CoNLL format dataset. An example script is as follows.

```shell
python3 train_demo.py \
--lr 3e-5 --batch_size 1 --totalN 6 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 64 --model ProML --seed 0 --eval-mix-rate 0.7 --name RUN_ALL_ProML_OntoNotes_TagExtensionA1shot_seed=0_mix-rate=0.7 --only_test --full-test --load_ckpt checkpoint/RUN_ALL_ProML_OntoNotes_TagExtensionA1shot_seed=0_mix-rate=0.7.pth.tar --use-support data/ontoA/1shot/0.txt --use-query data/ontoNotes __test_A.txt --use_sampled_data --proj-dim 128
```

## Citation
Please cite us if ProML is useful in your work:
```
@inproceedings{Chen2022PromptBasedML,
  title={Prompt-Based Metric Learning for Few-Shot NER},
  author={Yanru Chen and Yanan Zheng and Zhilin Yang},
  year={2022}
}
```

## Acknowledgement
Part of the code is developed based on [**Few-NERD: Not Only a Few-shot NER Dataset**](https://github.com/thunlp/Few-NERD). We appreciate all the contributors for making their code publicly available. They provide a basic few-shot NER framework together with a large-scale dataset, which saves us a lot of work.