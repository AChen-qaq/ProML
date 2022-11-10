from transformers import BertTokenizer
from util.data_loader import get_loader
from util.framework import FewShotNERFramework
from util.word_encoder import BERTWordEncoder
from model.proto import Proto
from model.nnshot import NNShot
from model.container import Container
from model.proml import ProML
from model.supervised import TransferBERT
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import time
import torch
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='intra',
                        help='training mode, must be in [inter, intra, supervised]')
    parser.add_argument('--trainN', default=5, type=int,
                        help='N in train')
    parser.add_argument('--N', default=5, type=int,
                        help='N way')
    parser.add_argument('--trainK', default=1, type=int,
                        help='K in train')
    parser.add_argument('--K', default=1, type=int,
                        help='K shot')
    parser.add_argument('--Q', default=1, type=int,
                        help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--train_iter', default=600, type=int,
                        help='num of iters in training')
    parser.add_argument('--val_iter', default=100, type=int,
                        help='num of iters in validation')
    parser.add_argument('--test_iter', default=500, type=int,
                        help='num of iters in testing')
    parser.add_argument('--val_step', default=20, type=int,
                        help='val after training how many iters')
    parser.add_argument('--model', default='proto',
                        help='model name, must be basic-bert, proto, nnshot, or structshot')
    parser.add_argument('--max_length', default=100, type=int,
                        help='max length')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
                        help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
                        help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
                        help='only test')
    parser.add_argument('--ckpt_name', type=str, default='',
                        help='checkpoint name.')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--ignore_index', type=int, default=-1,
                        help='label index to ignore when calculating loss and metrics')
    parser.add_argument('--use_sampled_data', action='store_true',
                        help='use released sampled data, the data should be stored at "data/episode-data/" ')

    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default=None,
                        help='bert / roberta pre-trained checkpoint')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true',
                        help='use dot instead of L2 distance for proto')

    # only for structshot
    parser.add_argument('--tau', default=0.05, type=float,
                        help='StructShot parameter to re-normalizes the transition probabilities')

    # experiment
    parser.add_argument('--use_sgd_for_bert', action='store_true',
                        help='use SGD instead of AdamW for BERT.')

    parser.add_argument('--proj-dim', type=int, default=32, help='the dimension of gaussian embedding')
    parser.add_argument('--mask-rate', type=float)

    parser.add_argument('--eval-with-finetune', action='store_true', help='finetune on support set')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--use-onto-split', type=str, choices=['A', 'B', 'C'], help='flag for OntoNotesABC splits')
    parser.add_argument('--use-wnut', action='store_true', help='flag for WNUT')
    parser.add_argument('--use-ontonotes', action='store_true', help='flag for OntoNotes')
    parser.add_argument('--use-conll2003', action='store_true', help='flag for CoNLL 2003')
    parser.add_argument('--use-i2b2', action='store_true', help='flag for I2B2')
    parser.add_argument('--use-gum', action='store_true', help='flag for GUM')
    parser.add_argument('--full-test', action='store_true', help='run test in low-resource evaluation mode')
    parser.add_argument('--totalN', type=int, default=5, help='total N in support set used in low-resource evaluation')
    parser.add_argument('--mix-rate', type=float, default=0.5, help='the weighted averaging hyperparameter for ProML')
    parser.add_argument('--eval-mix-rate', type=float, help='the weighted averaging hyperparameter for ProML used in evalution')
    parser.add_argument('--topk', type=int, default=1, help='KNN in inference')
    parser.add_argument('--output-file', type=str, help='write inference results to file, only for low-resource evaluation')
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--train-classes', type=int, default=50, help='used for transferBERT baseline')
    parser.add_argument('--val-classes', type=int, default=50, help='used for transferBERT baseline')
    parser.add_argument('--test-classes', type=int, default=50, help='used for transferBERT baseline')
    parser.add_argument('--sample-support-with-dir', type=str, help='only sample support set with a specified directory')
    parser.add_argument('--use-support', type=str)
    parser.add_argument('--use-query', type=str)
    parser.add_argument('--no-sep', action='store_true', help='no separator in prompts')

    parser.add_argument('--name', type=str, required=True)

    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    trainK = opt.trainK
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    max_length = opt.max_length

    print(opt.name)

    print("{}-way-{}-shot Few-Shot NER".format(N, K))
    print("model: {}".format(model_name))
    print("max_length: {}".format(max_length))
    print('mode: {}'.format(opt.mode))

    set_seed(opt.seed)
    print('loading tokenizer...')
    pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print('loading data...')

    if opt.use_onto_split is not None:
        opt.train = 'data/ontoNotes/__train_{}.txt'.format(opt.use_onto_split)
        opt.test = 'data/ontoNotes/__test_{}.txt'.format(opt.use_onto_split)
        opt.dev = 'data/ontoNotes/__dev_{}.txt'.format(opt.use_onto_split)
        dataset_name = 'OntoNotes_{}'.format(opt.use_onto_split)

    elif opt.use_ontonotes:
        opt.train = 'data/ontoNotes/train.txt'
        opt.test = 'data/ontoNotes/test.txt'
        opt.dev = 'data/ontoNotes/dev.txt'
        dataset_name = 'OntoNotes'

    elif opt.use_wnut:
        opt.train = 'data/ontoNotes/train.txt'
        opt.dev = 'data/wnut-dev.txt'
        opt.test = 'data/wnut-test.txt'
        dataset_name = 'WNUT'

    elif opt.use_conll2003:
        opt.train = 'data/ontoNotes/train.txt'
        opt.dev = 'data/conll-dev.txt'
        opt.test = 'data/conll-test.txt'
        dataset_name = 'CoNLL2003'

    elif opt.use_i2b2:
        opt.train = 'data/ontoNotes/train.txt'
        opt.dev = 'data/i2b2-test.txt'
        opt.test = 'data/i2b2-test.txt'
        dataset_name = 'I2B2'

    elif opt.use_gum:
        opt.train = 'data/ontoNotes/train.txt'
        opt.dev = 'data/gum-test.txt'
        opt.test = 'data/gum-test.txt'
        dataset_name = 'GUM'

    elif not opt.use_sampled_data:
        opt.train = f'data/{opt.mode}/train.txt'
        opt.test = f'data/{opt.mode}/test.txt'
        opt.dev = f'data/{opt.mode}/dev.txt'
        if not (os.path.exists(opt.train) and os.path.exists(opt.dev) and os.path.exists(opt.test)):
            os.system(f'bash data/download.sh {opt.mode}')
    else:
        opt.train = f'data/episode-data/{opt.mode}/train_{opt.N}_{opt.K}.jsonl'
        opt.test = f'data/episode-data/{opt.mode}/test_{opt.N}_{opt.K}.jsonl'
        opt.dev = f'data/episode-data/{opt.mode}/dev_{opt.N}_{opt.K}.jsonl'
        if not (os.path.exists(opt.train) and os.path.exists(opt.dev) and os.path.exists(opt.test)):
            os.system(f'bash data/download.sh episode-data')
            os.system('unzip -d data/ data/episode-data.zip')

    if not opt.full_test:
        train_data_loader = get_loader(opt.train, tokenizer,
                                    N=trainN, K=trainK, Q=Q, batch_size=batch_size, max_length=max_length, ignore_index=opt.ignore_index, use_sampled_data=opt.use_sampled_data, no_shuffle=opt.no_shuffle, no_sep=opt.no_sep)
        val_data_loader = get_loader(opt.dev, tokenizer,
                                    N=N, K=K, Q=Q, batch_size=batch_size, max_length=max_length, ignore_index=opt.ignore_index, use_sampled_data=opt.use_sampled_data, no_shuffle=opt.no_shuffle, no_sep=opt.no_sep)
        test_data_loader = get_loader(opt.test, tokenizer,
                                      N=N, K=K, Q=Q, batch_size=batch_size, max_length=max_length, ignore_index=opt.ignore_index, use_sampled_data=opt.use_sampled_data, no_shuffle=opt.no_shuffle, no_sep=opt.no_sep)
    else :
        train_data_loader = val_data_loader = None
        extra_data_loader = get_loader(opt.use_support or opt.test, tokenizer,
                                    N=opt.totalN, K=K, Q=Q, batch_size=1, max_length=max_length, ignore_index=opt.ignore_index, use_sampled_data=opt.use_sampled_data, i2b2flag=opt.use_i2b2 or opt.use_gum, dataset_name=dataset_name, no_shuffle=opt.no_shuffle, is_extra=True, no_sep=opt.no_sep)
        
        test_data_loader_creator, test_data_set = get_loader(opt.use_query or opt.test, tokenizer,
                                                             N=N, K=K, Q=Q, batch_size=1, max_length=max_length, ignore_index=opt.ignore_index, use_sampled_data=opt.use_sampled_data, full_test=True, no_shuffle=opt.no_shuffle, no_sep=opt.no_sep)
        test_data_loader = test_data_loader_creator, test_data_set

    if opt.sample_support_with_dir is not None:
        os.makedirs(opt.sample_support_with_dir, exist_ok=True)
        assert opt.full_test
        for __, (support, not_used) in enumerate(extra_data_loader):
            if __ >= 10:
                break
            # ret = []
            # for index in support['index']:
            #     sample = test_data_set.samples[index]
            #     ret.append(json.dumps({'text': sample.words, 'label': [
            #                tag.replace(' ', '_').replace('GATE', 'I') for tag in sample.tags]}))
            # with open(os.path.join(opt.sample_support_with_dir, '{}.json'.format(__)), 'w') as f:
            #     f.write('\n'.join(ret))

            with open(os.path.join(opt.sample_support_with_dir, '{}.txt'.format(__)), 'w') as f:
                f.write('\n\n'.join([test_data_set.samples[index].__str__() for index in support['index']]))
                f.write('\n\n')


        return

    print('loading model...')
    word_encoder = BERTWordEncoder(
        pretrain_ckpt, tokenizer)

    # prefix = '-'.join([model_name, opt.mode, str(N), str(K), 'seed'+str(opt.seed)])
    # if opt.dot:
    #     prefix += '-dot'
    # if len(opt.ckpt_name) > 0:
    #     prefix += '-' + opt.ckpt_name
    prefix = opt.name

    if model_name == 'proto':
        print('use proto')
        model = Proto(word_encoder, dot=opt.dot, ignore_index=opt.ignore_index)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader,
                                        use_sampled_data=opt.use_sampled_data, extra_data_loader=extra_data_loader if opt.full_test else None)
    elif model_name == 'nnshot':
        print('use nnshot')
        model = NNShot(word_encoder, dot=opt.dot,
                       ignore_index=opt.ignore_index)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader,
                                        use_sampled_data=opt.use_sampled_data, extra_data_loader=extra_data_loader if opt.full_test else None)
    elif model_name == 'structshot':
        print('use structshot')
        model = NNShot(word_encoder, dot=opt.dot,
                       ignore_index=opt.ignore_index)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, N=opt.N, tau=opt.tau, train_fname=opt.train,
                                        viterbi=True, use_sampled_data=opt.use_sampled_data, extra_data_loader=extra_data_loader if opt.full_test else None)
    elif model_name == 'container':
        print('use container')
        model = Container(word_encoder, dot=opt.dot, ignore_index=opt.ignore_index,
                          gaussian_dim=opt.proj_dim)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, N=opt.N, tau=opt.tau, train_fname=opt.train,
                                        viterbi=False, use_sampled_data=opt.use_sampled_data, contrast=True, extra_data_loader=extra_data_loader if opt.full_test else None)

    elif model_name == 'ProML':
        print('use ProML')
        model = ProML(word_encoder, dot=opt.dot, ignore_index=opt.ignore_index,
                          proj_dim=opt.proj_dim, mix_rate=opt.mix_rate, topk=opt.topk)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, N=opt.N, tau=opt.tau, train_fname=opt.train, viterbi=False,
                                        use_sampled_data=opt.use_sampled_data, contrast=True, extra_data_loader=extra_data_loader if opt.full_test else None, eval_topk=opt.topk, eval_mix_rate=opt.eval_mix_rate)

    elif model_name == 'transfer':
        print('use transfer bert')
        model = TransferBERT(word_encoder, dot=opt.dot, ignore_index=opt.ignore_index,
                             train_classes=opt.train_classes, val_classes=opt.val_classes, test_classes=opt.test_classes)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, N=opt.N,
                                        use_sampled_data=opt.use_sampled_data, contrast=False, extra_data_loader=extra_data_loader if opt.full_test else None)

    else:
        raise NotImplementedError

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt
    print('model-save-path:', ckpt)

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if opt.lr == -1:
            opt.lr = 2e-5

        framework.train(model, prefix,
                        load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                        val_step=opt.val_step, fp16=opt.fp16,
                        train_iter=opt.train_iter, warmup_step=1000, val_iter=opt.val_iter, learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert, mask_rate=opt.mask_rate, mask_id=tokenizer.mask_token_id, finetuning=True if opt.model == 'transfer' else False)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print(
                "Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    # visualize
    if opt.visualize:
        print(opt.name)
        # framework.visualize(model, 100, ckpt=ckpt,
        #                     part='train', exp_name=opt.name)
        # framework.visualize(model, 100, ckpt=ckpt,
        #                     part='val', exp_name=opt.name)
        framework.visualize(model, 100, ckpt=ckpt,
                            part='test', exp_name=opt.name)

    # test
    precision, recall, f1, fp, fn, within, outer = framework.eval(model, opt.test_iter, ckpt=ckpt, finetuning=opt.eval_with_finetune,
                                                                  finetuning_lr=3e-5, full_test=opt.full_test, finetuning_mix_rate=False, output_file=opt.output_file)
    print("RESULT: precision: %.4f, recall: %.4f, f1:%.4f" %
          (precision, recall, f1))
    print('ERROR ANALYSIS: fp: %.4f, fn: %.4f, within:%.4f, outer: %.4f' %
          (fp, fn, within, outer))


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
