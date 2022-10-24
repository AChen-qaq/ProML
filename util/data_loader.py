from html import entities
from random import random
import torch
import torch.utils.data as data
import os
from .fewshotsampler import FewshotSampler, FewshotSampleBase
import numpy as np
import json

def get_class_name(rawtag):
    # get (finegrained) class name
    if rawtag.startswith('B-') or rawtag.startswith('I-'):
        return rawtag[2:]
    else:
        return rawtag

class Sample(FewshotSampleBase):
    def __init__(self, filelines):
        filelines = [line.split('\t') for line in filelines]
        # print(filelines)
        self.words, self.tags = zip(*filelines)
        self.words = [word.lower() for word in self.words]
        # strip B-, I-
        self.normalized_tags = list(map(get_class_name, self.tags))
        self.class_count = {}

    def __count_entities__(self):
        current_tag = self.normalized_tags[0]
        for tag in self.normalized_tags[1:]:
            if tag == current_tag:
                continue
            else:
                if current_tag != 'O':
                    if current_tag in self.class_count:
                        self.class_count[current_tag] += 1
                    else:
                        self.class_count[current_tag] = 1
                current_tag = tag
        if current_tag != 'O':
            if current_tag in self.class_count:
                self.class_count[current_tag] += 1
            else:
                self.class_count[current_tag] = 1

    def get_class_count(self):
        if self.class_count:
            return self.class_count
        else:
            self.__count_entities__()
            return self.class_count

    def get_tag_class(self):
        # strip 'B' 'I' 
        tag_class = list(set(self.normalized_tags))
        if 'O' in tag_class:
            tag_class.remove('O')
        return tag_class

    def valid(self, target_classes):
        return (set(self.get_class_count().keys()).intersection(set(target_classes))) and not (set(self.get_class_count().keys()).difference(set(target_classes)))

    def __str__(self):
        newlines = zip(self.words, self.tags)
        return '\n'.join(['\t'.join(line) for line in newlines])

class FewShotNERDatasetWithRandomSampling(data.Dataset):
    """
    Fewshot NER Dataset
    """
    def __init__(self, filepath, tokenizer, N, K, Q, max_length, ignore_label_id=-1, i2b2flag=False, dataset_name=None, no_shuffle=False):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.class2sampleid = {}
        self.N = N
        self.K = K
        self.Q = Q
        self.tokenizer = tokenizer
        self.samples, self.classes = self.__load_data_from_file__(filepath)
        self.max_length = max_length
        self.sampler = FewshotSampler(N, K, Q, self.samples, classes=self.classes, i2b2flag=i2b2flag, dataset_name=dataset_name, no_shuffle=no_shuffle)
        self.ignore_label_id = ignore_label_id

        print(filepath, len(self.classes), self.classes, flush=True)

    def __insert_sample__(self, index, sample_classes):
        for item in sample_classes:
            if item in self.class2sampleid:
                self.class2sampleid[item].append(index)
            else:
                self.class2sampleid[item] = [index]
    
    def __load_data_from_file__(self, filepath):
        samples = []
        classes = []
        with open(filepath, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        samplelines = []
        index = 0
        for line in lines:
            line = line.strip()
            if line:
                samplelines.append(line)
            else:
                sample = Sample(samplelines)
                samples.append(sample)
                sample_classes = sample.get_tag_class()
                self.__insert_sample__(index, sample_classes)
                classes += sample_classes
                samplelines = []
                index += 1
        classes = list(set(classes))
        return samples, classes

    def __get_token_label_list__(self, sample):
        tokens = []
        labels = []
        for word, tag in zip(sample.words, sample.normalized_tags):
            word_tokens = self.tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                word_labels = [self.tag2label[tag]] + [self.ignore_label_id] * (len(word_tokens) - 1)
                labels.extend(word_labels)
        return tokens, labels

    def construct_prompt_tanl(self, tokens, labels):
        from random import randint
        opt = randint(0, 0)
        # token -> ids
        # tokens_tanl = ['support', 'cases', '(', 'type', 'A' if opt == 0 else 'B', ')', ':']
        # text_mask_tanl = [0, 0, 0, 0, 0, 0, 0]

        tokens_tanl = []
        text_mask_tanl = []

        label_tanl = []
        lst_tag = None

        

        if opt == 0:

            for token, tag in zip(tokens, labels):
                if lst_tag is not None and lst_tag != tag and lst_tag > 0:
                    # label_seq = self.label2tag[lst_tag].split('-')[-1]
                    label_seq = self.label2tag[lst_tag].replace('-', ' - ')
                    label_seq = label_seq.replace('/', ' / ')
                    tokens_tanl.extend(['|']+label_seq.split()+[']'])
                    text_mask_tanl.extend([0]+[-2]*len(label_seq.split())+[0])
                    label_tanl.extend([tag] * (len(label_seq.split()) + 2))
                if tag > 0 and (lst_tag is None or lst_tag != tag):
                    tokens_tanl.extend(['['])
                    text_mask_tanl.extend([0])
                    label_tanl.extend([tag])
                
                tokens_tanl.extend([ token ])
                text_mask_tanl.extend([ 1 ])
                label_tanl.extend([ tag ])
                lst_tag = tag

            if lst_tag is not None and lst_tag > 0:
                # label_seq = self.label2tag[lst_tag].split('-')[-1]
                label_seq = self.label2tag[lst_tag].replace('-', ' - ')
                label_seq = label_seq.replace('/', ' / ')
                tokens_tanl.extend(['|']+label_seq.split()+[']'])
                text_mask_tanl.extend([0]+[-2]*len(label_seq.split())+[0])
                label_tanl.extend([lst_tag] * (len(label_seq.split()) + 2))

        # if opt == 1:
        #     for token, tag in zip(tokens, labels):
        #         if lst_tag is not None and lst_tag != tag and lst_tag > 0:
        #             # label_seq = self.label2tag[lst_tag].split('-')[-1]
        #             label_seq = self.label2tag[lst_tag].replace('-', ' - ')
        #             label_seq = label_seq.replace('/', ' / ')
        #             tokens_tanl.extend(['(']+label_seq.split()+[')'])
        #             text_mask_tanl.extend([0]+[-2]*len(label_seq.split())+[0])
        #             label_tanl.extend([tag] * (len(label_seq.split()) + 2))

                
        #         tokens_tanl.extend([ token ])
        #         text_mask_tanl.extend([ 1 ])
        #         label_tanl.extend([ tag ])
        #         lst_tag = tag

        #     if lst_tag is not None and lst_tag > 0:
        #         label_seq = self.label2tag[lst_tag].split('-')[-1]
        #         label_seq = label_seq.replace('/', ' / ')
        #         tokens_tanl.extend(['(']+label_seq.split()+[')'])
        #         text_mask_tanl.extend([0]+[-2]*len(label_seq.split())+[0])
        #         label_tanl.extend([lst_tag] * (len(label_seq.split()) + 2))
        

        
        return tokens_tanl, text_mask_tanl, label_tanl

    def construct_prompt_aug(self, tokens, labels):
        from random import randint
        # if torch.max(torch.tensor(labels).long()).item() < 0:
        #     return [], [], []
        # opt = randint(0, torch.max(torch.tensor(labels).long()).item())
        # token -> ids

        # entities = [self.label2tag[opt].split('-')[-1].replace('/', ' / ') for opt in range(torch.max(torch.tensor(labels)) + 1)]
        entities = [self.label2tag[opt].replace('-', ' - ').replace('/', ' / ') for opt in range(torch.max(torch.tensor(labels)) + 1)]
        
        
        # print(entities)
        # tokens_aug = 'extract {label} entities for cases :'.format(label=' and '.join(entities)).split()
        tokens_aug = ' , '.join(entities).split() + [':']
        text_mask_aug = [0] * len(tokens_aug)
        label_aug = labels
        
        tokens_aug.extend(tokens)
        text_mask_aug.extend([1] * len(tokens))
        # for token, tag in zip(tokens, labels):
        #     if tag == opt:
        #         label_aug.append(opt)
        #         text_mask_aug.append(1)
        #     else:
        #         text_mask_aug.append(0)
        
        return tokens_aug, text_mask_aug, label_aug
    

    def __getraw__(self, tokens, labels):
        # get tokenized word list, attention mask, text mask (mask [CLS], [SEP] as well), tags
        
        # split into chunks of length (max_length-2)
        # 2 is for special tokens [CLS] and [SEP]
        tokens_list = []
        labels_list = []
        while len(tokens) > self.max_length - 2:
            tokens_list.append(tokens[:self.max_length-2])
            tokens = tokens[self.max_length-2:]
            labels_list.append(labels[:self.max_length-2])
            labels = labels[self.max_length-2:]
        if tokens:
            tokens_list.append(tokens)
            labels_list.append(labels)

        # add special tokens and get masks
        indexed_tokens_list = []
        mask_list = []
        text_mask_list = []
        token_mask_list = []

        indexed_tokens_list_tanl = []
        text_mask_list_tanl = []
        label_list_tanl = []

        indexed_tokens_list_aug = []
        text_mask_list_aug = []
        label_list_aug = []
        for i, tokens in enumerate(tokens_list):

            


            tokens_tanl, text_mask_tanl, label_tanl = self.construct_prompt_tanl(tokens, labels_list[i])
            tokens_aug, text_mask_aug, label_aug = self.construct_prompt_aug(tokens, labels_list[i])


            
            tokens_tanl = ['[CLS]'] + tokens_tanl + ['[SEP]']
            tokens_aug = ['CLS'] + tokens_aug + ['[SEP]']
            
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            
            text_mask_tanl = [0] + text_mask_tanl + [0]
            text_mask_aug = [0] + text_mask_aug + [0]

            indexed_tokens_list_tanl.append(self.tokenizer.convert_tokens_to_ids(tokens_tanl))
            indexed_tokens_list_aug.append(self.tokenizer.convert_tokens_to_ids(tokens_aug))
            text_mask_list_tanl.append(text_mask_tanl)
            text_mask_list_aug.append(text_mask_aug)

            label_list_tanl.append(label_tanl)
            label_list_aug.append(label_aug)

            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
            # padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens_list.append(indexed_tokens)

            # mask
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(tokens)] = 1
            mask_list.append(mask)

            # text mask, also mask [CLS] and [SEP]
            text_mask = np.zeros((self.max_length), dtype=np.int32)
            text_mask[1:len(tokens)-1] = 1

            token_mask = np.zeros((self.max_length), dtype=np.int32)
            token_mask[1:len(tokens)-1] = 1

            for j,t in enumerate(labels_list[i]):
                if t != 0:
                    token_mask[1+j] = 0




            # text_mask[labels_list[i] == 0] = 0
            text_mask_list.append(text_mask)
            token_mask_list.append(token_mask)


            assert len(labels_list[i]) == len(tokens) - 2, print(labels_list[i], tokens)
        return indexed_tokens_list, mask_list, text_mask_list, labels_list, token_mask_list, indexed_tokens_list_tanl, text_mask_list_tanl, label_list_tanl, indexed_tokens_list_aug, text_mask_list_aug, label_list_aug

    def __additem__(self, index, d, word, mask, text_mask, label, token_mask, word_tanl, text_mask_tanl, label_tanl, word_aug, text_mask_aug, label_aug):
        d['index'].append(index)
        d['word'] += word
        d['mask'] += mask
        d['label'] += label
        d['text_mask'] += text_mask
        d['token_mask'] += token_mask
        d['word_tanl'] += word_tanl
        d['text_mask_tanl'] += text_mask_tanl
        d['label_tanl'] += label_tanl
        d['word_aug'] += word_aug
        d['text_mask_aug'] += text_mask_aug
        d['label_aug'] += label_aug


    def __populate__(self, idx_list, savelabeldic=False):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'index': sample_index
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'sentence_num': number of sentences in this set (a batch contains multiple sets)
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {'index':[], 'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'text_mask':[], 'token_mask':[], 'word_tanl':[], 'text_mask_tanl':[], 'label_tanl':[], 'word_aug': [], 'text_mask_aug': [], 'label_aug': [] }
        for idx in idx_list:
            tokens, labels = self.__get_token_label_list__(self.samples[idx])
            word, mask, text_mask, label, token_mask, word_tanl, text_mask_tanl, label_tanl, word_aug, text_mask_aug, label_aug = self.__getraw__(tokens, labels)
            word = torch.tensor(word).long()
            mask = torch.tensor(mask).long()

            text_mask = torch.tensor(text_mask).long()
            token_mask = torch.tensor(token_mask).long()

            word_tanl = [torch.tensor(x).long() for x in word_tanl]
            text_mask_tanl = [torch.tensor(x).long() for x in text_mask_tanl]

            word_aug = [torch.tensor(x).long() for x in word_aug]
            text_mask_aug = [torch.tensor(x).long() for x in text_mask_aug]

            self.__additem__(idx, dataset, word, mask, text_mask, label, token_mask, word_tanl, text_mask_tanl, label_tanl, word_aug, text_mask_aug, label_aug)
        dataset['sentence_num'] = [len(dataset['word'])]
        if savelabeldic:
            dataset['label2tag'] = [self.label2tag]
            dataset['tag2label'] = [self.tag2label]
        return dataset

    def __getitem__(self, index):
        target_classes, support_idx, query_idx = self.sampler.__next__()
        # add 'O' and make sure 'O' is labeled 0
        distinct_tags = ['O'] + target_classes
        self.tag2label = {tag:idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx:tag for idx, tag in enumerate(distinct_tags)}

        # print(self.tag2label,flush=True)
        # print(self.label2tag,flush=True)
        # exit(0)
        support_set = self.__populate__(support_idx)
        query_set = self.__populate__(query_idx, savelabeldic=True)
        return support_set, query_set
    
    def __len__(self):
        return 100000

class EasySampler(FewShotNERDatasetWithRandomSampling):
    def __init__(self, filepath, tokenizer, N, K, Q, max_length, ignore_label_id=-1):
       super(EasySampler, self).__init__(filepath, tokenizer, N, K, Q, max_length, ignore_label_id=-1)
    
    def set_label(self, tag2label=None, label2tag=None):
       self.tag2label = tag2label
       self.label2tag = label2tag
    def __getitem__(self, index):
        query_idx = list(range(index * 16, (index + 1) * 16))
        target_classes = list(set([ x for idx in query_idx for x in self.samples[idx].get_class_count().keys() ]))
        # target_classes, support_idx, query_idx = self.sampler.__next__()
        # add 'O' and make sure 'O' is labeled 0
        distinct_tags = ['O'] + target_classes
        # self.tag2label = {tag:idx for idx, tag in enumerate(distinct_tags)}
        # self.label2tag = {idx:tag for idx, tag in enumerate(distinct_tags)}
        
        # support_set = self.__populate__(support_idx)
        query_set = self.__populate__(query_idx, savelabeldic=True)
        return query_set

    def __len__(self):
        print('hello', len(self.samples))
        return len(self.samples)//16

class FewShotNERDataset(FewShotNERDatasetWithRandomSampling):
    def __init__(self, filepath, tokenizer, max_length, ignore_label_id=-1, no_shuffle=False):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.class2sampleid = {}
        self.tokenizer = tokenizer
        self.samples, self.classes = self.__load_data_from_file__(filepath)
        self.max_length = max_length
        self.ignore_label_id = ignore_label_id
        self.no_shuffle = no_shuffle
    
    def __load_data_from_file__(self, filepath):
        with open(filepath)as f:
            lines = f.readlines()
        classes = []
        for i in range(len(lines)):
            lines[i] = json.loads(lines[i].strip())
            classes += lines[i]['types']

        classes = list(set(classes))
        return lines, classes
    
    def __additem__(self, index, d, word, mask, text_mask, label, token_mask, word_tanl, text_mask_tanl, label_tanl, word_aug, text_mask_aug, label_aug):
        d['index'].append(index)
        d['word'] += word
        d['mask'] += mask
        d['label'] += label
        d['text_mask'] += text_mask
        d['token_mask'] += token_mask
        d['word_tanl'] += word_tanl
        d['text_mask_tanl'] += text_mask_tanl
        d['label_tanl'] += label_tanl
        d['word_aug'] += word_aug
        d['text_mask_aug'] += text_mask_aug
        d['label_aug'] += label_aug
    
    def __get_token_label_list__(self, words, tags):
        tokens = []
        labels = []
        for word, tag in zip(words, tags):
            word_tokens = self.tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                word_labels = [self.tag2label[tag]] + [self.ignore_label_id] * (len(word_tokens) - 1)
                labels.extend(word_labels)
        return tokens, labels

    def __populate__(self, data, savelabeldic=False):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'sentence_num': number of sentences in this set (a batch contains multiple sets)
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {'index':[], 'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'text_mask':[], 'token_mask':[], 'word_tanl':[], 'text_mask_tanl':[], 'label_tanl':[], 'word_aug': [], 'text_mask_aug': [], 'label_aug': [] }
        for i in range(len(data['word'])):
            tokens, labels = self.__get_token_label_list__(data['word'][i], data['label'][i])
            word, mask, text_mask, label, token_mask, word_tanl, text_mask_tanl, label_tanl, word_aug, text_mask_aug, label_aug = self.__getraw__(tokens, labels)
            word = torch.tensor(word).long()
            mask = torch.tensor(mask).long()
            text_mask = torch.tensor(text_mask).long()
            token_mask = torch.tensor(token_mask).long()

            word_tanl = [torch.tensor(x).long() for x in word_tanl]
            text_mask_tanl = [torch.tensor(x).long() for x in text_mask_tanl]

            word_aug = [torch.tensor(x).long() for x in word_aug]
            text_mask_aug = [torch.tensor(x).long() for x in text_mask_aug]

            self.__additem__(i, dataset, word, mask, text_mask, label, token_mask, word_tanl, text_mask_tanl, label_tanl, word_aug, text_mask_aug, label_aug)
        dataset['sentence_num'] = [len(dataset['word'])]
        if savelabeldic:
            dataset['label2tag'] = [self.label2tag]
            dataset['tag2label'] = [self.tag2label]
        return dataset

    def __getitem__(self, index):
        sample = self.samples[index]
        target_classes = self.classes if self.no_shuffle else sample['types']
        support = sample['support']
        query = sample['query']
        # add 'O' and make sure 'O' is labeled 0
        distinct_tags = ['O'] + target_classes
        self.tag2label = {tag:idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx:tag for idx, tag in enumerate(distinct_tags)}
        support_set = self.__populate__(support)
        query_set = self.__populate__(query, savelabeldic=True)
        return support_set, query_set

    def __len__(self):
        return len(self.samples)


def collate_fn(data):
    batch_support = {'index': [], 'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'text_mask':[], 'token_mask':[], 'word_tanl': [], 'text_mask_tanl': [], 'label_tanl': [], 'word_aug': [], 'text_mask_aug': [], 'label_aug': []}
    batch_query = {'index': [], 'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'label2tag':[], 'tag2label':[], 'text_mask':[], 'token_mask':[], 'word_tanl': [], 'text_mask_tanl': [], 'label_tanl': [], 'word_aug': [], 'text_mask_aug': [], 'label_aug': []}
    support_sets, query_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in batch_support:
            batch_support[k] += support_sets[i][k]
        for k in batch_query:
            batch_query[k] += query_sets[i][k]
    

    for k in batch_support:
        if k in ['word', 'mask', 'text_mask', 'token_mask']:
            batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        if k in ['word', 'mask', 'text_mask', 'token_mask']:
            batch_query[k] = torch.stack(batch_query[k], 0)
    
    batch_support['label'] = [torch.tensor(tag_list).long() for tag_list in batch_support['label']]
    batch_query['label'] = [torch.tensor(tag_list).long() for tag_list in batch_query['label']]

    batch_support['label_tanl'] = [torch.tensor(tag_list).long() for tag_list in batch_support['label_tanl']]
    batch_query['label_tanl'] = [torch.tensor(tag_list).long() for tag_list in batch_query['label_tanl']]

    batch_support['label_aug'] = [torch.tensor(tag_list).long() for tag_list in batch_support['label_aug']]
    batch_query['label_aug'] = [torch.tensor(tag_list).long() for tag_list in batch_query['label_aug']]

    return batch_support, batch_query

def collate_fn_easy(data):
    batch_query = {'index': [], 'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'label2tag':[], 'text_mask':[], 'token_mask':[], 'word_tanl': [], 'text_mask_tanl': [], 'label_tanl': [], 'word_aug': [], 'text_mask_aug': [], 'label_aug': []}
    query_sets = data
    for i in range(len(query_sets)):
        for k in batch_query:
            batch_query[k] += query_sets[i][k]
    
    for k in batch_query:
        if k in ['word', 'mask', 'text_mask', 'token_mask']:
            batch_query[k] = torch.stack(batch_query[k], 0)
    
    batch_query['label'] = [torch.tensor(tag_list).long() for tag_list in batch_query['label']]
    batch_query['label_tanl'] = [torch.tensor(tag_list).long() for tag_list in batch_query['label_tanl']]
    batch_query['label_aug'] = [torch.tensor(tag_list).long() for tag_list in batch_query['label_aug']]

    return batch_query


def get_loader(filepath, tokenizer, N, K, Q, batch_size, max_length, 
        num_workers=8, collate_fn=collate_fn, ignore_index=-1, use_sampled_data=True, full_test=False, i2b2flag=False, dataset_name=None, no_shuffle=False):
    assert (not use_sampled_data) or (not full_test)
    if full_test:
        dataset = EasySampler(filepath, tokenizer, N, K, Q, max_length, ignore_label_id=ignore_index)
    elif not use_sampled_data:
        dataset = FewShotNERDatasetWithRandomSampling(filepath, tokenizer, N, K, Q, max_length, ignore_label_id=ignore_index, i2b2flag=i2b2flag, dataset_name=dataset_name, no_shuffle=no_shuffle)
    else:
        dataset = FewShotNERDataset(filepath, tokenizer, max_length, ignore_label_id=ignore_index, no_shuffle=no_shuffle)
    
    def create_loader(dataset):
        data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn if not full_test else collate_fn_easy)
        return data_loader
    if dataset_name in ['CoNLL2003', 'WNUT', 'I2B2', 'GUM']:
        num_workers=1
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn if not full_test else collate_fn_easy)
    return data_loader if not full_test else (create_loader, dataset)