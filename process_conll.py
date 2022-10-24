import os
import numpy as np

class CoNLLDataset(object):
    labels = {}
    sentences = []
    tagset = {}
    def __init__(self, path, tagset):
        self.tagset = tagset
        with open(path) as f:
            lines = f.readlines()
        tmp = []
        for line in lines:
            line = line.strip()
            if line:
                if line.startswith('-DOCSTART-'):
                    continue
                token, label = line.split()[1], line.split()[-1]
                if label.startswith('B-') or label.startswith('I-'):
                    label = label[2:]
                tmp.append((token, label))
            else:
                self.sentences.append(tmp)
                tmp = []
    def re_tag(self, tag):
        return 'GATE-' + self.tagset.get(tag) if tag in self.tagset else 'O'
    def re_tag_and_write(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for sentence in self.sentences:
                f.writelines('\n'.join(['\t'.join([sample[0], self.re_tag(sample[1])]) for sample in sentence]))
                f.writelines('\n\n')
            



if __name__ == "__main__":
    A, B, C = {'ORG':'organization', 'NORP':'nationality religion political', 'ORDINAL':'ordinal', 'WORK_OF_ART':'work of art', 'QUANTITY':'quantity', 'LAW':'law'}, {'GPE':'geographical social political entity', 'CARDINAL':'cardinal', 'PERCENT':'percent', 'TIME':'time', 'EVENT':'event', 'LANGUAGE':'language'}, {'PERSON':'person', 'DATE':'date', 'MONEY':'money', 'LOC':'location', 'FAC':'facility', 'PRODUCT':'product'}

    # _A, _B, _C = {'ORG', 'NORP', 'ORDINAL', 'WORK_OF_ART', 'LAW'}, {'GPE', 'EVENT', 'LANGUAGE'}, {'PERSON', 'DATE', 'LOC', 'FAC', 'PRODUCT'}
    traintags = {**A, **B}
    #valtags = {'GPE', 'CARDINAL', 'PERCENT', 'TIME', 'EVENT', 'LANGUAGE'}
    #valtags = traintags
    testtags = {**C}
    valtags = {**C}
    trainset, valset, testset = CoNLLDataset('data/ontoNotes/train.sd.conllx', traintags), CoNLLDataset('data/ontoNotes/dev.sd.conllx', valtags), CoNLLDataset('data/ontoNotes/test.sd.conllx', testtags)
    
    trainset.re_tag_and_write('data/ontoNotes/__train_C.txt')
    valset.re_tag_and_write('data/ontoNotes/__dev_C.txt')
    testset.re_tag_and_write('data/ontoNotes/__test_C.txt')

    #print(list(trainset.labels.keys()))
    #print(len(list(trainset.labels.keys())))
    #print(list(valset.labels.items()))
    #print(list(testset.labels.items()))
