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
                token, label = line.split()[0], line.split()[-1]
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
                if len(sentence) == 0:
                    continue
                f.writelines('\n'.join(['\t'.join([sample[0], self.re_tag(sample[1])]) for sample in sentence]))
                f.writelines('\n\n')
            



if __name__ == "__main__":
    testtags = {'AGE': 'age',
'BIOID': 'biometric ID',
'CITY': 'city',
'COUNTRY': 'country',
'DATE': 'date',
'DEVICE': 'device',
'DOCTOR': 'doctor',
'EMAIL': 'email',
'FAX': 'fax',
'HEALTHPLAN': 'health plan number',
'HOSPITAL': 'hospital',
'IDNUM': 'ID number',
'LOCATION_OTHER': 'location',
'MEDICALRECORD': 'medical record',
'ORGANIZATION': 'organization',
'PATIENT': 'patient',
'PHONE': 'phone number',
'PROFESSION' :'profession',
'STATE' :'state',
'STREET' :'street',
'URL': 'url',
'USERNAME' : 'username',
'ZIP': 'zip code'}
    testset = CoNLLDataset('./data/i2b2_raw.txt', testtags)
    
    testset.re_tag_and_write('./data/test-i2b2.txt')

    #print(list(trainset.labels.keys()))
    #print(len(list(trainset.labels.keys())))
    #print(list(valset.labels.items()))
    #print(list(testset.labels.items()))
