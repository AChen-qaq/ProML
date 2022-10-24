import random
import pickle, os
class FewshotSampleBase:
    '''
    Abstract Class
    DO NOT USE
    Build your own Sample class and inherit from this class
    '''
    def __init__(self):
        self.class_count = {}

    def get_class_count(self):
        '''
        return a dictionary of {class_name:count} in format {any : int}
        '''
        return self.class_count


class FewshotSampler:
    '''
    sample one support set and one query set
    '''
    def __init__(self, N, K, Q, samples, classes=None, random_state=0, i2b2flag=False, dataset_name=None, no_shuffle=False):
        '''
        N: int, how many types in each set
        K: int, how many instances for each type in support set
        Q: int, how many instances for each type in query set
        samples: List[Sample], Sample class must have `get_class_count` attribute
        classes[Optional]: List[any], all unique classes in samples. If not given, the classes will be got from samples.get_class_count()
        random_state[Optional]: int, the random seed
        '''
        self.K = K
        self.N = N
        self.Q = Q
        self.samples = samples
        self.__check__() # check if samples have correct types
        if classes:
            self.classes = classes
        else:
            self.classes = self.__get_all_classes__()
        print(self.classes)
        random.seed(random_state)
        self.i2b2flag = i2b2flag
        self.name = dataset_name
        if dataset_name is not None:
            if os.path.exists(os.path.join('support_sample_cache', dataset_name + '_{}_'.format(K) + '.pkl')):
                with open(os.path.join('support_sample_cache', dataset_name + '_{}_'.format(K) + '.pkl'), 'rb') as f:
                    self.support_cache = pickle.load(f)
            else:
                self.support_cache = []
        self.global_cnt = 0
        self.no_shuffle = no_shuffle

    def __get_all_classes__(self):
        classes = []
        for sample in self.samples:
            classes += list(sample.get_class_count().keys())
        return list(set(classes))

    def __check__(self):
        for idx, sample in enumerate(self.samples):
            if not hasattr(sample,'get_class_count'):
                print('[ERROR] samples in self.samples expected to have `get_class_count` attribute, but self.samples[{idx}] does not')
                raise ValueError

    def __additem__(self, index, set_class):
        class_count = self.samples[index].get_class_count()
        for class_name in class_count:
            if class_name in set_class:
                set_class[class_name] += class_count[class_name]
            else:
                set_class[class_name] = class_count[class_name]

    def __valid_sample__(self, sample, set_class, target_classes):
        threshold = 2 * set_class['k']
        class_count = sample.get_class_count()
        if not class_count:
            return False
        isvalid = False
        for class_name in class_count:
            if class_name not in target_classes:
                return False
            if class_count[class_name] + set_class.get(class_name, 0) > threshold:
                return False
            if set_class.get(class_name, 0) < set_class['k']:
                isvalid = True
        return isvalid

    def __finish__(self, set_class):
        if len(set_class) < self.N+1:
            return False
        for k in set_class:
            if set_class[k] < set_class['k']:
                return False
        return True 

    def __get_candidates__(self, target_classes):
        return [idx for idx, sample in enumerate(self.samples) if sample.valid(target_classes)]

    def __next__(self):
        '''
        randomly sample one support set and one query set
        return:
        target_classes: List[any]
        support_idx: List[int], sample index in support set in samples list
        support_idx: List[int], sample index in query set in samples list
        '''
        target_classes = random.sample(self.classes, self.N)
        candidates = self.__get_candidates__(target_classes)
        while not candidates:
            target_classes = random.sample(self.classes, self.N)
            candidates = self.__get_candidates__(target_classes)

        if self.name in ['CoNLL2003', 'WNUT', 'I2B2', 'GUM']:
            if self.global_cnt < len(self.support_cache):
                support_idx = self.support_cache[self.global_cnt]
                query_idx = support_idx
                print(support_idx, flush=True)
            else:
                total_tries = 0
                while total_tries <= 100:
                    candidates = self.__get_candidates__(target_classes)
                    support_class = {'k':self.K}
                    support_idx = []
                    query_class = {'k':self.Q}
                    query_idx = []
                    # greedy search for support set
                    TLE_count = 0
                    while not self.__finish__(support_class) and TLE_count < 100000:
                        if TLE_count <= 10000:
                            index = random.choice(candidates)
                        else:
                            index = (index + 1) % len(candidates)
                        
                        TLE_count += 1

                        if index not in support_idx:
                            if self.__valid_sample__(self.samples[index], support_class, target_classes):
                                self.__additem__(index, support_class)
                                support_idx.append(index)
                                TLE_count = 0
                    
                    if self.__finish__(support_class):
                        break
                    else:
                        total_tries += 1
                
                query_idx = support_idx

            
                self.support_cache.append(support_idx)
                with open(os.path.join('support_sample_cache', self.name + '_{}_'.format(self.K) + '.pkl'), 'wb') as f:
                    pickle.dump(self.support_cache, f)
            # assert self.__finish__(support_class)
            self.global_cnt += 1
            return target_classes if not self.no_shuffle else self.classes, support_idx, query_idx


        support_class = {'k':self.K}
        support_idx = []
        query_class = {'k':self.Q}
        query_idx = []
        target_classes = random.sample(self.classes, self.N)
        candidates = self.__get_candidates__(target_classes)
        while not candidates:
            target_classes = random.sample(self.classes, self.N)
            candidates = self.__get_candidates__(target_classes)

        # greedy search for support set
        while not self.__finish__(support_class):
            index = random.choice(candidates)
            if index not in support_idx:
                if self.__valid_sample__(self.samples[index], support_class, target_classes):
                    self.__additem__(index, support_class)
                    support_idx.append(index)
        # same for query set
        while not self.__finish__(query_class):
            index = random.choice(candidates)
            if index not in query_idx and index not in support_idx:
                if self.__valid_sample__(self.samples[index], query_class, target_classes):
                    self.__additem__(index, query_class)
                    query_idx.append(index)
                                
        return target_classes if not self.no_shuffle else self.classes, support_idx, query_idx

    def __iter__(self):
        return self
    
