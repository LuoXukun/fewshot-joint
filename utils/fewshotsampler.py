#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Date:     2021.10.21

""" 
    Add based on the source code of Few-NERD. 
    Few-NERD: A Few-Shot Named Entity Recognition Dataset. Ding et al. ACL 2021.
"""

import random

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
    def __init__(self, N, K, Q, samples, classes=None, random_state=0):
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
        random.seed(random_state)

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
        # We do not want 1~2 shot when K = 1.
        #threshold = int(1.5 * float(set_class['k']))
        class_count = sample.get_class_count()
        if not class_count:
            return False
        
        # The origin version is wrong. Rewrite it based on the paper.
        """ isvalid = False
        for class_name in class_count:
            if class_name not in target_classes:
                isvalid = False
            elif class_name not in set_class:
                isvalid = True
            elif set_class[class_name] + class_count[class_name] > threshold:
                isvalid = False
            elif set_class[class_name] < set_class['k']:
                isvalid = True """
        isvalid = True
        for class_name in class_count:
            if class_name not in target_classes: continue
            if class_name not in set_class: continue
            if set_class[class_name] + class_count[class_name] > threshold:
                isvalid = False
                break
        """ print("isvalid: ", isvalid)
        if isvalid:
            print(sample) """
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
        support_class = {'k':self.K}
        support_idx = []
        query_class = {'k':self.Q}
        query_idx = []
        target_classes = random.sample(self.classes, self.N)
        #print("target_classes: ", target_classes)

        candidates = self.__get_candidates__(target_classes)
        while not candidates:
            target_classes = random.sample(self.classes, self.N)
            candidates = self.__get_candidates__(target_classes)
        #print("Candidate len:", len(candidates))

        # greedy search for support set
        support_steps = 0
        while not self.__finish__(support_class):
            index = random.choice(candidates)
            if index not in support_idx:
                if self.__valid_sample__(self.samples[index], support_class, target_classes):
                    self.__additem__(index, support_class)
                    support_idx.append(index)
            support_steps += 1
            if support_steps >= 10000:
                #print("support steps: ", support_steps)
                return self.__next__()
        #print("support steps: ", support_steps)
        #print("support_class: ", support_class)

        # same for query set
        query_steps = 0
        while not self.__finish__(query_class):
            index = random.choice(candidates)
            if index not in query_idx and index not in support_idx:
                if self.__valid_sample__(self.samples[index], query_class, target_classes):
                    self.__additem__(index, query_class)
                    query_idx.append(index)
            query_steps += 1
            if query_steps >= 10000:
                #print("query steps: ", query_steps)
                return self.__next__()
        #print("query steps: ", query_steps)
        #print("query_class: ", query_class)

        return target_classes, support_idx, query_idx

    def __iter__(self):
        return self
    