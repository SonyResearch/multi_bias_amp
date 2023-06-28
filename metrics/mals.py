# Bias Amplification Metric from Men Also Like Shopping
# Code adopted from Wang and Russakovsky (https://github.com/princetonvisualai/directional-bias-amp) 

import numpy as np
import pickle
import sys 

from typing import * 

class MALS(): 
    def __init__(self, train: List, pred: List, num_groups: int):
        self.train_group = train[:, :num_groups - 1]
        self.train_attributes = train[:, num_groups - 1:]
        self.pred_group = pred[:, :num_groups - 1]
        self.pred_attributes = pred[:, num_groups - 1:]
        self.num_groups = num_groups

    def create_bog(self, attributes: List, group_membership: List[int], groups: int) -> List:
        '''
            attributes: n x |A| where n is the number of instances and |A| is the number of attributes to be classified (e.g., 361 in imSitu)
            group_membership: n x |G| where n is the number of instances and |G| is the number of groups
            groups: |G| or number of groups 

            Will return the bias scores for every attribute and group of size n x |A|
        '''
        _, obj = attributes.shape
        bogs = []
        for group in range(groups - 1):
            for g in range(2):
                group_index = np.where(np.array(group_membership[:, group]) == g)
                select_attributes = attributes[group_index]
                return_bog = []
                for o in range(obj):
                    num = np.sum(select_attributes[np.where(select_attributes[:, o] == 1)], axis=0)[o] 
                    denom = len(attributes[np.where(attributes[:, o] == 1)])
                    att_bog = num / denom
                    return_bog.append(att_bog)
                bogs.append(return_bog)
        bogs = np.vstack(bogs)
        return bogs

    def bog_mals(self, bog_tilde: List, bog_pred: List, is_abs: bool = False) -> float:
        '''
            bog_tilde: n x |A| or the bias scores of the train set
            bog_pred: n x |A| or the bias scores of the predictions

            Will return the bias amplification score as a float
        '''
        groups = self.num_groups - 1
        data_bog = bog_tilde 
        pred_bog = bog_pred 
        diff = np.zeros_like(data_bog)
        for i in range(len(data_bog)):
            for j in range(len(data_bog[0])):
                if data_bog[i][j] > (1./groups):
                    diff[i][j] = pred_bog[i][j] - data_bog[i][j]
        if is_abs: 
            value = (1./data_bog.shape[0])*(np.nansum(np.abs(diff))) # report absolute value 
        else:
            value = (1./data_bog.shape[1])*(np.nansum(diff))
        var = np.nanvar(diff)
        return value, var, np.abs(diff)

    # def get_diff(self, bog_tilde, bog_pred, bog_tilde_train=None, toprint=True):
    #     if bog_tilde_train is None:
    #         bog_tilde_train = bog_tilde
    #     data_bog = bog_tilde 
    #     pred_bog = bog_pred 
    #     diff = np.zeros_like(data_bog)
    #     for i in range(len(data_bog)):
    #         for j in range(len(data_bog[0])):
    #             if data_bog[i][j] > (1./self.num_groups):
    #                 diff[i][j] = pred_bog[i][j] - data_bog[i][j]
    #     return np.abs(diff)