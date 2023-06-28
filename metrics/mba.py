# Metric code for multi-attribute bias amplification

import numpy as np
import pickle
import sys

sys.path.append("../")
from utils import threshold_preds, conf_interval

from typing import *
from tqdm import tqdm


class MBA:
    def __init__(
        self,
        train_atts: List[int],
        train_group: List[int],
        preds: Dict,
        thresh: int = 0,
        num_groups: int = 2,
        k: int = 1,
        calib: bool = True,
    ):
        self.num_groups = num_groups
        self.k = k

        # only include images that have attribute(s) and task(s) associated with it for calculation of indicator variable
        keep_indices = np.array(
            list(
                set(np.where(np.sum(train_atts, axis=1) > 0)[0]).union(
                    set(np.where(np.sum(train_group, axis=1) > 0)[0])
                )
            )
        )
        self.train_atts = train_atts[keep_indices]
        self.train_group = train_group[keep_indices]

        if calib:
            scores = threshold_preds(preds)
        else:
            scores = np.round(preds["scores"])

        self.pred_atts = scores[:, num_groups - 1 :]
        self.pred_group = scores[:, : num_groups - 1]
        self.pred_labels = preds["labels"]
        self.pred_task_labels = self.pred_labels[:, num_groups - 1 :]

        self.multi_atts = self.get_multiatts(thresh)

    def get_multiatts(self, thresh: int) -> List:
        """
        thresh: Limits the number of times the group of multi-attributes must occur
        Returns all the groups of multi-attributes which occur in both the ground-truth train and test set
        """
        multi_atts_train = set()
        multi_atts_train_count = {}
        multi_atts = set()

        for instance in self.train_atts:
            atts = np.where(instance == 1)[0]
            if len(atts) >= self.k:
                if tuple(atts) not in multi_atts_train_count:
                    multi_atts_train_count[tuple(atts)] = 0
                multi_atts_train_count[tuple(atts)] += 1
                if multi_atts_train_count[tuple(atts)] > thresh:
                    multi_atts_train.add(tuple(atts))
        for atts in multi_atts_train:
            select_preds = self.pred_task_labels[:, atts]
            indices = np.where(np.sum(select_preds, axis=1) == len(atts))[0]
            if len(indices) > thresh:
                multi_atts.add(tuple(atts))
        return list(multi_atts)

    def create_bog(self, atts_list: List, group: List) -> List:
        def select_indices(att_list: List, att: List) -> List:
            subset = att_list[:, att]
            sum_atts = np.sum(subset, axis=1)
            indices = np.where(sum_atts == len(att))
            return indices

        num_attributes = len(self.multi_atts)
        bog = np.zeros((num_attributes, self.num_groups - 1))
        if group.shape[1] == 1:
            group = group.flatten()
        else:
            group = np.argmax(group, axis=1)

        for index, atts in enumerate(tqdm(self.multi_atts)):
            total_indices = select_indices(atts_list, atts)
            total = len(total_indices[0])
            select_atts = atts_list[total_indices]
            group_atts = group[total_indices]

            for g in range(self.num_groups - 1):
                select = select_atts[np.where(group_atts == g)]
                try:
                    bog[index][g] = len(select) / total
                except:
                    bog[index][g] = None

        return bog

    def calculate_mba_mals(self, is_abs: bool) -> Tuple:
        """
        Overview: Calculate undirected multi-attribute bias amplification '

        Input:
            - is_abs: whether to take absolute value of the differences

        Output: 
            - value: Mean over all differences
            - var: Variance of the deltas

        """
        groups = self.num_groups - 1
        bog_tilde = self.create_bog(self.train_atts, self.train_group)
        bog_pred = self.create_bog(self.pred_atts, self.pred_group)
        data_bog = bog_tilde
        pred_bog = bog_pred

        diff = np.zeros_like(data_bog)
        for i in range(len(data_bog)):
            for j in range(len(data_bog[0])):
                if data_bog[i][j] > (1.0 / groups):
                    diff[i][j] = pred_bog[i][j] - data_bog[i][j]
        if is_abs:
            value = (1.0 / data_bog.shape[0]) * (
                np.nansum(np.abs(diff))
            )  
        else:
            value = (1.0 / data_bog.shape[0]) * (np.nansum(diff))
        var = np.nanvar(diff)
        return value, var

    def calculate_mba_gm(self, is_abs: bool = False) -> Tuple:
        """
        Overview: Calculate directed multi-attribute bias amplification from
        group to multi-attribute
        
        Input:
            - is_abs: whether to take absolute value of the differences

        Output: 
            - val: Mean over all differences
            - var: Variance of the deltas

        """

        # y_gm calculation
        groups = self.num_groups - 1
        multi_att_num = len(self.multi_atts)
        p_at = np.zeros((groups, multi_att_num))
        p_a_p_t = np.zeros((groups, multi_att_num))
        num_train = len(self.train_atts)

        # delta_mg calculation
        t_cond_a = np.zeros((groups, multi_att_num))
        that_cond_a = np.zeros((groups, multi_att_num))

        attribute_labels = self.pred_labels[:, groups:]
        group_labels = self.pred_labels[:, :groups]

        for a in tqdm(range(groups)):
            temp_labels = attribute_labels[np.where(group_labels[:, a] == 1)[0]]
            temp_labels_size = len(temp_labels)
            temp_preds = self.pred_atts[np.where(group_labels[:, a] == 1)[0]]
            temp_preds_size = len(temp_preds)

            a_indices = np.where(self.train_group[:, a] == 1)[0]

            for i, t in enumerate(tqdm(self.multi_atts)):
                t_indices = np.where(np.sum(self.train_atts[:, t], axis=1) == len(t))[0]

                at_indices = set(t_indices) & set(a_indices)
                p_a_p_t[a][i] = (len(t_indices) / num_train) * (
                    len(a_indices) / num_train
                )
                p_at[a][i] = len(at_indices) / num_train

                t_cond_a[a][i] = np.mean(
                    attribute_labels[:, t][np.where(group_labels[:, a] == 1)[0]]
                )

                total = len(np.where(np.sum(temp_labels[:, t], axis=1) == len(t))[0])
                t_cond_a[a][i] = total / temp_labels_size

                total = len(np.where(np.sum(temp_preds[:, t], axis=1) == len(t))[0])
                that_cond_a[a][i] = total / temp_preds_size

        y_at = np.sign(p_at - p_a_p_t)

        delta_at = that_cond_a - t_cond_a

        if is_abs:
            values = np.abs(delta_at)
        else:
            values = y_at * delta_at
        val = np.nanmean(values)
        var = np.nanvar(delta_at)
        return val, var

    def calculate_mba_mg(self, is_abs: bool) -> Tuple:
        """
        Overview: Calculate directed multi-attribute bias amplification from
        multi-attribute to group
        
        Input:
            - is_abs: whether to take absolute value of the differences

        Output: 
            - val: Mean over all differences
            - var: Variance of the deltas

        """

        # y_gm calculation
        groups = self.num_groups - 1
        multi_att_num = len(self.multi_atts)
        p_at = np.zeros((groups, multi_att_num))
        p_a_p_t = np.zeros((groups, multi_att_num))
        num_train = len(self.train_atts)

        # delta_mg calculation
        a_cond_t = np.zeros((groups, multi_att_num))
        ahat_cond_t = np.zeros((groups, multi_att_num))

        task_labels = self.pred_labels[:, groups:]
        group_labels = self.pred_labels[:, :groups]

        for i, t in enumerate(tqdm(self.multi_atts)):
            temp_labels = group_labels[
                np.where(np.sum(task_labels[:, t], axis=1) == len(t))
            ]
            temp_labels_size = len(temp_labels)

            temp_preds = self.pred_group[
                np.where(np.sum(task_labels[:, t], axis=1) == len(t))
            ]
            temp_preds_size = len(temp_preds)

            t_indices = np.where(np.sum(self.train_atts[:, t], axis=1) == len(t))[0]

            for a in range(groups):
                a_indices = np.where(self.train_group[:, a] == 1)[0]
                at_indices = set(t_indices) & set(a_indices)
                p_a_p_t[a][i] = (len(t_indices) / num_train) * (
                    len(a_indices) / num_train
                )
                p_at[a][i] = len(at_indices) / num_train

                a_cond_t[a][i] = (
                    len(temp_labels[np.where(temp_labels[:, a] == 1)[0]])
                    / temp_labels_size
                )
                ahat_cond_t[a][i] = (
                    len(temp_preds[np.where(temp_preds[:, a] == 1)[0]])
                    / temp_preds_size
                )
                
        y_at = np.sign(p_at - p_a_p_t)
        delta_at = ahat_cond_t - a_cond_t

        if is_abs:
            values = np.abs(delta_at)
        else:
            values = y_at * delta_at
        val = np.nanmean(values)
        var = np.nanvar(delta_at)
        return val, var
