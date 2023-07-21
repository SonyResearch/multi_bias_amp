import numpy as np 
from typing import * 
from sklearn.metrics import average_precision_score, precision_score

def conf_interval(values: List, latex: bool = False) -> Tuple: 
    """
        returns the 95% confidence interval given the values
    """
    av, interval = np.mean(values), 1.96 * np.std(values) / np.sqrt(len(values))
    if latex:
        print('${:.3f} \pm {:.3f}$'.format(av, interval))
    return av, interval

def get_map(labels_list: List, scores_list: List, num_classes: int, dataset: str) -> float:
    """
        labels_list: ground-truth labels
        scores_list: model predictions
        num_classes: number of classes
        dataset: if Imsitu, mAP is calculated over 2 classes for location and action 
    """
    APs = []
    if dataset == 'imsitu': 
        for k in range(3): 
            if k < 2: 
                APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))
            else: 
                labels = np.argmax(labels_list[:,k:], axis=1)
                preds = np.argmax(scores_list[:,k:], axis=1)
                APs.append(precision_score(labels, preds, average='macro'))
    else: 
        if num_classes == 1:
            APs.append(average_precision_score(labels_list, scores_list))
        else:
            for k in range(num_classes):
                APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))
    mAP = np.nanmean(APs)
    return mAP

def threshold_preds(res: Dict) -> List:
    '''
        res: Dictionary with the following keys "scores", "val_scores", and "val_lables"
        Returns an array of the calibrated predictions based on the validation set
    '''
    test_probs = res['scores']
    val_probs, val_labels = res['val_scores'], res['val_labels']
    thresholds = []
    test_preds = test_probs.copy()

    for l in range(len(val_labels[0])):
        # calibration is used to pick threshold
        this_thresholds = np.sort(val_probs[:, l].flatten())
        index = -int(np.sum(val_labels[:, l]))-1
        if -index >= len(this_thresholds): index = 0
        calib = this_thresholds[index]
        pred_num = int(np.sum(val_probs[:, l] > calib))
        actual_num = int(np.sum(val_labels[:, l]))
        if pred_num != actual_num:
            uniques = np.sort(np.unique(val_probs[:, l].flatten()))
            next_calib = uniques[list(uniques).index(calib)+1]
            next_num = int(np.sum(val_probs[:, l] > next_calib))
            if np.absolute(next_num - actual_num) < np.absolute(pred_num - actual_num):
                calib = next_calib
        thresholds.append(calib)
        test_preds[:, l] = test_probs[:, l] > thresholds[-1]
    return test_preds
