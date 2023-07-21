from metrics import mals, dba, mba
from typing import * 

import argparse
import pickle 
import numpy as np
from utils import *
from tqdm import tqdm 

def evaluate_mals(train_file: str, results_file: str, group: int = 2, calib: bool = True, verbose: bool = False, is_balanced: bool = False, is_abs: bool = False) -> int:
    # Return the single-attribute MALS metric
    train_file = pickle.load(open(train_file, 'rb'))
    results_file = pickle.load(open(results_file, 'rb'))
    if calib: 
        test_atts = threshold_preds(results_file)
    else: 
        test_atts = np.round(results_file['scores'])
    if is_balanced:
        train_atts = train_file[:, 1:].astype(int)
    else: 
        train_atts = np.stack([train_file[i] for i in train_file])
    mals_metric = mals.MALS(train_atts, test_atts, group)
    
    # calculate the bias scores for the train set and predictions
    bog_tilde = mals_metric.create_bog(mals_metric.train_attributes, mals_metric.train_group, group)
    bog_pred = mals_metric.create_bog(mals_metric.pred_attributes, mals_metric.pred_group, group)
    mals_value, mals_var, _ = mals_metric.bog_mals(bog_tilde, bog_pred, is_abs=is_abs)
    if verbose:
        print('MALS value: {}'.format(mals_value))
    return mals_value, mals_var

def evaluate_dba(train_file: str, results_file: str, group: int = 2, calib: bool = True, verbose: bool = False, is_balanced: bool = False, is_abs: bool = False) -> Tuple[Tuple[float, float], Tuple[float, float]]: 
    # Return the single-attribute directional bias amplification metric
    # Assumes that group membership is in the beginning of the annotations 

    train_file = pickle.load(open(train_file, 'rb'))
    results_file = pickle.load(open(results_file, 'rb'))
    if is_balanced:
        train = train_file[:, 1:].astype(int)
    else: 
        train = np.stack([train_file[i] for i in train_file])
    if calib: 
        pred = threshold_preds(results_file)
    else:
        pred = np.round(results_file['scores'])
    test = results_file['labels']
    train_group, train_attributes = train[:, :group - 1], train[:, group - 1:]
    test_group, test_attributes = test[:, :group - 1], test[:, group - 1:]
    pred_group, pred_attributes = pred[:, :group - 1], pred[:, group - 1:]
    ta, ta_var = dba.biasamp_task_to_attribute(test_attributes, test_group, pred_group, task_labels_train=train_attributes, attribute_labels_train=train_group, is_abs = is_abs)
    at, at_var = dba.biasamp_attribute_to_task(test_attributes, test_group, pred_attributes, task_labels_train=train_attributes, attribute_labels_train=train_group, is_abs = is_abs)
    if verbose:
        print('Bias Amp AG value:{}'.format(ta))
        print('Bias Amp GA value:{}'.format(at))
    return (ta, ta_var), (at, at_var)

def evaluate_mba(train_file: str, results_file: str, group: int = 2, calib: bool = True, verbose: bool = False, type: int = 2, k: int = 1, is_balanced: bool = False, is_abs: bool = False) -> Tuple:
    train_file = pickle.load(open(train_file, 'rb'))
    preds = pickle.load(open(results_file, 'rb'))
    if is_balanced:
        train = train_file[:, 1:].astype(int)
    else: 
        train = np.stack([train_file[i] for i in train_file])
    train_atts = train[:, group - 1:]
    train_group = train[:, :group - 1]     
    mba_metric = mba.MBA(train_atts, train_group, preds, num_groups=group, k=k, calib=calib)
    if type == 0:
        # MBA only
        _, mba_mals, mba_var = mba_metric.calculate_mba_mals()
        if verbose: 
            print("Multi MALS Mean {} Var {}".format(mba_mals, mba_var))
        return (mba_mals, mba_var), None, None
    if type == 1:
        # both Directional MBA
        mba_at, at_var = mba_metric.calculate_mba_at()
        mba_ta, ta_var = mba_metric.calculate_mba_ta()
        if verbose: 
            print("Multi MG Mean {} Var {}".format(mba_ta, ta_var))
            print("Multi GM Mean {} Var {}".format(mba_at, at_var))
        return None, mba_at, mba_ta
    if type == 2:
        # Both
        _, mba_mals, mba_var = mba_metric.calculate_mba_mals(is_abs=is_abs)
        mba_at, at_var = mba_metric.calculate_mba_at(is_abs=is_abs)
        mba_ta, ta_var = mba_metric.calculate_mba_ta(is_abs=is_abs)
        if verbose: 
            print("Multi MALS Mean {} Var {}".format(mba_mals, mba_var))
            print("Multi MG Mean {} Var {}".format(mba_ta, ta_var))
            print("Multi GM Mean {} Var {}".format(mba_at, at_var))
        return (mba_mals, mba_var), (mba_at, at_var), (mba_ta, ta_var)
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=None, help="Pickle file with the training images and labels (assumes a dictionary where the keys are the images and values are labels)")
    parser.add_argument('--results', type=str, default=None, help="Pickle file with the results") 
    parser.add_argument('--dataset', type=str, default=None, help="Name of the dataset")
    parser.add_argument('--outfile', type=str, default=None, help="Name of pickle file to write results")
    parser.add_argument('--group', type=int, default=2, help="Number of groups")
    parser.add_argument('--num', type=int, default=6, help="Number of seeds")
    parser.add_argument('--k', type=int, default=1, help="Minimum # of objects in multi-attributes -> 1 assumes we include both single + multi and 2 assumes only multi")
    parser.add_argument('--type', type=int, default=2, help="Choice between 0 (only Multi MALS), 1 (only directional Multi), and 2 (both)")
    parser.add_argument('--calib', action=argparse.BooleanOptionalAction, help="If true, calibrate predictions based on validation set. If false, round the results")
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help="If true, print out the values")
    parser.add_argument('--array', action=argparse.BooleanOptionalAction, help="If true, the input is a numpy array. If false, the input is a dict")
    parser.add_argument('--abs', action=argparse.BooleanOptionalAction, help="If true, use absolute value. If false, return raw")
    arg = vars(parser.parse_args())

    results = {}
    try:
        results_fpath = arg['results'].format(arg['dataset'], i)
        results_file = pickle.load(open(results_fpath, 'rb'))
        _, groups = results_file['scores'].shape
        results['maps'] = get_map(results_file['labels'], results_file['scores'], groups, arg['dataset']) * 100
        mals = evaluate_mals(arg['train'], results_fpath, arg['group'], arg['calib'], arg['verbose'], arg['array'], arg['abs'])
        results['single_mals'] = (mals[0] * 100, mals[1] * 100)
        mg, gm = evaluate_dba(arg['train'], results_fpath, arg['group'], arg['calib'], arg['verbose'], arg['array'], arg['abs'])
        results['single_mg'] = (mg[0] * 100, mg[1] * 100)
        results['single_gm'] = (gm[0] * 100, gm[1] * 100)
        mba, mba_gm, mba_mg = evaluate_mba(arg['train'], results_fpath, arg['group'], arg['calib'], arg['verbose'], arg['type'], k, arg['array'], arg['abs'])
        results['multi_mals'] = (mba[0] * 100, mba[1] * 100)
        results['multi_gm'] = (mba_gm[0] * 100, mba_gm[1] * 100)
        results['multi_mg'] = (mba_mg[0] * 100, mba_mg[1] * 100)
    except Exception as e:
        print(e)
    pickle.dump(results, open(arg['outfile'], 'wb'))




if __name__ == '__main__':
    main()
