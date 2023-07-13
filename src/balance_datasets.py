import pickle 
import numpy as np 
import math 
import argparse 

from typing import *
from tqdm import tqdm 

def check_bias(attributes: List, thresh: float, group_num: int) -> bool: 
    '''
        Helper function that checks whether the bias function is equal to 0.5
    '''
    balanced_thresh = 1 / group_num 
    _, obj = attributes.shape
    m_attributes = attributes[np.where(attributes[:, 0] == 0)]
    return_bog = []
    for o in range(1, obj - 1):
        att_bog = np.sum(m_attributes[np.where(m_attributes[:, o] == 1)], axis=0)[o] / len(attributes[np.where(attributes[:, o] == 1)])
        return_bog.append(att_bog)
        if math.isnan(att_bog): 
            continue
        elif np.abs(att_bog - balanced_thresh) > thresh: 
            print(o, att_bog)
            return False
    return True

def return_all_bogs(attributes: List) -> List: 
    '''
        Helper function that returns all of the bias scores for group g = 0
    '''
    _, obj = attributes.shape
    m_attributes = attributes[np.where(attributes[:, 0] == 0)]
    return_bog = []
    for o in range(1, obj - 1):
        att_bog = np.sum(m_attributes[np.where(m_attributes[:, o] == 1)], axis=0)[o] / len(attributes[np.where(attributes[:, o] == 1)])
        return_bog.append(att_bog)
    return return_bog

def get_bog(attributes: List, o: int) -> float:
    '''
        Helper function that returns the specific bias score for attribute o and group g = 0. 
    '''
    m_attributes = attributes[np.where(attributes[:, 0] == 0)]
    return np.sum(m_attributes[np.where(m_attributes[:, o] == 1)], axis=0)[o] / len(attributes[np.where(attributes[:, o] == 1)])

def greedy_sample(select_attributes: List) -> Tuple:
    '''
        Greedily undersamples to balance P(o | g = i) for all i in G
    '''
    male_index = np.where(select_attributes[:, 0] == 0)
    female_index = np.where(select_attributes[:, 0] == 1)
    male_atts, female_atts = select_attributes[male_index], select_attributes[female_index]
    num_male, num_female = len(male_atts), len(female_atts)
    sampled_indices = np.random.choice(np.arange(max([num_male, num_female])), min(num_male, num_female))
    if num_male > num_female: 
        orig_atts, added_atts = select_attributes[female_index], select_attributes[male_index][sampled_indices]
    elif num_female > num_male:
        orig_atts, added_atts = select_attributes[male_index], select_attributes[female_index][sampled_indices]
    return orig_atts, added_atts 

def greedy_oversample(select_attributes: List) -> Tuple:
    '''
        Greedily oversamples to balance P(o | g = i) for all i in G
    '''
    male_index = np.where(select_attributes[:, 0] == 0)
    female_index = np.where(select_attributes[:, 0] == 1)
    male_atts, female_atts = select_attributes[male_index], select_attributes[female_index]
    num_male, num_female = len(male_atts), len(female_atts)
    diff = np.abs(num_male - num_female)
    sampled_indices = np.random.choice(np.arange(min([num_male, num_female])), diff, replace=True)
    if num_male < num_female: 
        orig_atts, added_atts = select_attributes, select_attributes[male_index][sampled_indices]
    elif num_female < num_male:
        orig_atts, added_atts = select_attributes, select_attributes[female_index][sampled_indices]
    return orig_atts, added_atts

def balance_dataset(filename: str, thresh: int, seed:int, verbose: bool, save: bool, outfile: Optional[str], split_name: Optional[str], dataset_name: Optional[str]) -> int:
    '''
        filename: string for the name of the pickle file containing the filenames and attributes
        thresh: float for the error tolerance (i.e., e in 1 / |G| +\- e)
        seed: integer that sets random seed for oversampling
        verbose: boolean to determine whether we should print out the number of instances 
        save: boolean to determine whether we are saving the output of the oversampling
        outfile: if save is True, determines where the output pickle file will be written
        split_name: string choice between [train, val, test]
        dataset_name: string choice between [coco, imsitu]

        The greedily oversampling will stop after five iterations unless it converges
    '''
    iter_count = 0
    file = pickle.load(open(filename, 'rb'))
    img_names = np.array(list(file.keys())).reshape(-1, 1)
    img_numbers = np.arange(len(img_names)).reshape(-1, 1)
    attributes = np.stack(list(file.values()))
    attributes = np.hstack((attributes, img_numbers))
    _, obj = attributes.shape

    np.random.seed(seed) # seed = 0

    balanced_thresh = 1/2
    while not check_bias(attributes, thresh, 2): 
        if verbose:
            print("Iteration Count: {}".format(iter_count))

        bog = np.sum(attributes, axis=0)[0] / len(attributes)
        # deal with the attribute separately
        if np.abs(bog - balanced_thresh) > thresh: 
            orig_atts, added_atts = greedy_oversample(attributes)
            attributes = np.concatenate((orig_atts, added_atts))

        for o in tqdm(range(1, obj - 1)): 
            # first check what the bog is 
            bog = get_bog(attributes, o)
            if np.abs(bog - balanced_thresh) > thresh: 
                # greedily sample to balance the co-occurrences
                select_attributes = attributes[np.where(attributes[:, o] == 1)]
                try: 
                    orig_atts, added_atts = greedy_oversample(select_attributes)
                except:
                    print(bog)
                remaining_atts = attributes[np.where(attributes[:, o] == 0)]
                attributes = np.concatenate((orig_atts, added_atts, remaining_atts))
        iter_count += 1
        if iter_count > 5: break
    final_bog = return_all_bogs(attributes)
    if save:
        out_file = []
        for i in tqdm(attributes): 
            file = str(int(i[-1])).zfill(12)
            if dataset_name == 'coco': 
                filename = 'coco/{0}2014/COCO_{0}2014_{1}.jpg'.format(split_name, file)
            elif dataset_name == 'imsitu': 
                filename = 'imsitu/images/{}'.format(img_names[int(i[-1])][0])
            labels = list([int(x) for x in i[:-1]])
            labels.insert(0, filename)
            out_file.append(labels)
        out_file = np.stack(out_file)
        pickle.dump(out_file, open(outfile, 'wb'))
    print(final_bog)
    if verbose: 
        print('Number of instances: {}'.format(len(attributes)))
    return(len(attributes))

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help="Path to pickle file with annotations to balance")
    parser.add_argument('--thresh', type=int, default=0.025, help="Error threshold")
    parser.add_argument('--seed', type=int, default=None, help="Random seed")
    parser.add_argument('--outfile', type=str, default=None, help="Path to pickle file to save results")
    parser.add_argument('--split', type=str, default=None, help="Split name")
    parser.add_argument('--dataset', type=str, default=None, help="Dataset name")
    arg = vars(parser.parse_args())

    balance_dataset(arg['filename'], arg['thresh'], arg['seed'], True, True, arg['outfile'], arg['split'], arg['dataset'])