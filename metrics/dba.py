'''
Bias Amplification Metric from Directional Bias Amplification
Code from Wang and Russakovsky (https://github.com/princetonvisualai/directional-bias-amp) 
'''
import numpy as np
import pickle
import typing 
from decimal import Decimal
from utils import threshold_preds

# Note: in our work attributes are equivalent to "task" in Wang and Russakovsky's paper and group membership is equivalent to "attribute"

def biasamp_task_to_attribute(task_labels, attribute_labels, attribute_preds, task_labels_train=None, attribute_labels_train=None, names=None, is_abs = False) -> float:
    '''
    for each of the following, an entry of 1 is a prediction, and 0 is not
    task_labels: n x |T|, these are labels on the test set, where n is the number of samples, and |T| is the number of tasks to be classified
    attribute_labels: n x |A|, these are labels on the test set, where n is the number of samples, and |A| is the number of attributes to be classified
    attribute_preds: n x |A|, these are predictions on the test set for attribute
    optional: below are used for setting the direction of the indicator variable. if not provided, test labels are used
    task_labels_train: m x |T|, these are labels on the train set, where m is the number of samples, and |T| is the number of tasks to be classified
    attribute_labels_train: m x |A|, these are labels on the train set, where m is the number of samples, and |A| is the number of attributes to be classified
    names: list of [task_names, attribute_names]. if included, will print out the top 10 attribute-task pairs with the most bias amplification
    '''
    assert len(task_labels.shape) == 2 and len(attribute_labels.shape) == 2, 'Please read the shape of the expected inputs, which should be "num samples" by "num classification items"'
    if task_labels_train is None or attribute_labels_train is None:
        task_labels_train, attribute_labels_train = task_labels, attribute_labels
    num_t, num_a = task_labels.shape[1], attribute_labels.shape[1]
    
    # only include images that have attribute(s) and task(s) associated with it for calculation of indicator variable
    keep_indices = np.array(list(set(np.where(np.sum(task_labels_train, axis=1)>0)[0]).union(set(np.where(np.sum(attribute_labels_train, axis=1)>0)[0]))))
    task_labels_train, attribute_labels_train = task_labels_train[keep_indices], attribute_labels_train[keep_indices]
    
    # y_at calculation
    p_at = np.zeros((num_a, num_t))
    p_a_p_t = np.zeros((num_a, num_t))
    num_train = len(task_labels_train)
    for a in range(num_a):
        for t in range(num_t):
            t_indices = np.where(task_labels_train[:, t]==1)[0]
            a_indices = np.where(attribute_labels_train[:, a]==1)[0]
            at_indices = set(t_indices)&set(a_indices)
            p_a_p_t[a][t] = len(t_indices)/num_train * len(a_indices)/num_train
            p_at[a][t] = len(at_indices)/num_train
    y_at = np.sign(p_at - p_a_p_t)

    # delta_at calculation
    a_cond_t = np.zeros((num_a, num_t))
    ahat_cond_t = np.zeros((num_a, num_t))
    for a in range(num_a):
        for t in range(num_t):
            a_cond_t[a][t] = np.mean(attribute_labels[:, a][np.where(task_labels[:, t]==1)[0]])
            ahat_cond_t[a][t] = np.mean(attribute_preds[:, a][np.where(task_labels[:, t]==1)[0]])
    delta_at = ahat_cond_t - a_cond_t

    if is_abs:
        values = np.abs(y_at*delta_at)
    else:
        values = y_at*delta_at
        
    val = np.nanmean(values)
    var = np.nanvar(y_at*delta_at)
    if names is not None:
        assert len(names) == 2, "Names should be a list of the task names and attribute names"
        task_names, attribute_names = names
        assert len(task_names)==num_t and len(attribute_names)==num_a, "The number of names should match both the number of tasks and number of attributes"

        sorted_indices = np.argsort(np.absolute(values).flatten())
        for i in sorted_indices[::-1][:10]:
            a, t = i // num_t, i % num_t
            print("{0} - {1}: {2:.4f}".format(attribute_names[a], task_names[t], values[a][t]))
    return val, var
     
def biasamp_attribute_to_task(task_labels, attribute_labels, task_preds, task_labels_train=None, attribute_labels_train=None, names=None, is_abs = False) -> float:
    '''
    for each of the following, an entry of 1 is a prediction, and 0 is not
    task_labels: n x |T|, these are labels on the test set, where n is the number of samples, and |T| is the number of tasks to be classified
    attribute_labels: n x |A|, these are labels on the test set, where n is the number of samples, and |A| is the number of attributes to be classified
    task_preds: n x |T|, these are predictions on the test set for task
    optional: below are used for setting the direction of the indicator variable. if not provided, test labels are used
    task_labels_train: m x |T|, these are labels on the train set, where m is the number of samples, and |T| is the number of tasks to be classified
    attribute_labels_train: m x |A|, these are labels on the train set, where m is the number of samples, and |A| is the number of attributes to be classified
    names: list of [task_names, attribute_names]. if included, will print out the top 10 attribute-task pairs with the most bias amplification
    '''

    assert len(task_labels.shape) == 2 and len(attribute_labels.shape) == 2, 'Please read the shape of the expected inputs, which should be "num samples" by "num classification items"'
    if task_labels_train is None or attribute_labels_train is None:
        task_labels_train, attribute_labels_train = task_labels, attribute_labels
    num_t, num_a = task_labels.shape[1], attribute_labels.shape[1]
    
    # only include images that have attribute(s) and task(s) associated with it for calculation of indicator variable
    keep_indices = np.array(list(set(np.where(np.sum(task_labels_train, axis=1)>0)[0]).union(set(np.where(np.sum(attribute_labels_train, axis=1)>0)[0]))))
    task_labels_train, attribute_labels_train = task_labels_train[keep_indices], attribute_labels_train[keep_indices]
    
    # y_at calculation
    p_at = np.zeros((num_a, num_t))
    p_a_p_t = np.zeros((num_a, num_t))
    num_train = len(task_labels_train)
    for a in range(num_a):
        for t in range(num_t):
            t_indices = np.where(task_labels_train[:, t]==1)[0]
            a_indices = np.where(attribute_labels_train[:, a]==1)[0]
            at_indices = set(t_indices)&set(a_indices)
            # print(len(at_indices), num_train)
            p_a_p_t[a][t] = (len(t_indices)/num_train)*(len(a_indices)/num_train)
            p_at[a][t] = len(at_indices)/num_train
    y_at = np.sign(p_at - p_a_p_t)

    # delta_at calculation
    t_cond_a = np.zeros((num_a, num_t))
    that_cond_a = np.zeros((num_a, num_t))

    for a in range(num_a):
        for t in range(num_t):
            t_cond_a[a][t] = np.mean(task_labels[:, t][np.where(attribute_labels[:, a]==1)[0]])
            that_cond_a[a][t] = np.mean(task_preds[:, t][np.where(attribute_labels[:, a]==1)[0]])
    delta_at = that_cond_a - t_cond_a
    if is_abs:
        values = np.abs(y_at*delta_at)
    else:
        values = y_at*delta_at
    val = np.nanmean(values)
    var = np.nanvar(y_at*delta_at)

    if names is not None:
        assert len(names) == 2, "Names should be a list of the task names and attribute names"
        task_names, attribute_names = names
        assert len(task_names)==num_t and len(attribute_names)==num_a, "The number of names should match both the number of tasks and number of attributes"

        sorted_indices = np.argsort(np.absolute(values).flatten())
        for i in sorted_indices[::-1][:10]:
            a, t = i // num_t, i % num_t
            print("{0} - {1}: {2:.4f}".format(attribute_names[a], task_names[t], values[a][t]))
    return val, var

if __name__ == '__main__':
    i = 0
    train = pickle.load(open('../coco/anns/train.pkl'.format(i), 'rb'))
    # train = np.stack([train[i] for i in train])
    train = train[:, 1:].astype(int)
    train_atts = train[:, 1:]
    train_group = train[:, 0]
    preds = pickle.load(open('../results/coco_{}.pkl'.format(i), 'rb'))
    pred = threshold_preds(preds)
    test = preds['labels']
    train_group, train_attributes = train[:, 0].reshape((-1, 1)), train[:, 1:]
    test_group, test_attributes = test[:, 0].reshape((-1, 1)), test[:, 1:]
    pred_group, pred_attributes = pred[:, 0].reshape((-1, 1)), pred[:, 1:]
    at = biasamp_attribute_to_task(test_attributes, test_group, pred_attributes, task_labels_train=train_attributes, attribute_labels_train=train_group)