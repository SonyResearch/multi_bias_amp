import pickle, time, argparse
from os import path, mkdir
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from classifier import multilabel_classifier
from load_data import *
from utils import * 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nclasses', type=int, default=1, help="Number of classes in prediction")
    parser.add_argument('--modelname', type=str, default='resnet', choices=['resnet', 'lenet', 'vanillanet'], help="Model type to use")
    parser.add_argument('--modelpath', type=str, default=None, help="Path to existing model (if exists)")
    parser.add_argument('--labels_test', type=str, default=None, help="Path to pickle file with the test labels")
    parser.add_argument('--labels_val', type=str, default=None, help="Path to pickle file with the val labels")
    parser.add_argument('--batchsize', type=int, default=32, help="Batch size for evaluation")
    parser.add_argument('--hs', type=int, default=512, help="Hidden size")
    parser.add_argument('--device', default=0, help="Either CUDA device number OR CPU")
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--outfile', default=str, help="Path pickle file for saving results of evaluation")
    parser.add_argument('--dataset_name', type=str, default='mnist', help="Name of dataset used to determine image transforms")
    parser.add_argument('--freeze', type=int, default=0, help="0 = only train final layer (only applies if model architecture is ResNet)")
    arg = vars(parser.parse_args())
    
    print('\n', arg, '\n')

    if arg['device'] == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(arg['device']))

    # Create dataloader
    testset = create_dataset(arg['dataset_name'], arg['labels_test'], B=arg['batchsize'], train=False)

    # Load model
    feature_extract = True if arg['freeze'] == 0 else False
    classifier = multilabel_classifier(device, arg['dtype'], nclasses=arg['nclasses'], modelpath=arg['modelpath'],\
         hidden_size=arg['hs'], model_name=arg['modelname'], feature_extract=feature_extract)
    
    # Do inference with the model
    labels_list, scores_list, test_loss_list, files_list = classifier.test(testset)
    
    output = {'files': files_list, 'labels': labels_list, 'scores': scores_list, 'loss': test_loss_list}

    testset = create_dataset(arg['dataset_name'], arg['labels_val'], B=arg['batchsize'], train=False)
    feature_extract = True if arg['freeze'] == 0 else False
    classifier = multilabel_classifier(device, arg['dtype'], nclasses=arg['nclasses'], modelpath=arg['modelpath'],\
         hidden_size=arg['hs'], model_name=arg['modelname'], feature_extract=feature_extract)

    output['val_labels'] = labels_list
    output['val_scores'] = scores_list

    pickle.dump(output, open(arg['outfile'], 'wb'))

    # Calculate and record mAP
    mAP = get_map(labels_list, scores_list, arg['nclasses'], arg['dataset_name'])
    print('mAP (all): {:.2f}'.format(mAP*100.))


if __name__ == "__main__":
    main()