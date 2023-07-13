import pickle, time, argparse, random
from os import path, makedirs
import numpy as np
import torch
import json

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score

from classifier import multilabel_classifier
from load_data import *
from utils import * 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default=None, help="Path to existing model (if exists)")
    parser.add_argument('--modelname', type=str, default='resnet', choices=['resnet', 'lenet', 'vanillanet'], help="Model type to use")
    parser.add_argument('--dataset_name', type=str, default='mnist', help="Name of dataset used to determine image transforms")
    parser.add_argument('--pretrainedpath', type=str, default=None, help="Path to pretraining (if not using ILSVRC)")
    parser.add_argument('--outdir', type=str, default='models', help="Path where intermediary modelpaths and final modelpaths will be saved")
    parser.add_argument('--nclasses', type=int, default=5, help="Number of classes in prediction")
    parser.add_argument('--labels_train', type=str, default=None, help="Path to pickle file with the train labels")
    parser.add_argument('--labels_val', type=str, default=None, help="Path to pickle file with the val labels")
    parser.add_argument('--nepoch', type=int, default=1, help="Number of epochs to train for")
    parser.add_argument('--train_batchsize', type=int, default=128, help="Batch size for training")
    parser.add_argument('--val_batchsize', type=int, default=64, help="Batch size for evaluation")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")
    parser.add_argument('--hs', type=int, default=2048, help="Hidden size")
    parser.add_argument('--seed', type=int, default=999, help="Random seed")
    parser.add_argument('--device', default=0, help="Either CUDA device number OR CPU")
    parser.add_argument('--freeze', type=int, default=0, help="0 = only train final layer (only applies if model architecture is ResNet)")
    parser.add_argument('--dtype', default=torch.float32)
    
    arg = vars(parser.parse_args())
    print('\n', arg, '\n')
    print('\nTraining with {} GPUs'.format(torch.cuda.device_count()))

    if arg['device'] == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(arg['device']))

    # Set random seed
    random.seed(arg['seed'])
    np.random.seed(arg['seed'])
    torch.manual_seed(arg['seed'])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create output directory
    if not path.isdir(arg['outdir']):
        makedirs(arg['outdir'])


    # Create data loaders
    trainset = create_dataset(arg['dataset_name'], arg['labels_train'], 
                              B=arg['train_batchsize'], train=True)
    valset = create_dataset(arg['dataset_name'], arg['labels_val'],
                             B=arg['val_batchsize'], train=False)

    # Initialize classifier
    feature_extract = True if arg['freeze'] == 0 else False
    classifier = multilabel_classifier(device, arg['dtype'], nclasses=arg['nclasses'],
                                       modelpath=arg['modelpath'], hidden_size=arg['hs'], model_name=arg['modelname'], feature_extract=feature_extract)
    classifier.epoch = 1 # Reset epoch for stage 2 training
    optimizer = torch.optim.Adam(classifier.model.parameters(), lr=arg['lr'], weight_decay=arg['wd'])
    classifier.optimizer = optimizer

    # Keep track of loss and mAP/recall for best model selection
    loss_epoch_list = []; all_list = []

    # Start training
    tb = SummaryWriter(log_dir='{}/runs'.format(arg['outdir']))
    start_time = time.time()
    print('\nStarted training at {}\n'.format(start_time))
    for i in range(1, arg['nepoch']+1):

        # Reduce learning rate from 0.1 to 0.01
        train_loss_list = classifier.train(trainset)
        
        # Save the model
        if (i + 1) % 10 == 0:
            classifier.save_model('{}/model_{}.pth'.format(arg['outdir'], i))

        # Do inference with the model
        labels_list, scores_list, val_loss_list, _ = classifier.test(valset)
        # Record train/val loss
        tb.add_scalar('Loss/Train', np.mean(train_loss_list), i)
        tb.add_scalar('Loss/Val', np.mean(val_loss_list), i)
        loss_epoch_list.append(np.mean(val_loss_list))

        # Calculate and record mAP
        mAP = get_map(labels_list, scores_list, arg['nclasses'], arg['dataset_name'])
        tb.add_scalar('mAP/all', mAP*100, i)

        all_list.append(mAP*100)

        
        # Print out information
        print('\nEpoch: {}'.format(i))
        print('Loss: train {:.5f}, val {:.5f}'.format(np.mean(train_loss_list), np.mean(val_loss_list)))
        print('Val mAP: all {} {:.5f}'.format(arg['nclasses'], mAP*100))
        print('Time passed so far: {:.2f} minutes\n'.format((time.time()-start_time)/60.))


    # Print best model and close tensorboard logger
    tb.close()
    print('Best model at {} with lowest val loss {}'.format(np.argmin(loss_epoch_list) + 1, np.min(loss_epoch_list)))
    return mAP
    
if __name__ == "__main__":
    main()