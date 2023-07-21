import json 
import numpy as np
import pickle 

from typing import * 
from pycocotools.coco import COCO 


def preprocess_imsitu(anns: Dict, split: str) -> Dict:
    verb_map = json.load(open('../data/imsitu/verb_map.json', 'r'))
    io = json.load(open('../data/imsitu/places.json'))
    outdoors, indoors = io['outdoor'], io['indoor']
    
    agents = json.load(open('../data/imsitu/agents.json'))
    male, female = agents['m'], agents['f']
    out_file = {}
    for ann in anns:
        for frame in anns[ann]['frames']:
            if 'agent' in frame and (frame['agent'] in male or frame['agent'] in female): 
                g = 0 if frame['agent'] in male else 1
                if 'place' in frame: 
                    if frame['place'] in outdoors or frame['place'] in indoors: 
                        l = 0 if frame['place'] in outdoors else 1 
                        v = anns[ann]['verb']
                        if v in verb_map: 
                            labels= np.zeros(len(verb_map) + 2)
                            labels[verb_map[v] + 2] = 1
                            labels[1] = l
                            labels[0] = g
                            out_file[ann] = labels
                            break 
    pickle.dump(out_file, open('../data/imsitu/{}_unbalanced.pkl'.format(split), 'wb'))
    return out_file

def preprocess_coco(split: str):
    anns = pickle.load(open('../data/coco/all_anns.pkl', 'rb'))
    imgs = pickle.load(open('../data/coco/{}_gender.pkl'.format(split), 'rb'))
    out_file = {}
    for img in imgs: 
        out_file[img] = anns[img]
    print(np.sum(list(out_file.values()), axis=0) / len(list(out_file.values())))
    print('# of files in {}: {}'.format(split, len(out_file)))
    pickle.dump(out_file, open('../data/coco/{}_unbalanced.pkl'.format(split), 'wb'))


