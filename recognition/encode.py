from core.model import MobileFacenet
from PIL import Image

import numpy as np
import argparse
import torch
import json
import tqdm
import os

def encode(root_path, save_path, net, verbose):
    # assert not os.path.isfile('feature.json'), 'feature record file already exists!'
    
    persons = sorted(os.listdir(root_path))
    print("Now you have %d people in your database." % len(persons))
    
    encode_dict = {}
    
    for person_name in persons:
        if verbose:
            print("extract features from people: %s...  " % person_name, end = "")
        
        person_pictures = sorted(os.listdir(os.path.join(root_path, person_name)))
        
        for picture_name in person_pictures:
            picture_path = os.path.join(root_path, person_name, picture_name)
            
            if not picture_path.endswith(".jpg") and not picture_path.endswith(".png"):
                continue
            img = Image.open(picture_path)
            img = np.array(img)
            img_mirror = img[:, ::-1, :].copy()

            img = (img - 127.5) / 128.0
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(0)
            img = img.to(device)
            
            img_mirror = (img_mirror - 127.5) / 128.0
            img_mirror = img_mirror.transpose(2, 0, 1)
            img_mirror = torch.from_numpy(img_mirror).float()
            img_mirror = img_mirror.unsqueeze(0)
            img_mirror = img_mirror.to(device)
            # print(img.shape)
            
            with torch.no_grad():
                feature = net(img)
                feature_mirror = net(img_mirror)
            
            feature_list_origin = feature.detach().cpu().numpy().flatten().tolist()
            feature_list_mirror = feature_mirror.detach().cpu().numpy().flatten().tolist()
            
            # print(len(feature.detach().cpu().numpy().flatten().tolist()))
            encode_dict[picture_name[:-4]] = feature_list_origin + feature_list_mirror
        
        if verbose:
            print("done")
    
    with open(save_path, 'w') as fp:
        json.dump(encode_dict, fp)

    print("done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        default="./img_database/",
        help="path to store your own image dataset to implement face recognition"
    )
    parser.add_argument(
        "--save_dir",
        default="./feature.json",
        help="path to save predicted 256 features, json file required"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="choice on gpu or cpu to run MobileFacenet model"
    )
    parser.add_argument(
        "--verbose",
        default=True,
        type=bool,
        help="bool value to decide whether to show people's name during feature encoding stage"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print("Image dataset directory '%s' doesn't exists! Please check it and re-run this py." % args.root_dir)
        exit(0)
    
    if not args.save_dir.endswith(".json"):
        print("Current save path is %s, which should end with '.json'. Please correct it." % args.save_dir)
        exit(0)

    if not args.device == "cuda" and not args.device == 'cpu':
        print("device must be 'cuda' or 'cpu', while your selection is %s. Please check." % args.device)
        exit(0)
    
    if os.path.isfile(args.save_dir):
        print("[Reminder] File '%s' already exists! Do you want to re-write it?" % args.save_dir)
        answer = input("[Y]es, re-write it anyway.    [N]o, let me have a check : ")
        if not (answer == "Y" or answer == "Yes" or answer == "y" or answer == 'yes'):
            print("quit.")
            exit(0)

    print("Loading MobileFacenet model, please wait...")
    state_dict = torch.load('model/best/068.ckpt')
    net = MobileFacenet()
    net.load_state_dict(state_dict['net_state_dict'])
    net.eval()

    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
    net.to(device)

    print("Encoding starts!")
    encode(args.root_dir, args.save_dir, net, args.verbose)