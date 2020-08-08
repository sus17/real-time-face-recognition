# from MobileFaceNet import MobileFacenet
from PIL import Image

import numpy as np
import argparse
import torch
import json
import time
import os


def recognize(picture_name, net, device, feature_save_dir):
    if isinstance(picture_name, str):
        img = Image.open(picture_name)
    else:
        # you may directly use this function without firing 'python face_recognize.py'
        img = picture_name
    
    img = np.array(img)
    if img.shape != (112, 96, 3):
        img.resize(112, 96, 3)
    
    img_mirror = img[:, ::-1, :].copy()
    
    img = (img - 127.5) / 128.0
    img = img.transpose(2, 0, 1) # HxWxC -> CxHxW
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0) # GxCxHxW
    img = img.to(device)
    
    img_mirror = (img_mirror - 127.5) / 128.0
    img_mirror = img_mirror.transpose(2, 0, 1)
    img_mirror = torch.from_numpy(img_mirror).float()
    img_mirror = img_mirror.unsqueeze(0)
    img_mirror = img_mirror.to(device)
    
    with torch.no_grad():
        my_feature = net(img)
        my_feature_mirror = net(img_mirror)
    
    my_feature = my_feature.detach().cpu().numpy().flatten()
    my_feature_mirror = my_feature_mirror.detach().cpu().numpy().flatten() # (128, )
    my_feature = np.concatenate((my_feature, my_feature_mirror), 0)        # (256, )
    # print(my_feature.shape)
    
    with open(feature_save_dir, 'r') as fp:
        feature_dict = json.load(fp)
    
    cosine_similarity_collection = []
    print("|  name / number   |     similarity      ")
    print("-" * 40)
    
    for key, value in feature_dict.items():
        value = np.array(value)
        
        cosine_similarity = np.dot(my_feature, value) / (np.linalg.norm(my_feature) * np.linalg.norm(value))
        
        cosine_similarity_collection.append(cosine_similarity)
        print("|  %s" % key + " " * (15 - len(key)), "|      %.6f" % cosine_similarity)
    
    max_similarity = max(cosine_similarity_collection)
    best_selection = cosine_similarity_collection.index(max_similarity)
    print("\nfinal decision: %s\nsimilarity: %.4f" % \
          (list(feature_dict.keys())[best_selection], max_similarity))
    
    if max_similarity < 0.5:
        print("[warning] final result is not absolutely reliable! please check.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--picture_name',
        type=str,
        required=True,
        help="path to store picture which will be recognized"
    )
    parser.add_argument(
        '--feature_save_dir',
        default='feature.json',
        help='path to store encoded features from your image database'
    )
    parser.add_argument(
        '--model_path',
        default='model/best/068.ckpt',
        help='path to save MobileFacenet checkpoint'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='choice on gpu or cpu to run MobileFacenet model'
    )
    args = parser.parse_args()

    if not os.path.isfile(args.picture_name):
        print("Your given picture '%s' doesn't exist! Please check." % args.picture_name)
        exit(0)

    if not os.path.isfile(args.model_path):
        print("model file %s doesn't exist! Please check." % args.model_path)
        exit(0)
    
    if not args.device == "cuda" and not args.device == 'cpu':
        print("device must be 'cuda' or 'cpu', while your selection is %s. Please check." % args.device)
        exit(0)

    print("Loading MobileFacenet model, please wait...")
    state_dict = torch.load(args.model_path)
    net = MobileFacenet()
    net.load_state_dict(state_dict['net_state_dict'])
    net.eval()

    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
    net.to(device)
    
    print("Recognize begin!\n")
    start = time.time()
    recognize(args.picture_name, net, device, args.feature_save_dir)
    print("Runtime: %6fs (only excluding time to load model)" % (time.time() - start))