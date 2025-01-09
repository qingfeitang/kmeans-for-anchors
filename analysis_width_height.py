import os
import numpy as np
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil

def process(folder, anchors=None, normalize=False):
    files = os.listdir(src_folder)
    h_list = []
    w_list = []
    for file in files:
        with open(os.path.join(src_folder, file), 'r') as f:
            data = f.readlines()
        if len(data) == 0:
            continue
        for item in data:
            item = [float(i) for i in item.strip().split(' ')]
            if normalize:
                w, h = item[-2], item[-1]
            else:
                w = item[3] - item[1]
                h = item[2] - item[0]
            w_list.append(w)
            h_list.append(h)
          
    plt.scatter(w_list, h_list)
    if anchors is not None:
        w_anchor = anchors[:, 0]
        h_anchor = anchors[:, 1]
        plt.scatter(w_anchor, h_anchor, c='r')
        
    plt.ylabel('height')
    plt.xlabel('width')
    plt.savefig('vis-shape.png')

if __name__ == '__main__':
    normalize = True
    src_folder = '../Annotations-yolo' 
    anchors_file = "anchors.txt" 
    with open(anchors_file, 'r') as f:
        anchors = np.array(eval(f.read()))

    #anchors = anchors * np.array([[1280, 720]])

    process(src_folder, anchors if normalize else None, normalize)


