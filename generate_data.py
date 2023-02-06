import numpy as np
import sys
import os
import pickle

if __name__ == '__main__':
    caption = sys.argv[1]
    images_path = sys.argv[2]
    out_path = sys.argv[3]
    lst = os.listdir(images_path)
    cis = []
    for i in range(len(lst)):
        print(lst[i])
        cis.append((caption, os.path.join(images_path, lst[i])))
    
    with open(out_path, 'wb') as f:
        pickle.dump(cis, f)
    
