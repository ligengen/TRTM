import numpy as np
import os
import sys
import glob
import random

def check_flat(path):
    arr = np.loadtxt(path)
    max_z = np.max(arr[:, 2])
    if max_z <= 0.0005:
        return True
    return False


base_dir = '/Users/ligen/Desktop/cloth_recon/Depth_map/50/'

def check_similar():
    all_files = []
    for file in glob.glob(base_dir + '*.txt'):
        all_files.append(file)
    for i in range(len(all_files)):
        print(i)
        arr = np.loadtxt(all_files[i])
        flag = False
        for j in range(i+1, len(all_files)):
            arr2 = np.loadtxt(all_files[j])
            if(np.allclose(arr, arr2, atol=0.2)):
                flag = True
                print(all_files[i], " ", all_files[j])
        if flag == False and i % 20 == 0:
            print(all_files[i], "no similar files")

def rename():
    for file in glob.glob(base_dir + '*.png'):
        name = file[46:51]
        os.rename(file, base_dir + '%s.png' % name)

def train_val_split():
    # 8412
    # 3605
    all_files = [i for i in range(12017)]
    val_list = random.sample(all_files, 3605)
    train_list = [i for i in all_files if i not in val_list]
    print(set(train_list) & set(val_list))
    print("size of train: ", len(train_list))
    print("size of val: ", len(val_list))
    for i in range(len(val_list)):
        os.rename(base_dir + '%05d.png' % val_list[i], base_dir + 'val/%05d.png' % i)
        os.rename(base_dir + '%05d.txt' % val_list[i], base_dir + 'val/%05d.txt' % i)

    for i in range(len(train_list)):
        os.rename(base_dir + '%05d.png' % train_list[i], base_dir + 'train/%05d.png' % i)
        os.rename(base_dir + '%05d.txt' % train_list[i], base_dir + 'train/%05d.txt' % i)


if __name__ == '__main__':
    # rename()
    # train_val_split()
    for i in range(4303):
        os.rename('/Users/ligen/Desktop/cloth_recon/Depth_map/50/%05d.txt' % i, '/Users/ligen/Desktop/cloth_recon/Depth_map/50/test/%05d.txt'%i)
    #     os.rename('/Users/ligen/Desktop/cloth_recon/Depth_map/50/%05d.png' % i, '/Users/ligen/Desktop/cloth_recon/Depth_map/50/test/%05d.png'%i)