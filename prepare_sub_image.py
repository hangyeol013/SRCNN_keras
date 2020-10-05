import os
import cv2
import numpy as np
import h5py


Train_path_91 = "/Dataset/T91"
Train_path_200 = "/Dataset/BSD500/data/images/train"
Cross_val_path_5 = "/Dataset/Set5"
Cross_val_path_200 = "/Dataset/BSD500/data/images/val"


Patch_size = 33
F1 = 9
F2 = 5
F3 = 5
Label_size = Patch_size - F1 - F2 - F3 + 3
conv_side = (Patch_size - Label_size) // 2
stride = 14


def prepare_sub_image(_path):
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()
    
    scale = 2

    data = []
    label = []
    
    sub_image_num = 0
    
    for i in range(nums):
        
        name = _path + '/' + names[i]
        
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        hr_img_YCrCb = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img_Y = hr_img_YCrCb[:, :, 0]
        
        shape = hr_img_Y.shape
        
        lr_img_Y = cv2.resize(hr_img_Y, dsize = (shape[1] // scale, shape[0] // scale), interpolation = cv2.INTER_CUBIC)        
        # input img = bicubic img
        Input_img_Y = cv2.resize(lr_img_Y, dsize = (shape[1], shape[0]), interpolation = cv2.INTER_CUBIC)
        
        
        ##### 조심 #####
        width_num = ((shape[0] - Patch_size) / stride) + 1
        height_num = ((shape[1] - Patch_size) / stride) + 1
        
        
        sub_image = int(width_num) * int(height_num)

        # patch 짜르는 과정
        
        for j in range(int(width_num)):
            for k in range(int(height_num)):
                
                x = j * stride
                y = k * stride
                
                hr_patch = hr_img_Y[x:x+Patch_size, y:y+Patch_size]
                input_patch = Input_img_Y[x:x+Patch_size, y:y+Patch_size]
                
                
                hr_patch = hr_patch.astype(float) / 255.
                input_patch = input_patch.astype(float) / 255.
                
                hr = np.zeros((1, Label_size, Label_size), dtype = np.double)
                Input = np.zeros((1, Patch_size, Patch_size), dtype = np.double)
                
                
                hr[0, :, :] = hr_patch[conv_side:-conv_side, conv_side:-conv_side]
                Input[0, :, :] = input_patch
                

                data.append(Input)
                label.append(hr)
                
                
        sub_image_num += sub_image
                
                
    data = np.array(data, dtype = float)
    label = np.array(label, dtype = float)
    
    print(data.shape, label.shape)

    print(sub_image_num)
    
    return data, label


def write_hdf5(data, label, output_filename):
    
    x = data.astype(np.float32)
    y = label.astype(np.float32)
    
    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data = x, shape = x.shape)
        h.create_dataset('label', data = y, shape = y.shape)
        
        
def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 2, 3, 1))
        train_label = np.transpose(label, (0, 2, 3, 1))
        
        return train_data, train_label
    
    
if __name__ == '__main__':
    
    data, label = prepare_sub_image(Train_path_91)
    write_hdf5(data, label, 'train_91.h5')
    data, label = prepare_sub_image(Train_path_200)
    write_hdf5(data, label, 'train_200.h5')
    data, label = prepare_sub_image(Cross_val_path_5)
    write_hdf5(data, label, 'cross_val_5.h5')
    data, label = prepare_sub_image(Cross_val_path_200)
    write_hdf5(data, label, 'cross_val_200.h5')
      
