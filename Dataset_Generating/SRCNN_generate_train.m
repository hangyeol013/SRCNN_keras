clear;close all;
%% settings
%% ÇÒ¶§¸¶´Ù folder À§Ä¡ test¶û train ¹Ù²ãÁà¾ß ÇÑ´Ù.
folder = 'Train/91_images/HR';
%folder = 'Test/Set5/HR';

savepath_train = 'train_91_scale4.h5';
savepath_test = 'cross_val_Set5_scale4.h5';

size_input = 33;
size_label = 33;
scale = 4;
stride = 14;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.png'));
    
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
    
    im_label = modcrop(image, scale);
    [hei,wid] = size(im_label);
    im_input = imresize(imresize(im_label,1/scale,'bicubic'), scale,'bicubic');

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);

            count=count+1;
            data(:, :, 1, count) = subim_input;
            label(:, :, 1, count) = subim_label;
        end
    end
end

data_dims = size(data);
label_dims = size(label);

%% writing to HDF5

h5create(savepath_train,'/data',data_dims);
h5create(savepath_train, '/label', label_dims);

h5write(savepath_train, '/data', data);
h5write(savepath_train, '/label', label);

h5disp(savepath_train);
