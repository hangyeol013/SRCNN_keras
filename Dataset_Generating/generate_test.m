clear;close all;
%% settings
dataset = 'Set5';
scale = 2;

folder_hr = ['Test/', dataset, '/HR'];
folder_lr = ['Test/', dataset, '/LR/scale', num2str(scale)]; 

%% generate data
filepaths_hr = [];
filepaths_hr = [filepaths_hr; dir(fullfile(folder_hr, '*.png'))];

filepaths_lr = [];
filepaths_lr = [filepaths_lr; dir(fullfile(folder_lr, '*.png'))];

for i = 1 : length(filepaths_hr)
    
    im_hr = imread(fullfile(folder_hr,filepaths_hr(i).name));
    im_hr = modcrop(im_hr, scale);
    [hr_hei, hr_wid, hr_channel] = size(im_hr);
    %im_hr = double(im_hr);
    
    im_lr = imread(fullfile(folder_lr,filepaths_lr(i).name));
    [lr_hei, lr_wid, lr_channel] = size(im_lr);
    %im_lr = double(im_lr);
    
    im_bicubic = imresize(im_lr, scale, 'bicubic');
    [bicubic_hei, bicubic_wid, bicubic_channel] = size(im_bicubic);
    %im_bicubic = double(im_bicubic);
    
    % ycbcr 데이터 만드는 부분
    % Set14 흑백 이미지 고려
    if size(im_hr,3) > 1
        im_hr_ycbcr = rgb2ycbcr(im_hr);
    elseif size(im_hr,3) == 1
        img_hr = zeros(hr_hei, hr_wid, 3);
        im_hr_ycbcr = rgb2ycbcr(img_hr);
        im_hr_ycbcr(:, :, 1) = im_hr;
    end
    
    if size(im_lr,3) > 1
        im_lr_ycbcr = rgb2ycbcr(im_lr);
    elseif size(im_lr,3) == 1
        img_lr = zeros(lr_hei, lr_wid, 3);
        im_lr_ycbcr = rgb2ycbcr(img_lr);
        im_lr_ycbcr(:, :, 1) = im_lr;
    end
    
    if size(im_bicubic,3) > 1
        im_bicubic_ycbcr = rgb2ycbcr(im_bicubic);
    elseif size(im_lr,3) == 1
        img_bicubic = zeros(bicubic_hei, bicubic_wid, 3);
        im_bicubic_ycbcr = rgb2ycbcr(img_bicubic);
        im_bicubic_ycbcr(:, :, 1) = im_bicubic;
    end

    
    %rgb 데이터 만드는 부분
    %Set14 흑백 이미지 고려
    if size(im_hr, 3) == 1
        im_rgb_hr = zeros(hr_hei, hr_wid, 3);
        im_rgb_hr(:, :, 1) = 1.164 * (im_hr - 16);
        im_rgb_hr(:, :, 2) = 1.164 * (im_hr - 16);
        im_rgb_hr(:, :, 3) = 1.164 * (im_hr - 16);
        im_hr = im_rgb_hr;
    end
    
    if size(im_lr, 3) == 1
        im_rgb_lr = zeros(lr_hei, lr_wid, 3);
        im_rgb_lr(:, :, 1) = 1.164 * (im_lr - 16);
        im_rgb_lr(:, :, 2) = 1.164 * (im_lr - 16);
        im_rgb_lr(:, :, 3) = 1.164 * (im_lr - 16);
        im_lr = im_rgb_lr;
    end
    
    if size(im_bicubic, 3) == 1
        im_rgb_bicubic = zeros(bicubic_hei, bicubic_wid, 3);
        im_rgb_bicubic(:, :, 1) = 1.164 * (im_bicubic - 16);
        im_rgb_bicubic(:, :, 2) = 1.164 * (im_bicubic - 16);
        im_rgb_bicubic(:, :, 3) = 1.164 * (im_bicubic - 16);
        im_bicubic = im_rgb_bicubic;
    end
    
%     im_hr_rgb = im_hr;
%     im_lr_rgb = im_lr;
%     im_bicubic_rgb = im_bicubic;
    
    im_hr_ycbcr = double(im_hr_ycbcr);
    im_lr_ycbcr = double(im_lr_ycbcr);
    im_bicubic_ycbcr = double(im_bicubic_ycbcr);

    im_hr_rgb = double(im_hr);
    im_lr_rgb = double(im_lr);
    im_bicubic_rgb = double(im_bicubic);
    

    filename = ['Test/test_mat/', dataset, '/scale', num2str(scale), '/', dataset, '_scale', num2str(scale), '_', num2str(i), '.mat'];
    save(filename, 'im_hr_ycbcr', 'im_hr_rgb', 'im_bicubic_ycbcr', 'im_bicubic_rgb', 'im_lr_ycbcr', 'im_lr_rgb');
    
end