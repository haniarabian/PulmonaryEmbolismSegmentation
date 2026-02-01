clc; clear all; close all;

% Random index selecting

cnt = 127; % Number of total patients
cnt_aug = 20; % Number of augmented images
selected_idx = sort(round(cnt*rand(1,cnt_aug)));

imgoutputpath = 'output CT image path';
refoutputpath = 'output Reference path';

imginputpath = 'input CT image path';
refinputpath = 'input Reference path';

% 'Contrast', [0.8, 1.2]
% 'Brightness', [-0.1, 0.1]


for i = 1 :length(selected_idx)
    baseFileName = append('PE_' , num2str(selected_idx(i)) , '.nii');

    contrastFactor = 0.8 + (0.4*rand(1));
    brightnessFactor = -0.1 + (0.2*rand(1));

    cd(imginputpath)
    I = niftiread(baseFileName);    

    cd(refinputpath)
    M = niftiread(baseFileName);
    
    [a b c] = size(I);
    for j = 1 : c
        I(:,:,j) = imadjust(I(:,:,j), stretchlim(I(:,:,j), [0.02 , 0.98]) , [] , contrastFactor);
        I(:,:,j) = I(:,:,j) + brightnessFactor;
        I(:,:,j) = im2single(min(max(I(:,:,j), 0), 1));
    end

    savename =  append('PE_' , num2str(cnt) , '.nii');
    cnt = cnt + 1;
    
    cd(imgoutputpath)
    niftiwrite(I , savename)

    cd(refoutputpath)
    niftiwrite(M , savename)
    clear I M augmenter
end
