clear all; clc;

% crop and resize images 

numrows = 256;
numcols = 256;

% Preprocessed CAD PE Data
outputpath = 'output CT image path';
outrefpath = 'output Reference path';

inputpath = 'input CT image path'; % Output of first_preprocess.m
refpath = 'input Reference path'; % Output of first_preprocess.m
number_of_patients = 91; % Number of CAD PE participants
scale = [];
for k = 1 : number_of_patients
    cnt = k;
    cd(refpath)
    name = append('cadperef',num2str(k),'.nii');
    M = niftiread(name);
    cd(inputpath)
    name = append('cadpe',num2str(k),'.nii');
    V = niftiread(name);
    [a b c] = size(V);
    XY = V(:,:,floor(c/2));
    [counts,x] = imhist(XY);
    T = otsuthresh(counts);
    BW = imbinarize(XY,T);
    BW = imcomplement(BW);
    BW = imclearborder(BW);
    BW = imfill(BW, 'holes');
    radius = 3;
    decomposition = 0;
    se = strel('disk',radius,decomposition);
    BW = imopen(BW, se);
    % BW = imdilate(BW, se);
    BW = imclose(BW, se);

    % maskedImageXY = XY;
    % maskedImageXY(~BW) = 0;
    % maskedImageXY(BW) = 1;
    % imshow(maskedImageXY, [])

    x = find(sum(BW,2)>0);
    y = find(sum(BW, 1)>0);
    newV = V(x(1):x(end) , y(1):y(end) , :);
    newM = M(x(1):x(end) , y(1):y(end) , :);
    scale(cnt , :) = size(newV);

    newV = imresize3(newV , [numrows numcols c]);
    newM = imresize3(newM , [numrows numcols c]);

    name = append('PE_' , num2str(cnt) , '.nii');
    cd(outputpath)
    niftiwrite(newV, name);

    cd(outrefpath)
    niftiwrite(newM, name);

    clear newV newM V M x y BW idx check
end

% Preprocessed FUMPE Data
outputpath = 'output CT image path';
outrefpath = 'output Reference path';

inputpath = 'input CT image path'; % Output of first_preprocess.m
refpath = 'input Reference path'; % Output of first_preprocess.m

number_of_patients = 35; % Number of FUMPE participants

for k = 1 : number_of_patients
    cnt = cnt + 1;
    cd(refpath)
    name = append('fumperef',num2str(k),'.nii');
    M = niftiread(name);
    cd(inputpath)
    name = append('fumpe',num2str(k),'.nii');
    V = niftiread(name);
    [a b c] = size(V);
    XY = V(:,:,floor(c/2));
    [counts,x] = imhist(XY);
    T = otsuthresh(counts);
    BW = imbinarize(XY,T);
    BW = imcomplement(BW);
    BW = imclearborder(BW);
    BW = imfill(BW, 'holes');
    radius = 3;
    decomposition = 0;
    se = strel('disk',radius,decomposition);
    BW = imopen(BW, se);
    % BW = imdilate(BW, se);
    BW = imclose(BW, se);

    % maskedImageXY = XY;
    % maskedImageXY(~BW) = 0;
    % maskedImageXY(BW) = 1;
    % imshow(maskedImageXY, [])

    x = find(sum(BW,2)>0);
    y = find(sum(BW, 1)>0);
    newV = V(x(1):x(end) , y(1):y(end) , :);
    newM = M(x(1):x(end) , y(1):y(end) , :);
    scale(cnt , :) = size(newV);

    newV = imresize3(newV , [numrows numcols c]);
    newM = imresize3(newM , [numrows numcols c]);

    name = append('PE_' , num2str(cnt) , '.nii');
    cd(outputpath)
    niftiwrite(newV, name);

    cd(outrefpath)
    niftiwrite(newM, name);

    clear newV newM V M x y BW idx check
end

% xlswrite('scale.xlsx' , scale); 
