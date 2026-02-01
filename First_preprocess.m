clc; clear all; close all;

targetVoxelSize = [1 1 1];

%% CAD-PE data: Reading nrrd data and convert it to nifty

% set the output folder path for CT images
outputpath= 'output CT image path';

% select the input folder path of CT images and its file format
inputpath = 'input CT image path';
origfilename = fullfile(inputpath, '*.nrrd');
origFiles = dir(origfilename);
for k = 1 : length(origFiles)
    cd(inputpath)
    baseFileName = origFiles(k).name;
    medVol = medicalVolume(baseFileName);
    ratios = targetVoxelSize ./ medVol.VoxelSpacing;
    origSize = size(medVol.Voxels);
    newSize = round(origSize ./ ratios);
    origRef = medVol.VolumeGeometry;
    origMapping = intrinsicToWorldMapping(origRef);
    tform = origMapping.A;
    newMapping4by4 = tform.* [ratios([2 1 3]) 1];
    newMapping = affinetform3d(newMapping4by4);
    newRef = medicalref3d(newSize,newMapping);
    newRef = orient(newRef,origRef.PatientCoordinateSystem);
    newVol = resample(medVol,newRef);
    I = newVol.Voxels;
     
    % adjust the CT images to [0 4095]
    I = I + 1024;

    % intensity cropping of lung tissue
    I(find(I>2100)) = 2100;

    % adjust the CT images to [0 1]
    I = single(I);
    I = I ./ 2100;
    
    % write the nifty file
    cd(outputpath)
    outName = append('cadpe',num2str(k),'.nii');
    niftiwrite(I, outName);
    clear baseFileName I outName medVol newRef newVol
    clear origMapping tfrom newMapping4by4 newMapping origSize newSize origRef
end
clear inputpath outputpath k origfilename origFiles

% set the output folder path for annotation
outputpath= 'output Reference path';

% select the input folder path of annotation and its file format
inputpath = 'input Reference path';
origfilename = fullfile(inputpath, '*.nrrd');
origFiles = dir(origfilename);
for k = 1 : length(origFiles)
    cd(inputpath)
    baseFileName = origFiles(k).name;
    medVol = medicalVolume(baseFileName);
    ratios = targetVoxelSize ./ medVol.VoxelSpacing;
    origSize = size(medVol.Voxels);
    newSize = round(origSize ./ ratios);
    origRef = medVol.VolumeGeometry;
    origMapping = intrinsicToWorldMapping(origRef);
    tform = origMapping.A;
    newMapping4by4 = tform.* [ratios([2 1 3]) 1];
    newMapping = affinetform3d(newMapping4by4);
    newRef = medicalref3d(newSize,newMapping);
    newRef = orient(newRef,origRef.PatientCoordinateSystem);
    newVol = resample(medVol,newRef);
    I = newVol.Voxels;
    I(find(I > 0)) = 1;
    I = uint8(I);

    % write the nifty file
    cd(outputpath)
    outName = append('cadperef',num2str(k),'.nii');
    niftiwrite(I, outName);
    clear baseFileName I outName medVol newRef newVol
    clear origMapping tfrom newMapping4by4 newMapping origSize newSize origRef
end
clear inputpath outputpath k origfilename origFiles

%% FUMPE data: Reading dicom data and convert it to nifty

% set the output folder path for CT images
outputpath= 'output CT image path';

% select the input folder path of CT images and its file format
inputpath = 'input CT image path';
origFiles = dir(inputpath);
for k = 3 : length(origFiles)
    newpath = append(inputpath, '\', origFiles(k).name);
    cd(newpath)
    dcmfilename = fullfile(newpath, '*.dcm');
    dcmFiles = dir(dcmfilename);
    for i = 1 : length(dcmFiles)
        baseFileName = dcmFiles(i).name;
        medVol = medicalVolume(baseFileName);
        J(:,:,i) = medVol.Voxels;
        if (i == length(dcmFiles))
            % adjust the CT images to [0 4095]
            J = J + 1024;

            % intensity cropping of lung tissue
            J(find(J>2100)) = 2100;

            % adjust the CT images to [0 1]
            J = single(J);
            J = J ./ 2100;
            ratios = targetVoxelSize ./ medVol.VoxelSpacing;
            origSize = size(J);
            newSize = round(origSize ./ ratios);
            A = [0 medVol.VoxelSpacing(1) 0 medVol.VolumeGeometry.Position(1);
                medVol.VoxelSpacing(2) 0 0 medVol.VolumeGeometry.Position(2);
                0 0 medVol.VoxelSpacing(3) medVol.VolumeGeometry.Position(3);
                0 0 0 1];
            tform = affinetform3d(A);
            origRef = medicalref3d(origSize,tform);
            origRef.PatientCoordinateSystem = medVol.VolumeGeometry.PatientCoordinateSystem;
            newmedVol = medicalVolume(J,origRef);
            newmedVol.SpatialUnits = medVol.SpatialUnits;
            origMapping = intrinsicToWorldMapping(origRef);
            tform = origMapping.A;
            newMapping4by4 = tform.* [ratios([2 1 3]) 1];
            newMapping = affinetform3d(newMapping4by4);
            newRef = medicalref3d(newSize,newMapping);
            newRef = orient(newRef,origRef.PatientCoordinateSystem);
            newVol = resample(newmedVol,newRef);

            I = newVol.Voxels;

            clear baseFileName medVol newRef newVol newmedvol
            clear origMapping tfrom newMapping4by4 newMapping origSize newSize origRef
        end
    end
    sizeI(k-2 , :) = size(I);
    
    % write the nifty file
    cd(outputpath)
    outName = append('fumpe', num2str(k-2), '.nii');
    niftiwrite(I, outName);
    clear I J outName newpath dcmfilename dcmFiles
end
clear inputpath outputpath k origFiles

% set the output folder path for annotation
outputpath= 'output Reference path';

% select the input folder path of annotation and its file format
inputpath = 'input Reference path';
origfilename = fullfile(inputpath, '*.mat');
origFiles = dir(origfilename);
for k = 1 : length(origFiles)
    cd(inputpath)
    baseFileName = origFiles(k).name;
    J = load(baseFileName);    
    J = uint8(J.Mask);
    J = flip(imrotate(J,-90),2);
    J(find(J>0)) = 1;
    I = imresize3(J , sizeI(k,:));

    % write the nifty file
    cd(outputpath)
    outName = append('fumperef', num2str(k), '.nii');
    niftiwrite(I, outName);
    clear I J outName baseFileName ratio
end
clear inputpath outputpath k origFiles origfilename sizeI