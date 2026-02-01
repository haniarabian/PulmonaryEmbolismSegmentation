clc; clear all; close all;

% remove empty slices and negative subjects from cropped data
inputpath = 'input CT images path'; % Output of second_preprocess.m
refpath = 'output CT images path'; % Output of second_preprocess.m

outputpath = 'input Reference path';
outrefpath = 'output Reference path';
number_of_patients = 126; % % Number of total participants
slice = [];
for k = 1 : number_of_patients
    cd(refpath)
    name = append('PE_',num2str(k),'.nii');
    M = niftiread(name);    
    check = reshape(sum(sum(M,1),2) , 1 , []);
    idx = find(check>0);
    emp = find(check == 0);
    if length(idx) >= 1
        cd(inputpath)
        V = niftiread(name);
        newV = V(: , : , idx);
        newM = M(: , : , idx);

        slice.pe{k} = idx;
        % slice.non{k} = emp;
        
        cd(outputpath)
        niftiwrite(newV, name);

        cd(outrefpath)
        niftiwrite(newM, name);
    
    else
        slice.pe{k} = [0];
        slice.pe{k} = emp;
    end

    clear idx newV newM M V name emp
end

save('slice')

