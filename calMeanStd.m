function [ output_args ] = calMeanStd( input_args )
%CALMEANSTD Summary of this function goes here
%   This is the function to calculate the mean and std of the Raw X Matrix

% mean part
fId = fopen('F:/RawX.dat', 'r');
% fNum = 1916;
% fNum = 1916/4;
f = dir('F:/RawX.dat');
total = f.bytes/(64*64*3);
% total = 1916*1756;
fNum = 4096;
t = floor(total/fNum);
r = mod(total,fNum);
m = zeros(1, 64^2*3);
per = 0.01;
tic;
for i = 1:t
    if i/t >= per
        toc;
        fprintf('%d/%d\n', i, t);
        per = per + 0.01;
        tic;
    end
    
    tempMat = fread(fId, [64^2*3 fNum], 'uint8');
    tempMat = mean(tempMat,2)';
    m = m + tempMat;
end
tempMat = fread(fId, [64^2*3 r], 'uint8');
tempMat = mean(tempMat,2)';
m = (m*fNum + tempMat*r)/total;
toc;
fclose(fId);
RawXmean = m;
save('RawXmean.mat', 'RawXmean');
fprintf('End of mean stage\n');

% Std part
fId = fopen('F:/RawX.dat', 'r');
fNum = 4096;
t = floor(total/fNum);
r = mod(total, fNum);
m = zeros(1, 64^2*3);
per = 0.01;
tic;
for i = 1:t
    if i/t >= per
        toc;
        fprintf('%d/%d\n', i, t);
        per = per + 0.01;
        tic;
    end
    
    tempMat = fread(fId, [64^2*3 fNum], 'uint8');
    tempMat = calStd(tempMat, RawXmean);
    m = m + tempMat;
end
tempMat = fread(fId, [64^2*3 r], 'uint8');
tempMat = calStd(tempMat, RawXmean);
m = m + tempMat;
toc;
m = m/total;
m = sqrt(m);
RawXStd = m;
save('RawXStd.mat', 'RawXStd');
fclose(fId);
fprintf('End of std stage\n');



end

function  [newmat] = calStd(mat, meanMat)
    newmat = mat';
    newmat = bsxfun(@minus, newmat, meanMat);
%     newmat = mat' - meanMat;
    newmat = newmat.^2;
    newmat = sum(newmat);
end
