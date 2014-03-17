function [ rawXmem ] = genRawX( params )
%GENRAWX Summary of this function goes here
%   This is the function to generate the cooresponding X [m n] matrix, m is
%   the num of data record, n is the num of features.

if exist(params.cacheRawXMat, 'file')
    load(params.cacheRawXMat);
    return;
end

WindowSize = 64;
Stride = 16;
tF = @(l) floor((l - WindowSize)/Stride);
% Xend = 1916
% Yend = 1756
Xend = tF(params.numColumns);
Yend = tF(params.numRows);
% This is the num of X data.
total_case = Xend*Yend;
% This is the num of the X data's dimension.
total_dimension = WindowSize^2*3;
rawXmem = zeros(total_case, total_dimension, 'uint8');
% rawXmem = [];

pF = @(ind) (ind - 1)*Stride + 1;
dataNum = Xend*Yend;

% fId = fopen('F:/RawX.dat','w');

count = 0;
per = 0.01;
for i = 1:Xend
    for j = 1:Yend
        count = count + 1;
        if i==1 && j==1
            tic;
        end
        if i==Xend && j == Yend
            toc;
        end
        if count/dataNum >= per
            toc;
            fprintf('%d/%d\n', count, dataNum);
            per = per + 0.01;
            tic;
        end
        xStart = pF(i);
        yStart = pF(j);
        matSize = [WindowSize WindowSize];
%             tempMat = getImgMat(xStart, yStart, matSize, params);
        tempMat = params.BingMat(yStart:yStart+ matSize(2)-1, xStart:xStart+matSize(1)-1,:);
        tempMat = permute(tempMat, [3 1 2]);
        tempMat = tempMat(:)';
        rawXmem(count,:) = tempMat;
%         rawXmem = [rawXmem; tempMat];
%         fwrite(fId, tempMat, 'uint8');
    end
end
assert(count == total_case);
% fclose(fId);
save(params.cacheRawXMat, 'rawXmem', '-v7.3');

% rawXmem = memmapfile('F:/RawX.dat', 'format', {'uint8', [dataNum WindowSize^2*3], 'mat' });
% temp = rawXmem.Data.mat;
% meanMat = mean(temp);
% clear temp;
% temp = single(rawXmem.Data.mat);
% stdMat = std(temp);
% clear rawXmem temp;
% rawXmem = memmapfile('F:/RawX.dat', 'format', {'uint8', [dataNum WindowSize^2*3], 'mat' });
% temp = rawXmem.Data.mat;
% save('meanMat.mat', 'meanMat', 'stdMat');
% temp = bsxfun(@minus, double(temp), meanMat);
% temp = bsxfun(@rdivide, temp, std(temp));
% Sigma = temp'*temp/dataNum;
% [U S V] = svd(Sigma);
% save('pca.mat', 'U', 'S', 'V');
% clear U S V;


end

