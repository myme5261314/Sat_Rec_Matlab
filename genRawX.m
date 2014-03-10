function [ rawXmem ] = genRawX( params )
%GENRAWX Summary of this function goes here
%   This is the function to generate the cooresponding X [m n] matrix, m is
%   the num of data record, n is the num of features.
WindowSize = 64;
Stride = 16;
% fId = fopen('F:/RawX.dat','w');
tF = @(l) floor((l - WindowSize)/Stride);
% Xend = 1916
% Yend = 1756
Xend = tF(params.numColumns);
Yend = tF(params.numRows);
pF = @(ind) (ind - 1)*Stride + 1;
dataNum = Xend*Yend;
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
        tempMat = getImgMat(xStart, yStart, matSize, params);
        tempMat = permute(tempMat, [3 1 2]);
        tempMat = tempMat(:);
        fwrite(fId, tempMat, 'uint8');
    end
end
fclose(fId);

rawXmem = memmapfile('F:/RawX.dat', 'format', {'double', [dataNum WindowSize^2*3], 'mat' });
temp = double(rawXmem.Data.mat);
temp = bsxfun(@minus, temp, mean(temp));
temp = bsxfun(@rdivide, temp, std(temp));
Sigma = temp'*temp/dataNum;
[U S V] = svd(Sigma);
save('pca.mat', 'U', 'S', 'V');
clear U S V;


end

