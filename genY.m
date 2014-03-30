function [ rawYmem ] = genY( params )
%GENY Summary of this function goes here
%   This is the function to generate the cooresponding Y [m n] matrix, m is
%   the num of data record, n is the num of output.

if exist(params.cacheRawYMat, 'file')
    load(params.cacheRawYMat);
    return;
end

WindowSize = 64;
Stride = 16;
tF = @(l) floor((l - WindowSize)/Stride);
% Xend = 1916
% Yend = 1756
Xend = tF(params.numColumns);
Yend = tF(params.numRows);
pF = @(ind) (ind - 1)*Stride + 1 + (WindowSize-Stride)/2;
dataNum = Xend*Yend;

rawYmem = zeros(dataNum, Stride^2);

%     fId = fopen('F:/RawY.dat','w');
%     memMat = memmapfile('OSMMat.dat', 'format', {'double', [params.numRows params.numColumns], 'mat' });
%     rawYmem = memMat;
%     yMat = memMat.Data.mat;
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
%             matSize = [WindowSize WindowSize];
%             tempMat = getImgMat(xStart, yStart, matSize, params);
%             tempMat = permute(tempMat, [3 1 2]);
%             tempMat = yMat(yStart:yStart+Stride-1, xStart:xStart+Stride-1);
        tempMat = params.extendMat(yStart:yStart+Stride-1, xStart:xStart+Stride-1);
        tempMat = full(tempMat(:))';
        rawYmem(count,:) = tempMat;
    end
end
assert(count==dataNum);
save(params.cacheRawYMat, 'rawYmem', '-v7.3');

end


