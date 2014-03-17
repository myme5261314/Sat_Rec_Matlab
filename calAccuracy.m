function [ precision, recall ] = calAccuracy( Yorigin, Ypredict, params )
%CALACCURACY Summary of this function goes here
%   This is the function to calculate the precision and recall of the
%   predict.
%   Notice: only use Y(25:end-40,25:end-40) to calculate.
%   Comment: precision = tp/(tp+fp).
%   Comment: recall = tp/(tp+fn).
%   Comment: tp=> true positive, fp=> false positive, fn=>false negative.
%   Comment: precision = predict road pixels that has at least one true
%   road pixel around/predict road pixels.
%   Comment: recall tp = true road pixels that has at least one predict 
%   road pixel around/true road pixels.
tF = @(mat) mat(25:end-40,25:end-40);
Yorigin = tF(Yorigin);
Ypredict = tF(Ypredict);
threshold = 0.7;
Yorigin_temp = full(Yorigin);
Ypredict_temp = Ypredict;
Yorigin_temp(Yorigin_temp>=threshold) = 1;
Yorigin_temp(Yorigin_temp<threshold) = 0;
Ypredict_temp(Ypredict_temp>=threshold) = 1;
Ypredict_temp(Ypredict_temp<threshold) = 0;
[m, n] = size(Yorigin);
tp = 0;
fp = 0;
fn = 0;
% This result contains the all positive ones.
[rows, cols] = find(Ypredict_temp);
for i=1:length(rows)
    x = cols(i);
    y = rows(i);
    [irange, jrange] = getAroundMat(y, x, [m, n]);
    isPositiveRight = any(any(Yorigin_temp(irange, jrange)));
    if isPositiveRight
        tp = tp + 1;
    else
        fp = fp + 1;
    end
end
precision = tp/(tp+fp);
% This result contains the all true ones.
tp = 0;
fn = 0;
[rows, cols] = find(Yorigin_temp);
for i=1:length(rows)
    x = cols(i);
    y = rows(i);
    [irange, jrange] = getAroundMat(y, x, [m, n]);
    isTruePositive = any(any(Ypredict_temp(irange, jrange)));
    if isTruePositive
        tp = tp + 1;
    else
        fn = fn + 1;
    end
end
recall = tp/(tp+fn);


end

function [irange, jrange] = getAroundMat(i, j, matSize)
    m = matSize(1);
    n = matSize(2);
    istart = max(1, i-3);
    iend = min(m, i+3);
    jstart = max(1, j-3);
    jend = min(n, j+3);
    irange = istart:iend;
    jrange = jstart:jend;
    
end

