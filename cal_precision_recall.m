function [ precision, recall ] = cal_precision_recall( params, predyimgcell, yimgcell, thresholdlist )
%CAL_PRECISION_RECALL Summary of this function goes here
%   Detailed explanation goes here
m = size(thresholdlist,1);
precision = zeros(m,1);
recall = zeros(m,1);
for i=1:m
    threshold = thresholdlist(i);
    [precision(i), recall(i)] = cal_precision_recall_threshold(params, predyimgcell, yimgcell, threshold);
end

end

function [ oneprecision, onerecall ] = cal_precision_recall_threshold(params, predyimgcell, yimgcell, threshold)
    blank = (params.WindowSize-params.StrideSize)/2;
    total_tp = 0;
    total_t = 0;
    tota_p = 0;
    for i=1:size(predyimgcell,1)
        yimg = yimgcell{1,i};
        predyimg = predyimgcell{i,1};
        sz = size(predyimg);
        yimg = yimg(blank+1:blank+sz(1), blank+1:blank+sz(2));
        yimg(yimg>=threshold)=1;
        yimg(yimg<threshold)=0;
        [tp, p, t] = cal_one_img(yimg, predyimg);
        total_tp = total_tp + tp;
        total_t = total_t + t;
        total_p = total_p + p;
    end
    oneprecision = total_tp/total_p;
    onerecall = total_tp/total_t;
    
end

function [tp, p, t] = cal_one_img(yimg, predyimg)
    t = nnz(yimg);
    p = nnz(predyimg);
    pidx = find(predyimg);
    yimg = bwdist(yimg, 'city-clock');
    pdist = yimg(pidx);
    tp = nnz(pdist<=3);
end