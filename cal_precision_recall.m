function [ precision, recall ] = cal_precision_recall( params, predyimgcell, yimgcell, thresholdlist )
%CAL_PRECISION_RECALL Summary of this function goes here
%   Detailed explanation goes here
m = size(thresholdlist,1);
precision = zeros(m,1);
recall = zeros(m,1);
disp(m);
for i=1:m
    threshold = thresholdlist(i);
    [precision(i), recall(i)] = cal_precision_recall_threshold(params, predyimgcell, yimgcell, threshold);
end

end

function [ oneprecision, onerecall ] = cal_precision_recall_threshold(params, predyimgcell, yimgcell, threshold)
    blank = (params.WindowSize-params.StrideSize)/2;
    total_tp_p = 0;
    total_tp_t = 0;
    total_t = 0;
    total_p = 0;
    for i=1:size(predyimgcell,1)
        yimg = yimgcell{i,1};
        predyimg = predyimgcell{i,1};
        sz = size(predyimg);
        yimg = yimg(blank+1:blank+sz(1), blank+1:blank+sz(2));
        yimg = single(yimg);
        yimg(yimg==255)=1;
        predyimg(predyimg>=threshold)=1;
        predyimg(predyimg<threshold)=0;
        [tp_p, tp_t, p, t] = cal_one_img(yimg, predyimg);
        total_tp_p = total_tp_p + tp_p;
        total_tp_t = total_tp_t + tp_t;
        total_t = total_t + t;
        total_p = total_p + p;
    end
    oneprecision = total_tp_p/total_p;
    onerecall = total_tp_t/total_t;
    
end

function [tp_p, tp_t, p, t] = cal_one_img(yimg, predyimg)
    t = nnz(yimg);
    p = nnz(predyimg);
    
    pidx = find(predyimg);
    yimg_temp = bwdist(yimg, 'cityblock');
    pdist = yimg_temp(pidx);
    tp_p = nnz(pdist<=3);
    
    tidx = find(yimg);
    predyimg_temp = bwdist(predyimg, 'cityblock');
    tdist = predyimg_temp(tidx);
    tp_t = nnz(tdist<=3);
    
end