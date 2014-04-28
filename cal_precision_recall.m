function [ precision, recall ] = cal_precision_recall( blank, predyimgcell, yimgcell, thresholdlist )
%CAL_PRECISION_RECALL Summary of this function goes here
%   Detailed explanation goes here
m = size(thresholdlist,1);
precision = zeros(m,1);
recall = zeros(m,1);
disp(m);
for i=1:m
    threshold = thresholdlist(i);
    [precision(i), recall(i)] = cal_precision_recall_threshold(blank, predyimgcell, yimgcell, threshold);
    disp(['Finish index ', num2str(i), 'of ', num2str(m)]);
end

end

function [ oneprecision, onerecall ] = cal_precision_recall_threshold(blank, predyimgcell, yimgcell, threshold)
%     blank = (params.WindowSize-params.StrideSize)/2;
    num_img = size(predyimgcell,1);
    total_tp_p = zeros(num_img,1);
    total_tp_t = zeros(num_img,1);
    total_t = zeros(num_img,1);
    total_p = zeros(num_img,1);
    parfor i=1:num_img
        yimg = yimgcell{i};
        predyimg = predyimgcell{i};
        sz = size(predyimg);
        yimg = yimg(blank+1:blank+sz(1), blank+1:blank+sz(2));
        yimg = single(yimg);
        yimg(yimg==255)=1;
        predyimg(predyimg>=threshold)=1;
        predyimg(predyimg<threshold)=0;
        [tp_p, tp_t, p, t] = cal_one_img(yimg, predyimg);
        total_tp_p(i) = tp_p;
        total_tp_t(i) = tp_t;
        total_t(i) = t;
        total_p(i) = p;
    end
    oneprecision = sum(total_tp_p)/sum(total_p);
    onerecall = sum(total_tp_t)/sum(total_t);
    
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