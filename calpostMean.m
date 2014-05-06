function [ postmu, datanum_per_img ] = calpostMean( imgcell, windowsize, stridesize, premu, presigma, Ureduce )
%CALPOSTMEAN Summary of this function goes here
%   This is the function to cal the Mean of input after doing PCA partly.
m = size(imgcell,1);
% postmu = gpuArray(zeros(m,size(Ureduce,2)));
postmu = zeros(m,size(Ureduce,2));
datanum_per_img = zeros(m,1);
parfor i=1:m
    xdata = ximg2data(imgcell{i,1}, windowsize, stridesize);
    
    xdata = removeBlankData(xdata, windowsize, stridesize);
    datanum_per_img(i) = size(xdata,1);
%     xdata = gpuArray(double(xdata));
    % Normalization and PCA reduce.
    xdata = bsxfun(@rdivide, bsxfun(@minus, double(xdata), premu), presigma);
%     xdata = bsxfun(@minus, xdata, premu);
%     xdata = bsxfun(@rdivide, xdata, presigma);
    xdata = xdata * Ureduce;
    
    postmu(i,:) = mean(xdata);
end
% postmu = gather(postmu);
end

