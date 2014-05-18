function [ sigma ] = calpostNNSigma( params, predtrainyimgcell, postnnmu )
%CALPOSTNNSIGMA Summary of this function goes here
%   Detailed explanation goes here

m = size(predtrainyimgcell,1);
total = zeros(m,1);
sigma = zeros(m, params.WindowSize^2);

trainydata = params.trainXYimg;
WindowSize = params.WindowSize;
StrideSize = params.StrideSize;

parfor i=1:m
    [dataX, ~] = predyimg2data(predtrainyimgcell{i,1},...
        trainydata{i,2}, WindowSize, StrideSize);
    sigma(i,:) = sum(bsxfun(@minus, dataX, postnnmu).^2);
    total(i,:) = size(dataX,1);
end

sigma = sum(sigma)/sum(total);

end

