function [ mu ] = calpostNNMu( params, predtrainyimgcell )
%CALPOSTNNMU Summary of this function goes here
%   Detailed explanation goes here

m = size(predtrainyimgcell,1);
total = zeros(m,1);
mu = zeros(m, params.WindowSize^2);

trainydata = params.trainXYimg;
WindowSize = params.WindowSize;
StrideSize = params.StrideSize;

parfor i=1:m
    [dataX, ~] = predyimg2data(predtrainyimgcell{i,1},...
        trainydata{i,2}, WindowSize, StrideSize);
    mu(i,:) = sum(dataX);
    total(i,:) = size(dataX,1);
end

mu = sum(mu)/sum(total);
    


end

