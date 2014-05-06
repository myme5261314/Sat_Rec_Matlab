function [ premu, datanum_per_img ] = calpreMean( imgcell, windowsize, stridesize )
%CALMEAN Summary of this function goes here
%   This is the function to cal the Mean of input before doing PCA partly.
m = size(imgcell,1);
premu = zeros(m,windowsize^2*3);
datanum_per_img = zeros(m,1);
parfor i=1:m
    xdata = ximg2data(imgcell{i,1}, windowsize, stridesize);
    
    xdata = removeBlankData(xdata, windowsize, stridesize);
    datanum_per_img(i) = size(xdata,1);
%     mu(i,:) = mean(gpuArray(xdata));
    premu(i,:) = mean(xdata);
end
    


end

