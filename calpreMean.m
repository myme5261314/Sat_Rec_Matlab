function [ premu ] = calpreMean( imgcell, windowsize, stridesize )
%CALMEAN Summary of this function goes here
%   This is the function to cal the Mean of input before doing PCA partly.
m = size(imgcell,1);
premu = zeros(m,windowsize^2*3);
for i=1:m
    xdata = ximg2data(imgcell{i,1}, windowsize, stridesize);
%     mu(i,:) = mean(gpuArray(xdata));
    premu(i,:) = mean(xdata);
end
    


end

