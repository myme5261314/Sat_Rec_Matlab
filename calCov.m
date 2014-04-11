function [ sig ] = calCov( imgcell, windowsize, stridesize, mu, sigma )
%CALCOV Summary of this function goes here
%   This is the function to cal the Cov of input partly.
m = size(imgcell,1);
% sigma = gpuArray(zeros(m,windowsize^2*3));
sig = zeros(windowsize^2*3, windowsize^2*3);
for i=1:m
    xdata = ximg2data(imgcell{i,1}, windowsize, stridesize);
%     xdata = gpuArray(double(xdata));
    xdata = double(xdata);
    xdata = bsxfun(@minus, xdata, mu);
    xdata = bsxfun(@rdivide, xdata, sigma);
    n = size(xdata,1);
    xdata = xdata'*xdata/n;
    sig = sig + xdata;
end
sig = sig/m;

end

