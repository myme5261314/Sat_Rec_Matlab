function [ presigma ] = calpreStd( imgcell, windowsize, stridesize, mu )
%CALSTD Summary of this function goes here
%   This is the function to cal the Std of input before doing PCA partly.
m = size(imgcell,1);
% sigma = gpuArray(zeros(m,windowsize^2*3));
presigma = zeros(m, windowsize^2*3);
for i=1:m
    xdata = ximg2data(imgcell{i,1}, windowsize, stridesize);
%     xdata = gpuArray(double(xdata));
    xdata = double(xdata);
    xdata = bsxfun(@minus, xdata, mu);
    xdata = sum(xdata.^2)/size(xdata,1);
    presigma(i,:) = xdata;
end
% sigma = gather(sigma);

end

