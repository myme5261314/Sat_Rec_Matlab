function [ presigma, datanum_per_img ] = calpreStd( imgcell, windowsize, stridesize, mu )
%CALSTD Summary of this function goes here
%   This is the function to cal the Std of input before doing PCA partly.
m = size(imgcell,1);
% sigma = gpuArray(zeros(m,windowsize^2*3));
presigma = zeros(m, windowsize^2*3);
datanum_per_img = zeros(m,1);
parfor i=1:m
    xdata = ximg2data(imgcell{i,1}, windowsize, stridesize);
    
    xdata = removeBlankData(xdata, windowsize, stridesize);
    datanum_per_img(i) = size(xdata,1);
%     xdata = gpuArray(double(xdata));
    xdata = double(xdata);
    xdata = bsxfun(@minus, xdata, mu);
    xdata = sum(xdata.^2)/size(xdata,1);
    presigma(i,:) = xdata;
end
% sigma = gather(sigma);

end

