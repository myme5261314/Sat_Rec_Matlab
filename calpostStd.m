function [ postsigma, datanum_per_img ] = calpostStd( imgcell, windowsize, stridesize, premu, presigma, Ureduce, postmu )
%CALPOSTSTD Summary of this function goes here
%   This is the function to cal the Std of input after doing PCA partly.
m = size(imgcell,1);
% g_Ureduce = gpuArray(Ureduce);
% postsigma = gpuArray(zeros(m, size(Ureduce,2)));
postsigma = zeros(m, size(Ureduce,2));
datanum_per_img = zeros(m,1);
parfor i=1:m
    xdata = ximg2data(imgcell{i,1}, windowsize, stridesize);
    
    xdata = removeBlankData(xdata, windowsize, stridesize);
    datanum_per_img(i) = size(xdata,1);
%     xdata = gpuArray(double(xdata));
    xdata = bsxfun(@rdivide, bsxfun(@minus, double(xdata), premu), presigma);
    xdata = xdata * Ureduce;
    xdata = bsxfun(@minus, xdata, postmu);
    xdata = sum(xdata.^2)/size(xdata,1);
    postsigma(i,:) = xdata;
end
% postsigma = gather(postsigma);

end

