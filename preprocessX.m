function [ Xmem ] = preprocessX( params )
%PREPROCESSX Summary of this function goes here
%   This is the function to do the whitening and dimension reduce work.
if ~exist('E:/wuhan/pcaX.mat', 'file')
    Xmem = pca_Reduce(params.rawXmem, size(params.rawXmem, 2));
%     X = bsxfun(@rdivide, bsxfun(@minus, X, min(X)), max(X)-min(X));
    save('E:/wuhan/pcaX.mat', 'X');
else
    load('E:/wuhan/pcaX.mat');
end
mu = mean(Xmem);
sigma = std(single(Xmem));
Xmem = bsxfun(@rdivide, bsxfun(@minus, Xmem, mu), sigma);

end

