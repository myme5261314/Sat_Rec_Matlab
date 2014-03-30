function [ reduceMat ] = pca_Reduce( rawMat, reduceDimension )
%PCA_REDUCE Summary of this function goes here
%   This is the function to perform the pca dimension reduce operation. The
%   input is rawMat [m n](m data case, n feature num)
%   reduceDimension is the num of feature dimension after pca reduce.
%   The output is the data after pca reduce.
% rawMat = bsxfun(@minus, single(rawMat), mean(rawMat));
% rawMat = bsxfun(@rdivide, rawMat, max(rawMat) - min(rawMat));

% [coeff, score, latent] = pca(rawMat);
[m, n] = size(rawMat);
sig = zeros(n, n);
mu = single(mean(rawMat));
sigma = std(single(rawMat));
batchsize = 1000;
for i = 1:batchsize:m
    temp = rawMat(i:i+batchsize-1, :);
    temp = bsxfun(@rdivide, bsxfun(@minus, temp, mu), sigma);
    sig = sig + temp'*temp;
end
r = mod(m, batchsize);
if r ~= 0
    temp = rawMat(end-r+1:end, :);
    temp = bsxfun(@rdivide, bsxfun(@minus, temp, mu), sigma);
    sig = sig + temp'*temp;
end
sig = sig/m;
% If the rawMat size is too big, then cov(rawMat) may cause out of memory.
% sig = cov(single(rawMat));
[U, S, ~] = svd(sig);
clear sig;
Ureduce = U(:,1:reduceDimension);
% per = sum(latent(1:reduceDimension,:))/sum(latent);
per = sum(sum(S(:,1:reduceDimension)))/sum(sum(S));
fprintf('The remaining covariance is %f', per);
% reduceMat = rawMat * coeff(:,1:reduceDimension);
reduceMat = single(rawMat) * Ureduce;

end

