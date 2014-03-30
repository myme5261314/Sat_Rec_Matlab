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
% sig = zeros(n, n);
batchsize = 1000;

mu = partExec(rawMat, batchsize, @sumFun);
stdHandler = @(mat) stdFun(mat, mu);
sigma = sqrt(partExec(rawMat, batchsize, stdHandler));
% sigma = std(single(rawMat));

covHandler = @(mat) covFun(mat, mu, sigma);
sig = partExec(rawMat, batchsize, covHandler);

% If the rawMat size is too big, then cov(rawMat) may cause out of memory.
% sig = cov(single(rawMat));
[U, S, ~] = svd(sig);
clear sig;
Ureduce = U(:,1:reduceDimension);
% per = sum(latent(1:reduceDimension,:))/sum(latent);
per = sum(sum(S(:,1:reduceDimension)))/sum(sum(S));
fprintf('The remaining covariance is %f', per);
% reduceMat = rawMat * coeff(:,1:reduceDimension);
reduceMat = zeros(m, reduceDimension, 'single');
for i=1:batchsize:m
    reduceMat(i:i+batchsize-1,:) = single(rawMat(i:i+batchsize-1,:) * Ureduce);
end
r = mod(m, batchsize);
if r~=0
    reduceMat(end-r+1:end,:) = single(rawMat(end-r+1:end,:) * Ureduce);
end
% reduceMat = single(rawMat) * Ureduce;

end



function result = sumFun(mat)

if size(mat,1) ~= 1
    result =sum(mat);
else
    result = mat;
end
end

function result = stdFun(mat, mu)

temp = power(bsxfun(@minus, mat, mu), 2);
if size(mat,1) ~= 1
    result =sum(temp);
else
    result = temp;
end
end

function result = covFun(mat, mu, sigma)

if size(mat, 1) ~=1
    temp = bsxfun(@rdivide, bsxfun(@minus, mat, mu), sigma);
else
    temp = (mat-mu)./sigma;
end
result = temp'*temp;

end
