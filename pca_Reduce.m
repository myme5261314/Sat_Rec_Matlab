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

mu = partExec(rawMat, batchsize, @sumFun, 'gpu');
mu = mu/m;
stdHandler = @(mat) stdFun(mat, mu);
sigma = sqrt(partExec(rawMat, batchsize, stdHandler, 'gpu') / m);
% sigma = std(single(rawMat));
mu = gather(mu);
sigma = gather(sigma);
covHandler = @(mat) covFun(mat, mu, sigma);
sig = single(partExec(rawMat, batchsize, covHandler)) / m;


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
idx = 1:batchsize:m;
if idx(end) ~= m
    idx = [idx m];
end
for i=1:size(idx, 2)-1
    if i ~= size(idx,2)-1
        temp = rawMat(idx(i):idx(i+1)-1, :);
        temp = single(temp);
        reduceMat(idx(i):idx(i+1)-1,:) = temp * Ureduce;
    else
        temp = rawMat(idx(i):idx(i+1), :);
        temp = single(temp);
        reduceMat(idx(i):idx(i+1),:) = temp * Ureduce;
    end
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
