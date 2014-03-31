function [ Xmem ] = preprocessX( params )
%PREPROCESSX Summary of this function goes here
%   This is the function to do the whitening and dimension reduce work.
if ~exist('E:/wuhan/pcaX.mat', 'file')
    Xmem = pca_Reduce(params.rawXmem, 4096);
%     X = bsxfun(@rdivide, bsxfun(@minus, X, min(X)), max(X)-min(X));
    save('E:/wuhan/pcaX.mat', 'Xmem', '-v7.3');
else
    load('E:/wuhan/pcaX.mat');
end
batchsize = 1000;
% mu = mean(Xmem);
mu = partExec(Xmem, batchsize, @sumFun, 'gpu');
mu = mu/size(Xmem, 1);
stdHandler = @(mat) stdFun(mat, mu);
sigma = sqrt(partExec(Xmem, batchsize, stdHandler, 'gpu')/ size(Xmem, 1));
% sigma = std(single(Xmem));
mu = gather(mu);
sigma = gather(sigma);
Xmem = bsxfun(@rdivide, bsxfun(@minus, Xmem, single(mu)), single(sigma));

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