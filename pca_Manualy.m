function [ reduceMat ] = pca_Manualy( input_args )
%PCA_MANUALY Summary of this function goes here
%   Detailed explanation goes here
os = getenv('os');
if strcmp(os, 'Windows_NT')
    dataFloder = 'E:/wuhan/';
else
    dataFloder = '/data/wuhan/';
end
load(fullfile(dataFloder, 'RawX.mat'));
[m, n] = size(rawXmem);
batchsize = 10000;

mu = partExec(rawXmem, batchsize, @sum, 'gpu');
mu = mu/m;
stdHandler = @(mat) sum(bsxfun(@minus, mat, mu).^2);
sigma = sqrt(partExec(rawXmem, batchsize, stdHandler, 'gpu') / m);
% sigma = std(single(rawMat));

covHandler = @(mat) covFun(mat, mu, sigma);
sig = single(partExec(rawXmem, batchsize, covHandler, 'gpu')) / m;
% clear mu sigma;
mu = single(gather(mu));
sigma = single(gather(sigma));

% If the rawMat size is too big, then cov(rawMat) may cause out of memory.
% sig = cov(single(rawMat));
[U, S, ~] = svd(sig);
clear sig;
U = gather(U);
S = gather(S);
Ureduce = U(:,1:10);
% per = sum(latent(1:reduceDimension,:))/sum(latent);
per = sum(sum(S(:,1:reduceDimension)))/sum(sum(S));
fprintf('The remaining covariance is %f', per);
% reduceMat = rawMat * coeff(:,1:reduceDimension);
% reduceMat = zeros(m, reduceDimension, 'single');
% reduceMat = [];
% idx = 1:batchsize:m;
% if idx(end) ~= m
%     idx = [idx m];
% end
% for i=1:size(idx, 2)-1
%     if i ~= size(idx,2)-1
%         temp = rawXmem(idx(i):idx(i+1)-1, :);
% %         reduceMat(idx(i):idx(i+1)-1,:) = temp * Ureduce;
%     else
%         temp = rawMat(idx(i):idx(i+1), :);
% %         reduceMat(idx(i):idx(i+1),:) = temp * Ureduce;
%     end
%     reduceMat = [reduceMat; single(single(temp) * Ureduce)];
% end
rawXmem = single(rawXmem);
rawXmem = bsxfun(@minus, rawXmem, mu);
rawXmem = bsxfun(@rdivide, rawXmem, sigma);
rawXmem = rawXmem * Ureduce;
Xmem = rawXmem;
save(fullfile(dataFloder, 'correct_pcaX.mat'), 'Xmem');
clear Xmem;
reduceMat = rawXmem;

end

function result = covFun(mat, mu, sigma)
    result = bsxfun(@rdivide, bsxfun(@minus, mat, mu), sigma).^2;
    result = result'*result;
end

