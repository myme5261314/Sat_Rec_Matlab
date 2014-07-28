function [ g_vW, g_vb, err ] = calRBMGradient( batchX, g_premu, g_presigma, g_Ureduce, g_postmu, g_postsigma, g_W, g_c, g_b, g_L2, g_batchsize, g_alpha )
%CALRBMGRADIENT Summary of this function goes here
%   Detailed explanation goes here

batch = single(batchX);
batch = gpuArray(batch);
%% Preprocess the Raw Batch X.
batch = bsxfun(@rdivide, bsxfun(@minus, batch, g_premu), g_presigma);
batch = batch * g_Ureduce;
batch = bsxfun(@rdivide, bsxfun(@minus, batch, g_postmu), g_postsigma);

g_v1 = batch;
g_h1 = sigm(bsxfun(@plus, g_c, g_v1*g_W));
g_c1 = g_v1' * g_h1;
g_h1 = single(g_h1 > gpuArray.rand(size(g_h1), 'single'));
g_v2 = bsxfun(@plus, g_b, g_h1*g_W') + gpuArray.randn(size(g_v1));
clear g_h1;
g_h2 = sigm(bsxfun(@plus, g_c, g_v2*g_W));
g_c1 = g_c1 - g_v2' * g_h2;
clear g_h2;

err = sum(mean((g_v1-g_v2).^2));

g_c1 = g_c1/g_batchsize;
g_vW = g_alpha * ( g_c1 - g_L2*g_W );
clear g_c1;
g_vb = g_alpha * mean(g_v1 - g_v2);
clear g_v1 g_v2;


end

