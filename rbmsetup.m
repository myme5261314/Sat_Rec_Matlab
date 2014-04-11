function [ rbm ] = rbmsetup( params, hidden_layer_size, opts )
%RBMSETUP Summary of this function goes here
%   Detailed explanation goes here
    n = params.reduce;
    rbm.precision = 'single';
    if strcmp(rbm.precision, 'single')
        rbm.premu = single(params.premu);
        rbm.presigma = single(params.presigma);
        rbm.Ureduce = single(params.Ureduce);
        rbm.postmu = single(params.postmu);
        rbm.postsigma = single(params.postsigma);
    else
        rbm.premu = double(params.premu);
        rbm.presigma = double(params.presigma);
        rbm.Ureduce = double(params.Ureduce);
        rbm.postmu = double(params.postmu);
        rbm.postsigma = double(params.postsigma);
    end
    rbm.alpha = opts.alpha;
    rbm.momentum = opts.momentum;
    rbm.W = 0.01*randn(n, hidden_layer_size, rbm.precision);
%     rbm.W = gpuArray(rbm.W);
    rbm.vW = zeros(n, hidden_layer_size, rbm.precision);
%     rbm.vW = gpuArray(rbm.vW);
    % Visible unit bias
    rbm.b = zeros( 1, n, rbm.precision );
%     rbm.b = gpuArray(rbm.b);
    rbm.vb = zeros( 1, n, rbm.precision );
%     rbm.vb = gpuArray(rbm.vb);
    % Hidden unit bias;
    rbm.c = ones( 1, hidden_layer_size, rbm.precision )*-4;
%     rbm.c = gpuArray(rbm.c);
    rbm.vc = zeros( 1, hidden_layer_size, rbm.precision );
%     rbm.vc = gpuArray(rbm.vc);

end

