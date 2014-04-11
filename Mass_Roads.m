function [ nn ] = Mass_Roads( input_args )
%MASS_ROADS Summary of this function goes here
%   Detailed explanation goes here
params = initParams();
hidden_layer_size = 4096*3;
opts.numepochs =   2;
opts.batchsize = 64;
opts.momentum  =   0.9;
opts.alpha     =   0.001;
opts.L2 = 0.0002;
if ~exist(params.cacheRBM, 'file')
    rbm = rbmsetup(params, hidden_layer_size, opts);
    rbm = rbm_train(params, rbm, opts);
    save(params.cacheRBM, 'rbm', '-v7.3');
else
    load(params.cacheRBM);
end
figure; visualize(rbm.W);   %  Visualize the RBM weights

w = [rbm.c; rbm.W];
clear rbm;

if ~exist(params.cacheNN, 'file')
    nn = nn_setup(params, w, opts);
    nn = nn_train(params, nn, opts);
    save(params.cacheNN, 'nn', '-v7.3');
else
    load(params.cacheNN);
end

end

