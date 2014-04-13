function [ nn ] = Mass_Roads( input_args )
%MASS_ROADS Summary of this function goes here
%   Detailed explanation goes here
params = initParams();
hidden_layer_size = 4096*3;
opts.numepochs =   10;
opts.batchsize = 64;
opts.momentum  =   0.9;
opts.alpha     =   0.001;
opts.L2 = 0.0002;
if ~params.restart && exist(params.cacheRBM, 'file')
    load(params.cacheRBM);
else
    rbm = rbmsetup(params, hidden_layer_size, opts);
    rbm = rbm_train(params, rbm, opts);
    save(params.cacheRBM, 'rbm', '-v7.3');
end
figure; visualize(rbm.W);   %  Visualize the RBM weights

w = [rbm.c; rbm.W];
clear rbm;


nn.alpha = 0.0005;
if ~params.restart && exist(params.cacheNN, 'file')
    load(params.cacheNN);
else
    nn = nn_setup(params, w, opts);
    nn = nn_train(params, nn, opts);
    save(params.cacheNN, 'nn', '-v7.3');
end

test_img_num = size(params.testXYimg,2);
predtesty = zeros(test_img_num*params.data_per_img, 256);
for i=1:size(params.testXYimg,2)
    [x, ~] = xyimgIdx2data(params, params.testXYimg, i);
    x = [ ones(g_batchsize,1) x ];
    Z2 = nn.Theta1 * x';

    A2 = sigm( Z2 );
    A2 = [ ones(1,size(A2,2)) ; A2 ];
%     Z2 = [ ones(1,size(Z2,2)) ; Z2 ];
    idx = (i-1)*params.data_per_img:i*params.data_per_img;
    predtesty(idx,:) = sigm( g_Theta2 * A2 );
end
save(params.cacheTestY, 'predtesty', '-v7.3');

end

