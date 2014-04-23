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

if ~params.restart && exist(params.cacheTestY, 'file')
    load(params.cacheTestY);
else
    test_img_num = size(params.testXYimg,1);
    predtesty = zeros(test_img_num*params.data_per_img, 256);
    for i=1:test_img_num
        [x, ~] = xyimgIdx2data(params, params.testXYimg, i);
        x = single(x);
        x = bsxfun(@rdivide, bsxfun(@minus, x, params.premu), params.presigma);
        x = x * params.Ureduce;
        x = bsxfun(@rdivide, bsxfun(@minus, x, params.postmu), params.postsigma);
        x = [ ones(size(x,1),1) x ];
        Z2 = nn.Theta1 * x';

        A2 = sigm( Z2 );
        A2 = [ ones(1,size(A2,2)) ; A2 ];
    %     Z2 = [ ones(1,size(Z2,2)) ; Z2 ];
        idx = ((i-1)*params.data_per_img+1):i*params.data_per_img;
        predtesty(idx,:) = sigm( nn.Theta2 * A2 )';
    end
    save(params.cacheTestY, 'predtesty', '-v7.3');
    
end

[ predyimgcell, thresholdlist ] = predy2img( params, predtesty );
thresholdlist_new = (0:1e-2:1)';

[precision, recall] = cal_precision_recall(params, predyimgcell, params.testXYimg(:,2), thresholdlist_new);

figure;
plot(recall, precision);

end