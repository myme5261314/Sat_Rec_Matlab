function [ postnn ] = post_nn_train( params, predtrainyimgcell, postnn, opts, postnnmu, postnnsigma )
%POST_NN_TRAIN Summary of this function goes here
%   Detailed explanation goes here

g_batchsize = opts.batchsize;
g_batchsize = gpuArray(single(g_batchsize));
numepochs = opts.numepochs;
data_per_img = prod(params.datasize_per_img);
m = params.trainImgNum * data_per_img;
numbatches = m / opts.batchsize;
numbatches = floor(numbatches);

g_alpha = gpuArray(single(postnn.alpha));
g_momentum = gpuArray(single(postnn.momentum));
g_L2 = gpuArray(single(postnn.L2));


g_mu = gpuArray(postnnmu);
g_sigma = gpuArray(postnnsigma);

if ~params.restart && exist(params.cacheEpochPostNN, 'file')
    load(params.cacheEpochPostNN);
    disp(['Finish Load Cache of Epoch-', num2str(epoch_cache)]);
    epoch_start = epoch_cache+1;
else
    epoch_start = 1;
end

g_Theta1 = gpuArray(postnn.Theta1);
g_Theta2 = gpuArray(postnn.Theta2);
g_vTheta1 = gpuArray(postnn.vTheta1);
g_vTheta2 = gpuArray(postnn.vTheta2);

% gpu = gpuDevice();

for i = epoch_start : numepochs
    tic;
    err = gpuArray.zeros(2*numbatches, 1);
    %% cache X from one img and transfer it to GPU.
    currentIdx = 0;
    currentPartIdx = 0;
    g_cacheX = [];
    g_cacheY = [];
    for l = 1 : 2*numbatches
%         tic;
       %% Extract Raw Batch X,Y.
        if currentPartIdx>=params.trainImgNum && size(g_cacheX,1)<=currentIdx+opts.batchsize-1
            break;
        else
            [currentPartIdx, g_cacheX, g_cacheY, batchX, batchY, currentIdx] = getNextBatchX(g_cacheX, g_cacheY, currentPartIdx, params, predtrainyimgcell, opts, currentIdx);
        end
        
       %% Preprocess the Raw Batch X.
        batchX = gpuArray(single(batchX));
        batchY = gpuArray(single(batchY));
        batchX = bsxfun(@rdivide, bsxfun(@minus, batchX, g_mu), g_sigma);

        batchY = single(batchY==255);
       %% Perform forward propogation.
        batchX = [ gpuArray.ones(g_batchsize,1, 'single') batchX ];
        Z2 = g_Theta1 * batchX';

        A2 = sigm( Z2 );
        A2 = [ gpuArray.ones(1,size(A2,2), 'single') ; A2 ];
        Z2 = [ gpuArray.ones(1,size(Z2,2), 'single') ; Z2 ];
        hx = sigm( g_Theta2 * A2 );

       %% Perform Back Propgation.
       	Y_trans = batchY;
        delta3 = hx - Y_trans';
        clear  Y_trans batchY;
        clear hx;
        delta2 = g_Theta2'*delta3 .* sigmoidGradient(Z2);
        clear Z2;

        delta2 = delta2(2:end,:);
        %% Update the Theta.
        
        
        err(l,1) = sum(sum(delta3.^2));
        g_Theta1_grad = delta2*batchX;
        g_Theta2_grad = delta3*A2';
        g_vTheta1 = g_momentum*g_vTheta1 + g_alpha*(g_Theta1_grad/g_batchsize + g_L2*[gpuArray.zeros(size(g_Theta1,1),1,'single') g_Theta1(:,2:end)]);
        g_Theta1 = g_Theta1 - g_vTheta1;
        g_vTheta2 = g_momentum*g_vTheta2 + g_alpha*(g_Theta2_grad/g_batchsize + g_L2*[gpuArray.zeros(size(g_Theta2,1),1,'single') g_Theta2(:,2:end)]);
        g_Theta2 = g_Theta2 - g_vTheta2;

    end
    t = toc;
    err(err==0) = [];
    str_perf = sprintf('; Full-batch train err = %f', gather(sum(err)));
    disp(['epoch ' num2str(i) '/' num2str(numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(gather(mean(err))) str_perf]);
    
    postnn.Theta1 = gather(g_Theta1);
    postnn.Theta2 = gather(g_Theta2);
    postnn.vTheta1 = gather(g_vTheta1);
    postnn.vTheta2 = gather(g_vTheta2);
    epoch_cache = i;
    save(params.cacheEpochPostNN, 'postnn', 'epoch_cache', '-v7.3');
    disp(['cache result of epoch-', num2str(i)]);
end


end


function [partIdx, cacheX, cacheY, batchX, batchY, Idx] = getNextBatchX(cacheX, cacheY, partIdx, params, predtrainyimgcell, opts, Idx)
    if size(cacheX,1)<=Idx+opts.batchsize-1
        cacheX(1:Idx-1,:) = [];
        cacheY(1:Idx-1,:) = [];
        Idx = 1;
        for i=1:params.cacheImageNum
            if partIdx>=params.trainImgNum
                break;
            end
            partIdx = partIdx+1;
            [nextimgX, nextimgY] = predyimg2data(predtrainyimgcell{partIdx,1},...
                params.trainXYimg{partIdx,2}, params.WindowSize, params.StrideSize);
            rIdx = randperm(size(nextimgX,1));
            nextimgX = nextimgX( rIdx, : );
            nextimgX = gpuArray(nextimgX);
            cacheX = [cacheX; nextimgX];
            nextimgY = nextimgY( rIdx, : );
            nextimgY = gpuArray(nextimgY);
            cacheY = [cacheY; nextimgY];
        end
    end
    batchX = cacheX(Idx:Idx+opts.batchsize-1,:);
    batchY = cacheY(Idx:Idx+opts.batchsize-1,:);
    Idx = Idx + opts.batchsize;
%     cacheX(1:opts.batchsize,:) = [];
%     cacheY(1:opts.batchsize,:) = [];
end
