function [ rbm ] = rbm_train( params, rbm, opts )
%RBM_TRAIN Summary of this function goes here
%   Detailed explanation goes here
data_per_img = prod(params.datasize_per_img);
m = params.trainImgNum * data_per_img;
numbatches = m / opts.batchsize;
numbatches = floor(numbatches);

g_premu = gpuArray(rbm.premu);
g_presigma = gpuArray(rbm.presigma);
g_Ureduce = gpuArray(rbm.Ureduce);
g_postmu = gpuArray(rbm.postmu);
g_postsigma = gpuArray(rbm.postsigma);

if ~params.restart && exist(params.cacheEpochRBM, 'file')
    load(params.cacheEpochRBM);
    disp(['Finish Load Cach of Epoch-', num2str(epoch_cache)]);
    epoch_start = epoch_cache+1;
else
    epoch_start = 1;
end

g_W = gpuArray(rbm.W);
g_vW = gpuArray(rbm.vW);
g_c = gpuArray(rbm.c);
% g_vc = gpuArray(rbm.vc);
g_b = gpuArray(rbm.b);
g_vb = gpuArray(single(rbm.vb));

g_L2 = gpuArray(single(opts.L2));
g_batchsize = gpuArray(single(opts.batchsize));
g_momentum = gpuArray(single(rbm.momentum));
g_alpha = gpuArray(single(rbm.alpha));

% err = gpuArray(0);
checkNaN_Inf_flag=1;
checkNaN_Inf = @(mat) any(any(isinf(mat))) || any(any(isnan(mat)));
checkThresholdF = @(mat) all(all(abs(mat)>1e30));

for i = 1 : opts.numepochs
    [params.imgIdx, params.imgDataIdx] = randIdx(params);
    t1 = tic;
%         kk = randperm(m);
    err = gpuArray.zeros(numbatches,1, 'single');
    %% cache X from two img and transfer it to GPU.
    currentIdx = 1;
    currentPartIdx = 1;
    [g_cacheX, ~] = xyimgIdx2data(params, params.trainXYimg,...
        params.imgIdx(currentPartIdx), params.imgDataIdx(currentPartIdx,:));
    currentPartIdx = currentPartIdx + 1;
    [temp, ~] = xyimgIdx2data(params, params.trainXYimg,...
        params.imgIdx(currentPartIdx), params.imgDataIdx(currentPartIdx,:));
    g_cacheX = [g_cacheX; temp];
    g_cacheX = gpuArray(g_cacheX);
    clear temp;
    small_batch_debug_size = 10000;
    for l = 1 : numbatches
        if mod(l,small_batch_debug_size) == 1
            batch_err = gpuArray.zeros(small_batch_debug_size,1,'single');
            tic;
        end
       %% Extract Raw Batch X.
       
        [currentPartIdx, g_cacheX, batch, currentIdx] = getNextBatchX(g_cacheX, currentPartIdx, params, opts, currentIdx);
       
%         batch = gpuArray.rand(g_batchsize, 12288);
        batch = single(batch);
        batch = gpuArray(batch);
       %% Preprocess the Raw Batch X.
        batch = bsxfun(@rdivide, bsxfun(@minus, batch, g_premu), g_presigma);
        batch = batch * g_Ureduce;
        batch = bsxfun(@rdivide, bsxfun(@minus, batch, g_postmu), g_postsigma);

        g_v1 = batch;
%         if checkNaN_Inf_flag && checkNaN_Inf(g_v1) || checkThresholdF(g_v1)
%             disp('g_v1 failed!');
%         end
        g_h1 = sigm(bsxfun(@plus, g_c, g_v1*g_W));
        g_c1 = g_v1' * g_h1;
        g_h1 = single(g_h1 > gpuArray.rand(size(g_h1), 'single'));
%             g_v2 = sigmrnd(bsxfun(@plus, g_b', g_h1*g_W));
%             g_v2 = gpuArray(zeros(size(g_v1),'single'));
%         if checkNaN_Inf_flag && checkNaN_Inf(g_h1) || checkThresholdF(g_h1)
%             disp('g_h1 failed!');
%         end
        g_v2 = bsxfun(@plus, g_b, g_h1*g_W') + gpuArray.randn(size(g_v1));
        clear g_h1;
%         if checkNaN_Inf_flag && checkNaN_Inf(g_v2) || checkThresholdF(g_v2)
%             disp('g_v2 failed!');
%         end
        g_h2 = sigm(bsxfun(@plus, g_c, g_v2*g_W));
%         if checkNaN_Inf_flag && checkNaN_Inf(g_h2) || checkThresholdF(g_h2)
%             disp('g_h2 failed!');
%         end
        
%         if checkNaN_Inf_flag && checkNaN_Inf(g_v1) || checkThresholdF(g_c1)
%             disp('g_c1 failed!');
%         end
%         g_c2 = g_v2' * g_h2;
%         clear g_h2;
%         g_c1 = bsxfun(@minus, g_c1, g_c2);
        g_c1 = g_c1 - g_v2' * g_h2;
%         clear g_c2;
%         if checkNaN_Inf_flag && checkNaN_Inf(g_c2) || checkThresholdF(g_c2)
%             disp('g_c2 failed!');
%         end
        
        err_temp = sum(mean((g_v1-g_v2).^2));
%         if checkNaN_Inf_flag && checkNaN_Inf(err_temp)
%             disp('err_temp failed!');
%         end
%         if isnan(err_temp)
%             disp(['mini-batch', num2str(l) '/' num2str(numbatches) '.Average reconstruction error: ' num2str(gather(err_temp))]);
%         end
            

%         g_vW = bsxfun(@plus, bsxfun(@times, g_vW, g_momentum), bsxfun(@times,...
%                 bsxfun(@minus, bsxfun(@rdivide, g_c1, g_batchsize),...
%                 bsxfun(@times, g_W, g_L2)), g_alpha));
%         g_vW = bsxfun(@times, g_vW, g_momentum);
%         g_c1 = bsxfun(@rdivide, g_c1, g_batchsize);
%         g_c1 = bsxfun(@minus, g_c1, bsxfun(@times, g_W, g_L2));
%         g_c1 = bsxfun(@times, g_c1, g_alpha);
%         g_vW = bsxfun(@plus, g_vW, g_c1);
%         g_vW = g_vW * g_momentum;
%         g_c1 = g_c1/g_batchsize;
%         g_c1 = g_c1 - g_W * g_L2;
%         g_c1 = g_alpha * g_c1;
%         g_vW = g_vW + g_c1;
%         clear g_c1;
        g_vW = g_momentum * g_vW;
        g_c1 = g_c1/g_batchsize;
        g_vW = g_vW + g_alpha * ( g_c1 - g_L2*g_W );
%         clear g_c1;
%         if checkNaN_Inf_flag && checkNaN_Inf(g_vW) || checkThresholdF(g_vW)
%             disp('g_vW failed!');
%         end
%         toc;
        g_vb = g_momentum * g_vb + g_alpha * mean(g_v1 - g_v2);
%         if checkNaN_Inf_flag && checkNaN_Inf(g_vb) || checkThresholdF(g_vb)
%             disp('g_vb failed!');
%         end
%             g_vc = rbm.momentum * g_vc + rbm.alpha * sum(g_h1 - g_h2)' / opts.batchsize;

        g_W = g_W + g_vW;
%         if checkNaN_Inf_flag && checkNaN_Inf(g_W) || checkThresholdF(g_W)
%             disp('g_W failed!');
%         end
        g_b = g_b + g_vb;
%         if checkNaN_Inf_flag && checkNaN_Inf(g_b) || checkThresholdF(g_b)
%             disp('g_b failed!');
%         end
%             g_c = g_c + g_vc;
        batch_err(mod(l,small_batch_debug_size)+1) = err_temp;
        err(l) = err_temp;
%         if checkNaN_Inf(err)
%             disp('err failed!');
%         end
        if mod(l,small_batch_debug_size) == 0
            toc;
            nan_num = nnz(isnan(batch_err)|isinf(batch_err));
            if nan_num~=0
                disp(['NaN or Inf occur times: ', num2str(nan_num), '/', num2str(small_batch_debug_size)]);
                if nan_num==numel(batch_err)
                    disp('Need break');
                end
                batch_err(isnan(batch_err)|isnan(batch_err)) = [];
            end
            disp(['Epoch ', num2str(i), '- mini-batch: ', num2str(l) '/' num2str(numbatches) '.Average reconstruction error: ' num2str(gather(mean(batch_err)))]);
        end
%         err = err + sum(sum((g_v1 - g_v2) .^ 2)) / g_batchsize;
% %             err = err + gather(sum(sum((g_v1 - g_v2) .^ 2))) / opts.batchsize;
%         toc;


    end
    toc(t1);
    nan_num = nnz(isnan(err)|isnan(err));
    if nan_num~=0
        disp(['NaN or Inf occur times: ', num2str(nan_num), '/', num2str(numbatches)]);
        if nan_num==numel(err)
            disp('Need break');
        end
        err(isnan(err)|isnan(err)) = [];
    end
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(gather(mean(err)))]);


    rbm.W = gather(g_W);
    rbm.vW = gather(g_vW);
%     rbm.c = gather(g_c);
%     rbm.vc = gather(g_vc);
    rbm.b = gather(g_b);
    rbm.vb = gather(g_vb);

    epoch_cache = i;
    save(params.cacheEpochRBM, 'rbm', 'epoch_cache', '-v7.3');
    disp(['cache result of epoch-', num2str(i)]);
    end

end

function [partIdx, cacheX, batchx, Idx] = getNextBatchX(cacheX, partIdx, params, opts, Idx)
    if size(cacheX,1)<=Idx+opts.batchsize-1
        cacheX(1:Idx-1,:) = [];
        Idx = 1;
        for i=1:param.cacheImageNum
            if partIdx>=numel(params.trainXfile)
                break;
            end
            partIdx = partIdx+1;
            [nextimgX, ~] = xyimgIdx2data(params, params.trainXYimg,...
                    params.imgIdx(partIdx),...
                    params.imgDataIdx(partIdx,:));
            nextimgX = gpuArray(nextimgX);
            cacheX = [cacheX; nextimgX];
        end
    end
    batchx = cacheX(Idx:Idx+opts.batchsize-1,:);
    Idx = Idx + opts.batchsize;
%     cacheX(1:opts.batchsize,:) = [];
end

