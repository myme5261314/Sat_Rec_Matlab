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

g_W = gpuArray(rbm.W);
g_vW = gpuArray(rbm.vW);
g_c = gpuArray(rbm.c);
g_vc = gpuArray(rbm.vc);
g_b = gpuArray(rbm.b);
g_vb = gpuArray(single(rbm.vb));

g_L2 = gpuArray(single(opts.L2));
g_batchsize = gpuArray(single(opts.batchsize));
g_momentum = gpuArray(single(rbm.momentum));
g_alpha = gpuArray(single(rbm.alpha));

% err = gpuArray(0);


for i = 1 : opts.numepochs
    tic;
%         kk = randperm(m);
    err = gpuArray(0);
    %% cache X from one img and transfer it to GPU.
    currentPartIdx = 1;
    [g_cacheX, ~] = xyimgIdx2data(params, params.trainXYimg,...
        params.imgIdx(currentPartIdx), params.imgDataIdx(currentPartIdx,:));
    g_cacheX = gpuArray(g_cacheX);
    for l = 1 : numbatches
%         tic;
        dataidx = [(l-1)*opts.batchsize+1 l*opts.batchsize];
        data_imgidx = ceil(dataidx/data_per_img);
        data_dataidx = mod(dataidx-1, data_per_img)+1;
       %% Extract Raw Batch X.
        if all(data_imgidx == currentPartIdx)
            batch = g_cacheX(data_dataidx(1):data_dataidx(2),:);
        else
            currentPartIdx = data_imgidx(2);
            temp = g_cacheX(data_dataidx(1):end,:);
            % Update the cache to the next img data.
            clear g_cacheX;
            [g_cacheX, ~] = xyimgIdx2data(params, params.trainXYimg,...
                params.imgIdx(currentPartIdx),...
                params.imgDataIdx(currentPartIdx,:));
            
            g_cacheX = gpuArray(g_cacheX);
            batch = [temp; g_cacheX(1:data_dataidx(2),:)];
        end
        
        batch = single(batch);
        batch = gpuArray(batch);
       %% Preprocess the Raw Batch X.
        batch = bsxfun(@rdivide, bsxfun(@minus, batch, g_premu), g_presigma);
        batch = batch * g_Ureduce;
        batch = bsxfun(@rdivide, bsxfun(@minus, batch, g_postmu), g_postsigma);

        g_v1 = batch;
        g_h1 = sigmrnd(bsxfun(@plus, g_c, g_v1*g_W));
%             g_v2 = sigmrnd(bsxfun(@plus, g_b', g_h1*g_W));
%             g_v2 = gpuArray(zeros(size(g_v1),'single'));
        g_v2 = normrnd( bsxfun(@plus, g_b, g_h1*g_W'), 1 );

        g_h2 = sigm(bsxfun(@plus, g_c, g_v2*g_W));
        
        g_c1 = g_v1' * g_h1;
        g_c2 = g_v2' * g_h2;

        g_vb = g_momentum * g_vb + g_alpha * sum(g_v1 - g_v2)/g_batchsize;
        err = err + sum(sum((g_v1 - g_v2) .^ 2)) / g_batchsize;
%         g_temp = (g_c1-g_c2)/g_batchsize;
%         g_temp = g_temp - g_L2*g_W;
%         g_temp = g_alpha*g_temp;
%         tic;
%         g_temp = g_momentum*g_vW + g_temp;
%         toc;
%         g_vW = g_temp;
        clear g_v1 g_v2 g_h1 g_h2;
%         tic;
        g_c1 = g_c1 - g_c2;
        clear g_c2;
        g_c1 = g_c1/g_batchsize;
        g_c1 = g_c1 - g_L2*g_W;
        g_c1 = g_c1*g_alpha;
        g_vW = g_momentum*g_vW + g_c1;
        clear g_c1;
%         g_vW = g_momentum * g_vW + g_alpha * ( (g_c1-g_c2)/g_batchsize - g_L2*g_W );
%         toc;
%         g_vb = g_momentum * g_vb + g_alpha * sum(g_v1 - g_v2)/g_batchsize;
%             g_vc = rbm.momentum * g_vc + rbm.alpha * sum(g_h1 - g_h2)' / opts.batchsize;

        g_W = g_W + g_vW;
        g_b = g_b + g_vb;
%             g_c = g_c + g_vc;

%         err = err + sum(sum((g_v1 - g_v2) .^ 2)) / g_batchsize;
% %             err = err + gather(sum(sum((g_v1 - g_v2) .^ 2))) / opts.batchsize;
%         toc;

    end
    toc;
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(gather(err) / numbatches)]);


rbm.W = gather(g_W);
rbm.vW = gather(g_vW);
rbm.c = gather(g_c);
rbm.vc = gather(g_vc);
rbm.b = gather(g_b);
rbm.vb = gather(g_vb);

end

end


