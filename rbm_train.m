function [ rbm ] = rbm_train( params, rbm, opts )
%RBM_TRAIN Summary of this function goes here
%   Detailed explanation goes here
data_per_img = prod(params.datasize_per_img);
m = params.trainImgNum * data_per_img;
numbatches = m / opts.batchsize;
numbatches = floor(numbatches);

gpu_num = gpuDeviceCount();
n_push = 10;
n_fetch = 50;
n_next = 100;

g_premu = gpuArray(rbm.premu);
g_presigma = gpuArray(rbm.presigma);
g_Ureduce = gpuArray(rbm.Ureduce);
g_postmu = gpuArray(rbm.postmu);
g_postsigma = gpuArray(rbm.postsigma);

if ~params.restart && exist(params.cacheEpochRBM, 'file')
    load(params.cacheEpochRBM);
    disp(['Finish Load Cache of Epoch-', num2str(epoch_cache)]);
    epoch_start = epoch_cache+1;
else
    epoch_start = 1;
end

W = cell(1, gpu_num+1);
accrued_vW = cell(1, gpu_num);
vW = cell(1, gpu_num);
c = cell(1, gpu_num+1);
accrued_vc = cell(1, gpu_num);
vc = cell(1, gpu_num);
b = cell(1, gpu_num+1);
accrued_vb = cell(1, gpu_num);
vb = cell(1, gpu_num);
for i=1:gpu_num+1
    W{i} = rbm.W;
    b{i} = rbm.b;
    c{i} = rbm.c;
    if i~=gpu_num+1
        vW{i} = rbm.vW;
        vb{i} = rbm.vb;
        vc{i} = rbm.vc;
        accrued_vW{i} = zeros(size(rbm.W), 'single');
        accrued_vc{i} = zeros(size(rbm.c), 'single');
        accrued_vb{i} = zeros(size(rbm.b), 'single');
    end
end
% g_W = gpuArray(rbm.W);
% g_vW = gpuArray(rbm.vW);
% g_c = gpuArray(rbm.c);
% % g_vc = gpuArray(rbm.vc);
% g_b = gpuArray(rbm.b);
% g_vb = gpuArray(single(rbm.vb));

g_L2 = gpuArray(single(opts.L2));
g_batchsize = gpuArray(single(opts.batchsize));
g_momentum = gpuArray(single(rbm.momentum));
g_alpha = gpuArray(single(rbm.alpha));




for i = epoch_start : opts.numepochs
    gpu_no = 1;
    step = 1;
    [params.imgIdx, params.imgDataIdx] = randIdx(params);
    t1 = tic;
%         kk = randperm(m);
    err = gpuArray.zeros(numbatches,1, 'single');
    %% cache X from two img and transfer it to GPU.
    currentIdx = 0;
    currentPartIdx = 0;
    g_cacheX = [];
    if params.debug
        small_batch_debug_size = 1000;
    else
        small_batch_debug_size = 10000;
    end
    for l = 1 : 2*numbatches
        if mod(l,small_batch_debug_size) == 1
            batch_err = gpuArray.zeros(small_batch_debug_size,1,'single');
            tic;
        end
       %% Extract Raw Batch X.
        if currentPartIdx>=params.trainImgNum && size(g_cacheX,1)<=currentIdx+opts.batchsize-1
            break;
        else
            [currentPartIdx, g_cacheX, batch, currentIdx] = getNextBatchX(g_cacheX, currentPartIdx, params, opts, currentIdx);
        end
       
        %% Calculate the RBM gradient
        if mod(step-1, n_fetch)==0
            W{gpu_no} = W{gpu_num+1};
%             vW{gpu_no} = vW{gpu_num+1};
            b{gpu_no} = b{gpu_num+1};
%             vb{gpu_no} = vb{gpu_num+1};
            c{gpu_no} = c{gpu_num+1};
            g_W = gpuArray(W{gpu_no});
            g_vW = gpuArray(vW{gpu_no});
            g_b = gpuArray(b{gpu_no});
            g_vb = gpuArray(vb{gpu_no});
            g_c = gpuArray(c{gpu_no});
            g_vc = gpuArray(vc{gpu_no});
        end
        [t_vW, t_vb, t_vc, err_temp] = calRBMGradient(batch, g_premu, g_presigma, g_Ureduce, g_postmu, g_postsigma, g_W, g_c, g_b, g_L2, g_batchsize, g_alpha);
        g_vW = g_momentum*g_vW + t_vW;
        g_vb = g_momentum*g_vb + t_vb;
        g_vc = g_momentum*g_vc + t_vc;
        clear t_vW t_vb t_vc;
        g_W = g_W + g_vW;
        accrued_vW{gpu_no} = accrued_vW{gpu_no} + gather(g_vW);
        g_b = g_b + g_vb;
        accrued_vb{gpu_no} = accrued_vb{gpu_no} + gather(g_vb);
        g_c = g_c + g_vc;
        temp = g_c+4;
        temp(temp>0) = 0;
        accrued_vc{gpu_no} = accrued_vc{gpu_no} + gather(g_vc) - temp;
        % Upper bound for c
        g_c(g_c>-4) = -4;
        batch_err(mod(l,small_batch_debug_size)+1) = err_temp;
        err(l) = err_temp;

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
            batch_err(batch_err==0)=[];
            disp(['Epoch ', num2str(i), '- mini-batch: ', num2str(l) '/' num2str(numbatches) '.Average reconstruction error: ' num2str(gather(mean(batch_err)))]);
        end
        if mod(step-1, n_push)==0
            W{gpu_no} = gather(g_W);
            W{gpu_num+1} = W{gpu_num+1} + accrued_vW{gpu_no};
            accrued_vW{gpu_no} = zeros(size(rbm.W), 'single');
            b{gpu_no} = gather(g_b);
            b{gpu_num+1} = b{gpu_num+1} + accrued_vb{gpu_no};
            accrued_vb{gpu_no} = zeros(size(rbm.b), 'single');
            c{gpu_no} = gather(g_c);
            c{gpu_num+1} = c{gpu_num+1} + accrued_vc{gpu_no};
            accrued_vc{gpu_no} = zeros(size(rbm.c), 'single');
            % Upper Bound
            c{gpu_num+1}(vc{gpu_num+1}>-4) = -4;
%             vW{gpu_no} = zeros(size(vW{gpu_no}), 'single');
%             vb{gpu_no} = zeros(size(vb{gpu_no}), 'single');
%             g_vW = gpuArray.zeros(size(g_vW), 'single');
%             g_vb = gpuArray.zeros(size(g_vb), 'single');
        end
        
        if mod(step-1, n_next)==0
            gpu_no = mod(gpu_no+1, gpu_num) + 1;
        end   
        step = step + 1;


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
    err(err==0) = [];
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
        for i=1:params.cacheImageNum
            if partIdx>=params.trainImgNum
                break;
            end
            partIdx = partIdx+1;
            xyimg = params.trainXYimg(params.imgIdx(partIdx),:);
            rand_angle = rand * 360;
            xyimg{1} = imrotate(xyimg{1}, rand_angle);
            xyimg{2} = imrotate(xyimg{2}, rand_angle);
            [nextimgX, ~] = xyimgIdx2data(params.WindowSize, params.StrideSize,...
                    xyimg);
            uselessDataIdx = findUselessData(nextimgX, params.WindowSize, params.StrideSize);
            nextimgX(uselessDataIdx,:) = [];
            nextimgX = nextimgX( randperm(size(nextimgX,1)), : );
            nextimgX = gpuArray(nextimgX);
            cacheX = [cacheX; nextimgX];
            clear nextimgX;
        end
    end
    batchx = cacheX(Idx:Idx+opts.batchsize-1,:);
    Idx = Idx + opts.batchsize;
%     cacheX(1:opts.batchsize,:) = [];
end

function [idx] = findUselessData(data, WindowSize, StrideSize)
    blank = (WindowSize - StrideSize)/2;
    assert(rem(blank,1)==0, 'The blank should be integer, which actual is %f'...
    , blank);
%     rowIdx = blank+1:blank+StrideSize;
%     colIdx = blank+1:blank+StrideSize;
%     newrowIdx = repmat(rowIdx, [1 StrideSize]);
%     newColIdx = zeros(1, StrideSize^2);
%     newColIdx(1,:) = colIdx(1,ceil((1:StrideSize^2)/StrideSize));
%     subIdx = zeros(1, 3*StrideSize^2);
%     matSize = [WindowSize WindowSize 3];
%     channelIdx = ones(1, StrideSize^2);
%     subIdx(1,1:StrideSize^2) = sub2ind(matSize, newrowIdx, newColIdx, 1*channelIdx);
%     subIdx(1, StrideSize^2+1:2*StrideSize^2) = sub2ind(matSize, newrowIdx, newColIdx, 2*channelIdx);
%     subIdx(1, 2*StrideSize^2+1:3*StrideSize^2) = sub2ind(matSize, newrowIdx, newColIdx, 3*channelIdx);
%     idx = ~any(data(:,subIdx),2);
    idx = ~all(data, 2);
end

