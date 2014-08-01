function [ nn ] = nn_train( params, nn, opts )
%NN_TRAIN Summary of this function goes here
%   Detailed explanation goes here
g_batchsize = opts.batchsize;
g_batchsize = gpuArray(single(g_batchsize));
numepochs = opts.numepochs;
data_per_img = prod(params.datasize_per_img);
m = params.trainImgNum * data_per_img;
numbatches = m / opts.batchsize;
numbatches = floor(numbatches);

g_alpha = gpuArray(single(nn.alpha));
g_momentum = gpuArray(single(nn.momentum));
g_L2 = gpuArray(single(nn.L2));


g_premu = gpuArray(params.premu);
g_presigma = gpuArray(params.presigma);
g_Ureduce = gpuArray(params.Ureduce);
g_postmu = gpuArray(params.postmu);
g_postsigma = gpuArray(params.postsigma);

if ~params.restart && exist(params.cacheEpochNN, 'file')
    load(params.cacheEpochNN);
    disp(['Finish Load Cache of Epoch-', num2str(epoch_cache)]);
    epoch_start = epoch_cache+1;
else
    epoch_start = 1;
end

g_Theta1 = gpuArray(nn.Theta1);
g_Theta2 = gpuArray(nn.Theta2);
g_vTheta1 = gpuArray(nn.vTheta1);
g_vTheta2 = gpuArray(nn.vTheta2);

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
            [currentPartIdx, g_cacheX, g_cacheY, batchX, batchY, currentIdx] = getNextBatchX(g_cacheX, g_cacheY, currentPartIdx, params, opts, currentIdx);
        end
        
        %% Calculate the gradient.
        [t_vTheta1, t_vTheta2, t_error] = calNNGradient( batchX, batchY, g_premu, g_presigma, g_Ureduce, g_postmu, g_postsigma, g_Theta1, g_Theta2 );

        %% Update the Theta.
        
        
        err(l,1) = t_error;

        g_vTheta1 = g_momentum*g_vTheta1 + g_alpha*(t_vTheta1/g_batchsize + g_L2*[gpuArray.zeros(size(g_Theta1,1),1,'single') g_Theta1(:,2:end)]);
        g_Theta1 = g_Theta1 - g_vTheta1;
        g_vTheta2 = g_momentum*g_vTheta2 + g_alpha*(t_vTheta2/g_batchsize + g_L2*[gpuArray.zeros(size(g_Theta2,1),1,'single') g_Theta2(:,2:end)]);
        g_Theta2 = g_Theta2 - g_vTheta2;
        
    end
    t = toc;
    err(err==0) = [];
    str_perf = sprintf('; Full-batch train err = %f', gather(sum(err)));
    disp(['epoch ' num2str(i) '/' num2str(numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(gather(mean(err))) str_perf]);
    
    nn.Theta1 = gather(g_Theta1);
    nn.Theta2 = gather(g_Theta2);
    nn.vTheta1 = gather(g_vTheta1);
    nn.vTheta2 = gather(g_vTheta2);
    epoch_cache = i;
    save(params.cacheEpochNN, 'nn', 'epoch_cache', '-v7.3');
    disp(['cache result of epoch-', num2str(i)]);
end


end


function [partIdx, cacheX, cacheY, batchX, batchY, Idx] = getNextBatchX(cacheX, cacheY, partIdx, params, opts, Idx)
    if size(cacheX,1)<=Idx+opts.batchsize-1
        cacheX(1:Idx-1,:) = [];
        cacheY(1:Idx-1,:) = [];
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
            [nextimgX, nextimgY] = xyimgIdx2data(params.WindowSize, params.StrideSize,...
                    xyimg);
            uselessDataIdx = findUselessData(nextimgX, params.WindowSize, params.StrideSize);
            nextimgX(uselessDataIdx,:) = [];
            rIdx = randperm(size(nextimgX,1));
            nextimgX = nextimgX( rIdx, : );
            nextimgX = gpuArray(nextimgX);
            cacheX = [cacheX; nextimgX];
            nextimgY(uselessDataIdx,:) = [];
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
    idx = ~all(data ,2);
end
