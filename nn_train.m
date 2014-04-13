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

g_Theta1 = gpuArray(nn.Theta1);
g_Theta2 = gpuArray(nn.Theta2);
g_vTheta1 = gpuArray(nn.vTheta1);
g_vTheta2 = gpuArray(nn.vTheta2);

% gpu = gpuDevice();

for i=1:numepochs
    tic;
    err = gpuArray.zeros(numbatches, 1);
    %% cache X from one img and transfer it to GPU.
    currentPartIdx = 1;
    [g_cacheX, g_cacheY] = xyimgIdx2data(params, params.trainXYimg,...
        params.imgIdx(currentPartIdx), params.imgDataIdx(currentPartIdx,:));
    g_cacheX = gpuArray(g_cacheX);
    g_cacheY = gpuArray(single(g_cacheY/255));
    for l = 1 : numbatches
%         tic;
        dataidx = [(l-1)*opts.batchsize+1 l*opts.batchsize];
        data_imgidx = ceil(dataidx/data_per_img);
        data_dataidx = mod(dataidx-1, data_per_img)+1;
       %% Extract Raw Batch X,Y.
        if all(data_imgidx == currentPartIdx)
            batchX = g_cacheX(data_dataidx(1):data_dataidx(2),:);
            batchY = g_cacheY(data_dataidx(1):data_dataidx(2),:);
        else
            currentPartIdx = data_imgidx(2);
            tempX = g_cacheX(data_dataidx(1):end,:);
            tempY = g_cacheY(data_dataidx(1):end,:);
            % Update the cache to the next img data.
            clear g_cacheX;
            [g_cacheX, g_cacheY] = xyimgIdx2data(params, params.trainXYimg,...
                params.imgIdx(currentPartIdx),...
                params.imgDataIdx(currentPartIdx,:));
            
            g_cacheX = gpuArray(g_cacheX);
            batchX = [tempX; g_cacheX(1:data_dataidx(2),:)];
            clear tempX;
            g_cacheY = gpuArray(single(g_cacheY/255));
            batchY = [tempY; g_cacheY(1:data_dataidx(2),:)];
            clear tempY;
        end
       %% Preprocess the Raw Batch X.
        batchX = gpuArray(single(batchX));
        batchY = gpuArray(single(batchY));
        batchX = bsxfun(@rdivide, bsxfun(@minus, batchX, g_premu), g_presigma);
        batchX = batchX * g_Ureduce;
        batchX = bsxfun(@rdivide, bsxfun(@minus, batchX, g_postmu), g_postsigma);
       %% Perform forward propogation.
        batchX = [ gpuArray.ones(g_batchsize,1, 'single') batchX ];
        Z2 = g_Theta1 * batchX';

        A2 = sigm( Z2 );
        A2 = [ gpuArray.ones(1,size(A2,2), 'single') ; A2 ];
        Z2 = [ gpuArray.ones(1,size(Z2,2), 'single') ; Z2 ];
        hx = sigm( g_Theta2 * A2 );

%         J = sum(sum( Y_trans .* log(hx)' + (1-Y_trans) .* log(1-hx)' ))/-m;
%         temp1 = g_Theta1(:,2:end).^2;
%         temp1 = sum(sum(temp1));
%         temp2 = g_Theta2(:,2:end).^2;
%         temp2 = sum(sum(temp2));
%         J = J + (temp1 + temp2)*lambda/2/m;
        % J = J + ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) )*lambda/2/m;
       %% Perform Back Propgation.
       	Y_trans = batchY;
        delta3 = hx - Y_trans';
        clear  Y_trans batchY;
        clear hx;
        delta2 = g_Theta2'*delta3 .* sigmoidGradient(Z2);
        clear Z2;

        delta2 = delta2(2:end,:);
        %delta3 = delta3(2:end);

        % Theta1_grad = (Theta1_grad + delta2*X)/m;
        % Theta2_grad = (Theta2_grad + delta3*A2')/m;
        %% Update the Theta.
        
        
        err(i,1) = sum(sum(delta3.^2));
        g_Theta2_grad = delta3*A2'/g_batchsize;
        clear  delta3 A2;
%         wait(gpu);
        g_Theta2_grad = g_alpha*(g_Theta2_grad/g_batchsize + g_L2* [gpuArray.zeros(size(g_Theta2,1),1,'single') g_Theta2(:,2:end)]);
        g_vTheta2 = g_vTheta2*g_momentum + g_Theta2_grad;
        g_Theta2 = g_Theta2 - g_vTheta2;
        clear g_Theta2_grad;
%         wait(gpu);
        g_vTheta1 = g_vTheta1*g_momentum;
        g_Theta1_grad = delta2*batchX;
%         g_Theta1_grad = g_Theta1_grad/g_batchsize;
%         g_Theta1_grad = delta2*batchX/g_batchsize;
%         disp(gpu.FreeMemory);
        clear batchX delta2;
%         disp(gpu.FreeMemory);
%         wait(gpu);
        g_Theta1_grad = g_Theta1_grad * (g_alpha/g_batchsize);
        g_temp = gpuArray.zeros(size(g_Theta1),'single');
        g_temp(:,2:end) = g_Theta1(:,2:end);
%         g_temp = [gpuArray.zeros(size(g_Theta1,1),1) g_Theta1(:,2:end)];
%         g_temp = g_Theta1;
%         g_temp(:,1) = 0;
        g_temp = g_temp * (g_alpha*g_L2);
        g_Theta1_grad = g_Theta1_grad + g_temp;
        clear g_temp;
%         wait(gpu);
%         g_vTheta1 = g_vTheta1*g_momentum;
        g_vTheta1 = g_vTheta1 + g_Theta1_grad;
        clear g_Theta1_grad;
%         wait(gpu);
        
%         g_Theta1_grad = g_alpha*(g_Theta1_grad/g_batchsize + g_L2* [gpuArray.zeros(size(g_Theta1,1),1,'single') g_Theta1(:,2:end)]);
%         g_vTheta1 = g_vTheta1*g_momentum + g_Theta1_grad;
        g_Theta1 = g_Theta1 - g_vTheta1;


       
%         toc;
        
        
%         wait(gpu);
        
    end
end
        

end
