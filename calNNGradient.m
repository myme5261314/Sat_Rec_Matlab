function [ t_vTheta1, t_vTheta2, t_error ] = calNNGradient( batchX, batchY, g_premu, g_presigma, g_Ureduce, g_postmu, g_postsigma, g_Theta1, g_Theta2 )
%CALNNGRADIENT Summary of this function goes here
%   Detailed explanation goes here

       %% Preprocess the Raw Batch X.
        batchX = gpuArray(single(batchX));
        batchY = gpuArray(single(batchY));
        batchX = bsxfun(@rdivide, bsxfun(@minus, batchX, g_premu), g_presigma);
        batchX = batchX * g_Ureduce;
        batchX = bsxfun(@rdivide, bsxfun(@minus, batchX, g_postmu), g_postsigma);
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
        t_error = sum(sum(delta3.^2));
        t_vTheta1 = delta2*batchX;
        t_vTheta2 = delta3*A2';

end

