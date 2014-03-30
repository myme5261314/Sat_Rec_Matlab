function [ predY ] = dbnTrain( params )
%DBNTRAIN Summary of this function goes here
%   Detailed explanation goes here

input_layer_size = 64*64;
hidden_layer_size = 64*64*3;
output_layer_size = 16*16;

if ~exist('E:/wuhan/pcaX.mat', 'file')
    X = pca_Reduce(params.rawXmem, input_layer_size);
%     X = bsxfun(@rdivide, bsxfun(@minus, X, min(X)), max(X)-min(X));
    save('E:/wuhan/pcaX.mat', 'X');
else
    load('E:/wuhan/pcaX.mat');
end
X = bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), std(X));
y = params.rawYmem;

%%  ex1 train a 100 hidden unit RBM and visualize its weights
factorlist = factor(size(X,1));
opts.batchsize = 1;
for i=length(factorlist):-1:1
    opts.batchsize = opts.batchsize * factorlist(i);
    if opts.batchsize>=40
        break;
    end
end
% opts.batchsize = 48;
if ~exist('e:/wuhan/InitialTheta.mat', 'file')
    rand('state',0)
    dbn.sizes = [ hidden_layer_size ];
    opts.numepochs =   50;
    % opts.batchsize = 100;
    opts.momentum  =   0.9;
    opts.alpha     =   0.001;
    opts.L2 = 0.0002;
    dbn = dbnsetup(dbn, X, opts);
    dbn = dbntrain(dbn, X, opts);
    figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

% %%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
% rand('state',0)
% %train dbn
% dbn.sizes = [100 100];
% opts.numepochs =   1;
% opts.batchsize = 100;
% opts.momentum  =   0;
% opts.alpha     =   1;
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn

    nn = dbnunfoldtonn(dbn, output_layer_size);
    nn.activation_function = 'sigm';
    save('e:/wuhan/InitialTheta.mat', 'nn');
else
    load('e:/wuhan/InitialTheta.mat');
end


%train nn
opts.numepochs =  50;
% opts.batchsize = 100;
opts.alpha = 0.001;
opts.validation = 0;
nn.learningRate = 0.0005;
nn.momentum = 0.9;
nn.weightPenaltyL2 = 0.0002;
nn.W{1} = gpuArray(nn.W{1});
nn.W{2} = gpuArray(nn.W{2});
nn.vW{1} = gpuArray(nn.vW{1});
nn.vW{2} = gpuArray(nn.vW{2});
nn = nntrain(nn, X, y, opts);
% [er, bad] = nntest(nn, test_x, test_y);
% predY = nnpredict(nn, X);
nn.testing = 1;
batchsize = opts.batchsize;
numbatches = size(X,1) / batchsize;
predY = [];
for i=1:numbatches
    batch_x = X((i-1)*batchsize+1:i*batchsize, :);
    batch_x = gpuArray(batch_x);
    nn = nnff(nn, batch_x, zeros(size(batch_x,1), nn.size(end)));
    batch_y = gather(nn.a{end});
    predY = [predY; batch_y];
end
% nn = nnff(nn, X, zeros(size(X,1), nn.size(end)));
nn.testing = 0;
% predY = gather(nn.a{end});

% assert(er < 0.10, 'Too big error');
end

