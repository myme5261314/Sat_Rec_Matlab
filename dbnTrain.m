function [ predY ] = dbnTrain( params )
%DBNTRAIN Summary of this function goes here
%   Detailed explanation goes here

input_layer_size = 64*64;
hidden_layer_size = 64*64*3;
output_layer_size = 16*16;

X = pca_Reduce(params.rawXmem, input_layer_size);
X = bsxfun(@rdivide, bsxfun(@minus, X, min(X)), max(X)-min(X));

y = params.rawYmem;

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0)
dbn.sizes = [ hidden_layer_size ];
opts.numepochs =   50;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
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

%train nn
opts.numepochs =  50;
opts.batchsize = 100;
nn = nntrain(nn, X, y, opts);
% [er, bad] = nntest(nn, test_x, test_y);
predY = nnpredict(nn, X);

% assert(er < 0.10, 'Too big error');
end

