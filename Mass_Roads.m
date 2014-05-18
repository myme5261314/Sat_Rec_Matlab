function [ nn, trainprecision, trainrecall, testprecision, testrecall ] = Mass_Roads( input_args )
%MASS_ROADS Summary of this function goes here
%   Detailed explanation goes here

%% Matlab Pool
if matlabpool('size')==0
    matlabpool open;
end
disp('Start Init Params');
tic;
params = initParams();
toc;
hidden_layer_size = 4096*3;
opts.numepochs =   10;
opts.batchsize = 64;
opts.momentum  =   0.9;
opts.alpha     =   0.001;
opts.L2 = 0.0002;

disp('Start RBM Stage');
tic;
if ~params.restart && exist(params.cacheRBM, 'file')
    if ~exist(params.cacheNN, 'file')
        load(params.cacheRBM);
    end
else
    rbm = rbmsetup(params, hidden_layer_size, opts);
    rbm = rbm_train(params, rbm, opts);
    save(params.cacheRBM, 'rbm', '-v7.3');
end
toc;
if exist('rbm', 'var')
    figure; visualize(rbm.W);   %  Visualize the RBM weights
end




nn.alpha = 0.0005;
disp('Start NN Stage');
tic;
if ~params.restart && exist(params.cacheNN, 'file')
    load(params.cacheNN);
else
    w = [rbm.c; rbm.W];
    clear rbm;
    nn = nn_setup(params.reduce, params.rawD, params.StrideSize^2, opts);
    nn.Theta1 = w';
    nn = nn_train(params, nn, opts);
    save(params.cacheNN, 'nn', '-v7.3');
end
toc;
%% Calculate Precision and Recall on TestSet.
disp('Start PredY on TestSet');
tic;
if ~params.restart && exist(params.cacheTestY, 'file')
    load(params.cacheTestY);
else
    params.testXYimg = loadXYFile(params.testXfile, params.testYfile);
    test_img_num = size(params.testXYimg,1);
    predtesty = cell(test_img_num,1);

    Theta1 = nn.Theta1;
    Theta2 = nn.Theta2;
    data_per_img = params.data_per_img;
    WindowSize = params.WindowSize;
    StrideSize = params.StrideSize;
    premu = params.premu;
    presigma = params.presigma;
    postmu = params.postmu;
    postsigma = params.postsigma;
    Ureduce = params.Ureduce;
    
    testXYimg = params.testXYimg;
    parfor i=1:test_img_num
        [x, ~] = xyimgIdx2data(data_per_img, WindowSize, StrideSize, testXYimg(i,:));
        x = single(x);
        x = bsxfun(@rdivide, bsxfun(@minus, x, premu), presigma);
        x = x * Ureduce;
        x = bsxfun(@rdivide, bsxfun(@minus, x, postmu), postsigma);
        x = [ ones(size(x,1),1) x ];
        Z2 = Theta1 * x';

        A2 = sigm( Z2 );
        A2 = [ ones(1,size(A2,2)) ; A2 ];
    %     Z2 = [ ones(1,size(Z2,2)) ; Z2 ];
%         idx = ((i-1)*params.data_per_img+1):i*params.data_per_img;
%         predtesty(idx,:) = sigm( nn.Theta2 * A2 )';
        predtesty{i} = sigm( Theta2 * A2 )';
        disp(['Finish ', num2str(i), ' of ', num2str(test_img_num)]);
    end
    clear Theta1 Theta2 data_per_img WindowSize StrideSize premu presigma postmu postsigma Ureduce testXYimg;
    predtesty = cell2mat(predtesty);
    save(params.cacheTestY, 'predtesty', '-v7.3');
    
end
toc;

disp('Start TestSet precision and recall Stage');
tic;
if ~params.restart && exist(params.cacheTestYmetric, 'file')
    load(params.cacheTestYmetric);
else
    data_per_img = params.data_per_img;
    datasize_per_img = params.datasize_per_img;
    [ predyimgcell ] = predy2img( data_per_img, datasize_per_img, predtesty );
    predtestyimgcell = predyimgcell;
    clear predtesty;
    thresholdlist_new = (0:1e-2:1)';
    blank = (params.WindowSize-params.StrideSize)/2;
    [testprecision, testrecall] = cal_precision_recall(blank, predyimgcell, params.testXYimg(:,2), thresholdlist_new);
    save(params.cacheTestYmetric, 'testprecision', 'testrecall');
end
[p, r] = getBestPrecisionRecall(testprecision, testrecall);
disp(['The best precision: ', num2str(p), '. And the best recall: ', num2str(r), '.']);
toc;

figure('Name', 'Test Set');
plot(testrecall, testprecision);

%% Calculate Precision and Recall on TrainSet
disp('Start PredY Stage on TrainSet');
tic;
if ~params.restart && exist(params.cacheTrainY, 'file')
    load(params.cacheTrainY);
else
    params.trainXYimg = loadXYFile(params.trainXfile, params.trainYfile);
    train_img_num = size(params.trainXYimg,1);
%     predtrainy = zeros(train_img_num*params.data_per_img, 256);
    predtrainy = cell(train_img_num,1);
    
    Theta1 = nn.Theta1;
    Theta2 = nn.Theta2;
    data_per_img = params.data_per_img;
    WindowSize = params.WindowSize;
    StrideSize = params.StrideSize;
    premu = params.premu;
    presigma = params.presigma;
    postmu = params.postmu;
    postsigma = params.postsigma;
    Ureduce = params.Ureduce;
    
    trainXYimg = params.trainXYimg;
    parfor i=1:train_img_num
        [x, ~] = xyimgIdx2data(data_per_img, WindowSize, StrideSize, trainXYimg(i,:));
        x = single(x);
        x = bsxfun(@rdivide, bsxfun(@minus, x, premu), presigma);
        x = x * Ureduce;
        x = bsxfun(@rdivide, bsxfun(@minus, x, postmu), postsigma);
        x = [ ones(size(x,1),1) x ];
        Z2 = Theta1 * x';

        A2 = sigm( Z2 );
        A2 = [ ones(1,size(A2,2)) ; A2 ];
    %     Z2 = [ ones(1,size(Z2,2)) ; Z2 ];
%         idx = ((i-1)*params.data_per_img+1):i*params.data_per_img;
%         predtesty(idx,:) = sigm( nn.Theta2 * A2 )';
        predtrainy{i} = sigm( Theta2 * A2 )';
        disp(['Finish ', num2str(i), ' of ', num2str(train_img_num)]);
    end
    clear Theta1 Theta2 data_per_img WindowSize StrideSize premu presigma postmu postsigma Ureduce testXYimg;
    predtrainy = cell2mat(predtrainy);
    save(params.cacheTrainY, 'predtrainy', '-v7.3');
    
end
toc;

disp('Start TrainSet precision and recall Stage');
tic;
if ~params.restart && exist(params.cacheTrainYmetric, 'file')
    load(params.cacheTrainYmetric);
else
    data_per_img = params.data_per_img;
    datasize_per_img = params.datasize_per_img;
    [ predyimgcell ] = predy2img( data_per_img, datasize_per_img, predtrainy );
    predtrainyimgcell = predyimgcell;
    clear predtrainy;
    thresholdlist_new = (0:1e-2:1)';
    [trainprecision, trainrecall] = cal_precision_recall(blank, predyimgcell, params.trainXYimg(:,2), thresholdlist_new);
    save(params.cacheTrainYmetric, 'trainprecision', 'trainrecall');
end

[p, r] = getBestPrecisionRecall(trainprecision, trainprecision);
disp(['The best precision: ', num2str(p), '. And the best recall: ', num2str(r), '.']);
toc;

figure('Name', 'Train Set');
plot(trainrecall, trainprecision);



save('precision_recall_rbm.mat', 'trainprecision', 'trainrecall', 'testprecision', 'testrecall');

%% Post-Process
if ~params.restart && exist(params.cachePostNN, 'file')
    load(params.cachePostNN);
else
    postnn = nn_setup(params.reduce, params.reduce, params.StrideSize^2, opts);
    if ~params.restart && exist(params.cachePostNNMu, 'file')
        load(params.cachePostNNMu);
    else
        postnnmu = calpostNNMu(params, predtrainyimgcell);
        save(params.cachePostNNMu, 'postnnmu');
    end
    if ~params.restart && exist(params.cachePostNNSigma, 'file')
        load(params.cachePostNNSigma);
    else
        postnnsigma = calpostNNSigma(params, predtrainyimgcell, postnnmu);
    end
end

%% Post-Process Evaluation.
%% Calculate Precision and Recall on TestSet.
disp('Start PredY on TestSet With Post-Process');
tic;
if ~params.restart && exist(params.cachePostTestY, 'file')
    load(params.cachePostTestY);
else
    params.testXYimg = loadXYFile(params.testXfile, params.testYfile);
    test_img_num = size(params.testXYimg,1);
    predposttesty = cell(test_img_num,1);

    Theta1 = postnn.Theta1;
    Theta2 = postnn.Theta2;

    WindowSize = params.WindowSize;
    StrideSize = params.StrideSize;

    
    testXYimg = params.testXYimg;
    parfor i=1:test_img_num
        [x, ~] = predyimg2data(predtestyimgcell{i,1}, testXYimg{i,2},...
            WindowSize, StrideSize, false);
        x = single(x);
        x = bsxfun(@rdivide, bsxfun(@minus, x, postnnmu), postnnsigma);
        x = [ ones(size(x,1),1) x ];
        Z2 = Theta1 * x';

        A2 = sigm( Z2 );
        A2 = [ ones(1,size(A2,2)) ; A2 ];

        predposttesty{i} = sigm( Theta2 * A2 )';
    end
    clear Theta1 Theta2 WindowSize StrideSize testXYimg;
    predposttesty = cell2mat(predposttesty);
    save(params.cachePostTestY, 'predposttesty', '-v7.3');
    
end
toc;

disp('Start TestSet precision and recall Stage');
tic;
if ~params.restart && exist(params.cachePostTestYmetric, 'file')
    load(params.cachePostTestYmetric);
else
    data_per_img = params.data_per_img;
    datasize_per_img = params.datasize_per_img;

    [ predyimgcell ] = predy2img( data_per_img, datasize_per_img, predposttesty );
    clear predtesty;
    thresholdlist_new = (0:1e-2:1)';
    blank = (params.WindowSize-params.StrideSize)/2;
    [posttestprecision, posttestrecall] = cal_precision_recall(blank, predyimgcell, params.testXYimg(:,2), thresholdlist_new);
    save(params.cachePostTestYmetric, 'posttestprecision', 'posttestrecall');
end
[p, r] = getBestPrecisionRecall(posttestprecision, posttestrecall);
disp(['The best precision: ', num2str(p), '. And the best recall: ', num2str(r), '.']);
toc;

figure('Name', 'Test Set');
plot(posttestrecall, posttestprecision);

matlabpool close;

end

function [ p, r ] = getBestPrecisionRecall(precision, recall)
    f1 = precision.*recall./(precision+recall);
    f1(isnan(f1)) = -1;
    [~, idx] = max(f1);
    p = precision(idx);
    r = recall(idx);
end
