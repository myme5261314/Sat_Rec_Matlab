function params = initParams()
% This is the configuration file of this project

%% Init params configuration
params.debug = 1;
params.debugSize = 100;
params.portion = 0.001;

params.restart = 0;
params.imgSize = [1500 1500];
params.WindowSize = 64;
params.StrideSize = 16;
params.datasize_per_img = img2matsize(params.imgSize, params.WindowSize,...
    params.StrideSize);
params.data_per_img = prod(params.datasize_per_img);

params.rawD = params.WindowSize^2*3;
params.reduce = params.WindowSize^2;

% Set the data folder here
os = getenv('os');
if strcmp(os, 'Windows_NT')
    params.dataFloder = 'E:/Sat_Rec_Dataset/Mass_Roads/';
else
    params.dataFloder = '/home/cug/Sat_Rec_Dataset/Mass_Roads/';
end
params.trainFloder = 'Train';
params.validFloder = 'Valid';
params.testFloder = 'Test';
params.satFloder = 'Sat';
params.mapFloder = 'Map';

params.cacheFloder = fullfile(params.dataFloder, 'cache');
params.cacheRBM = fullfile(params.cacheFloder, 'rbm.mat');
params.cacheEpochRBM = fullfile(params.cacheFloder, 'epochrbm.mat');
params.cacheEpochNN = fullfile(params.cacheFloder, 'epochnn.mat');
params.cacheNN = fullfile(params.cacheFloder, 'nn.mat');
params.cacheTestY = fullfile(params.cacheFloder, 'testy.mat');
params.cacheTrainY = fullfile(params.cacheFloder, 'trainy.mat');

params.cacheImageNum = 5;

%% Get Data File path.
[params.trainXfile, params.trainYfile] = getDataSetFilePath(...
    params.dataFloder, params.trainFloder, params.satFloder,...
    params.mapFloder);
[params.validXfile, params.validYfile] = getDataSetFilePath(...
    params.dataFloder, params.validFloder, params.satFloder,...
    params.mapFloder);
[params.testXfile, params.testYfile] = getDataSetFilePath(...
    params.dataFloder, params.testFloder, params.satFloder,...
    params.mapFloder);
if params.debug
    idx = 1:params.debugSize;
    randidx = randperm(size(params.trainXfile,2));
    idx = randidx(idx);
    params.trainXfile = params.trainXfile(1,idx);
    params.trainYfile = params.trainYfile(1,idx);
    idx = 1:5;
    params.validXfile = params.validXfile(1,idx);
    params.validYfile = params.validYfile(1,idx);
    params.testXfile = params.testXfile(1,idx);
    params.testYfile = params.testYfile(1,idx);
end
params.trainImgNum = size(params.trainXfile,2);
params.m = params.trainImgNum * params.data_per_img;

%% Load Img.
params.trainXYimg = loadXYFile(params.trainXfile, params.trainYfile);
params.validXYimg = loadXYFile(params.validXfile, params.validYfile);
params.testXYimg = loadXYFile(params.testXfile, params.testYfile);

%% Calculate the mean of X among entire dataset.
params.cachepreMeanFile = fullfile(params.cacheFloder, 'premu.mat');
if ~params.restart && exist(params.cachepreMeanFile, 'file')
    load(params.cachepreMeanFile);
    params.premu = premu;
    clear premu;
else
    premu = calpreMean(params.trainXYimg... 
        , params.WindowSize, params.StrideSize);
    premu = [premu; ...
        calpreMean(params.validXYimg, params.WindowSize, params.StrideSize)];
    premu = [premu; ...
        calpreMean(params.testXYimg, params.WindowSize, params.StrideSize)];
    premu = mean(premu);
    params.premu = premu;
    save(params.cachepreMeanFile, 'premu');
    clear premu;
end

%% Calculate the std of X among the entire dataset.
params.cachepreStdFile = fullfile(params.cacheFloder, 'presigma.mat');
if ~params.restart && exist(params.cachepreStdFile, 'file')
    load(params.cachepreStdFile, 'presigma');
    params.presigma = presigma;
    clear presigma;
else
    presigma = calpreStd(params.trainXYimg... 
        , params.WindowSize, params.StrideSize, params.premu);
    presigma = [presigma; ...
        calpreStd(params.validXYimg, params.WindowSize, params.StrideSize, params.premu)];
    presigma = [presigma; ...
        calpreStd(params.testXYimg, params.WindowSize, params.StrideSize, params.premu)];
    presigma = sqrt(mean(presigma));
    params.presigma = presigma;
    save(params.cachepreStdFile, 'presigma');
    clear presigma;
end

%% Calculate the pca Ureduce Matrix.

params.cacheUreduceFile = fullfile(params.cacheFloder, 'Ureduce.mat');
if ~params.restart && exist(params.cacheUreduceFile, 'file')
    load(params.cacheUreduceFile, 'Ureduce');
    params.Ureduce = Ureduce;
    clear Ureduce;
    load(params.cacheUreduceFile, 'S');
    params.S = S;
    clear S;
    load(params.cacheUreduceFile, 'per');
    fprintf('The remaining covariance is %f.\n', per);
else
    sig = calCov(params.trainXYimg... 
        , params.WindowSize, params.StrideSize, params.premu, params.presigma);
    sig = sig + calCov(params.validXYimg,...
        params.WindowSize, params.StrideSize, params.premu, params.presigma);
    sig = sig + calCov(params.testXYimg,...
        params.WindowSize, params.StrideSize, params.premu, params.presigma);
    sig = sig/3;
    sig = single(sig);
    sig = gpuArray(sig);
    [U, S, ~] = svd(sig);
    U = gather(U);
    U = double(U);
    S = gather(S);
    S = double(S);
    Ureduce = U(:,1:params.reduce);
    
    params.Ureduce = Ureduce;
    S = diag(S);
    per = sum(S(1:4096))/sum(S);
    fprintf('The remaining covariance is %f', per);
    save(params.cacheUreduceFile, 'U', 'Ureduce', 'S', 'per');
    clear U;
end


%% Calculate the post mean after PCA reduce.
params.cachepostMeanFile = fullfile(params.cacheFloder, 'postmu.mat');
if ~params.restart && exist(params.cachepostMeanFile, 'file')
    load(params.cachepostMeanFile);
    params.postmu = postmu;
    clear postmu;
else
    postmu = calpostMean(params.trainXYimg... 
        , params.WindowSize, params.StrideSize...
        , params.premu, params.presigma, params.Ureduce);
    postmu = [postmu; ...
        calpostMean(params.validXYimg, params.WindowSize, params.StrideSize...
        , params.premu, params.presigma, params.Ureduce)];
    postmu = [postmu; ...
        calpostMean(params.testXYimg, params.WindowSize, params.StrideSize...
        , params.premu, params.presigma, params.Ureduce)];
    postmu = mean(postmu);
    params.postmu = postmu;
    save(params.cachepostMeanFile, 'postmu');
    clear postmu;
end

%% Calculate the post std after PCA reduce.
params.cachepostStdFile = fullfile(params.cacheFloder, 'postsigma.mat');
if ~params.restart && exist(params.cachepostStdFile, 'file')
    load(params.cachepostStdFile);
    params.postsigma = postsigma;
    clear postsigma;
else
    postsigma = calpostStd(params.trainXYimg... 
        , params.WindowSize, params.StrideSize...
        , params.premu, params.presigma, params.Ureduce, params.postmu);
    postsigma = [postsigma; ...
        calpostStd(params.validXYimg, params.WindowSize, params.StrideSize...
        , params.premu, params.presigma, params.Ureduce, params.postmu)];
    postsigma = [postsigma; ...
        calpostStd(params.testXYimg, params.WindowSize, params.StrideSize...
        , params.premu, params.presigma, params.Ureduce, params.postmu)];
    postsigma = sqrt(mean(postsigma));
    params.postsigma = postsigma;
    save(params.cachepostStdFile, 'postsigma');
    clear postsigma;
end

% disp(params);

%% Generate the random Index to use the Raw Img Data.
[params.imgIdx, params.imgDataIdx] = randIdx(params);
params.currentImgIdx = 1;
params.currentImgDataIdx = 1;



end
