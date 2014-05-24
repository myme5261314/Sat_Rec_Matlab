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
params.cachepreRotate = fullfile(params.cacheFloder, 'rotate.mat');

params.cacheImageNum = 5;

%% Get Data File path.
[params.trainXfile, params.trainYfile] = getDataSetFilePath(...
    params.dataFloder, params.trainFloder, params.satFloder,...
    params.mapFloder);
[params.testXfile, params.testYfile] = getDataSetFilePath(...
    params.dataFloder, params.testFloder, params.satFloder,...
    params.mapFloder);
if params.debug
    idx = 1:params.debugSize;
%     randidx = randperm(size(params.trainXfile,2));
%     idx = randidx(idx);
    params.trainXfile = params.trainXfile(1,idx);
    params.trainYfile = params.trainYfile(1,idx);
    idx = 1:5;
    params.testXfile = params.testXfile(1,idx);
    params.testYfile = params.testYfile(1,idx);
end
params.trainImgNum = size(params.trainXfile,2);
params.m = params.trainImgNum * params.data_per_img;

%% Load Img.
params.trainXYimg = loadXYFile(params.trainXfile, params.trainYfile);
params.testXYimg = loadXYFile(params.testXfile, params.testYfile);

%% Load Cached Rotate Angle
if ~params.restart && exist(params.cachepreRotate, 'file')
    load(params.cachepreRotate);
    params.trainRotate = trainRotate;
    params.testRotate = testRotate;
else
    trainRotate = 360*rand(params.trainImgNum,1);
    testRotate = 360*rand( size(params.testXfile, 2), 1 );
    save(params.cachepreRotate, 'trainRotate', 'testRotate');
    params.trainRotate = trainRotate;
    params.testRotate = testRotate;
end

% %% Rotate The Image.
% for i=1:params.trainImgNum
%     params.trainXYimg{i,1} = imrotate(params.trainXYimg{i,1}, params.trainRotate(i));
%     params.trainXYimg{i,2} = imrotate(params.trainXYimg{i,2}, params.trainRotate(i));
% end
% for i=1:size(params.testXfile, 2)
%     params.testXYimg{i,1} = imrotate(params.testXYimg{i,1}, params.testRotate(i));
%     params.testXYimg{i,2} = imrotate(params.testXYimg{i,2}, params.testRotate(i));
% end
    

%% Calculate the mean of X among entire dataset.
params.cachepreMeanFile = fullfile(params.cacheFloder, 'premu.mat');
if ~params.restart && exist(params.cachepreMeanFile, 'file')
    load(params.cachepreMeanFile);
    params.premu = premu;
    clear premu;
else
    datanum_per_img = [];
    [premu, datanum] = calpreMean(params.trainXYimg... 
        , params.WindowSize, params.StrideSize);
    datanum_per_img = [datanum_per_img; datanum];
    
    premu = sum(bsxfun(@times, premu, datanum_per_img))/sum(datanum_per_img);
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
    datanum_per_img = [];
    [presigma, datanum] = calpreStd(params.trainXYimg... 
        , params.WindowSize, params.StrideSize, params.premu);
    datanum_per_img = [datanum_per_img; datanum];
    
    presigma = sum(bsxfun(@times, presigma, datanum_per_img))/sum(datanum_per_img);
    presigma = sqrt(presigma);
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
    totalnum = 0;
    [sig, totalnum] = calCov(params.trainXYimg... 
        , params.WindowSize, params.StrideSize, params.premu, params.presigma);
    
    sig = sig/totalnum;
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
    datanum_per_img = [];
    [postmu, datanum_per_img] = calpostMean(params.trainXYimg... 
        , params.WindowSize, params.StrideSize...
        , params.premu, params.presigma, params.Ureduce);

    postmu = sum(bsxfun(@times, postmu, datanum_per_img))/sum(datanum_per_img);
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
    datanum_per_img = [];
    [postsigma, datanum_per_img] = calpostStd(params.trainXYimg... 
        , params.WindowSize, params.StrideSize...
        , params.premu, params.presigma, params.Ureduce, params.postmu);
    
    postsigma = sum(bsxfun(@times, postsigma, datanum_per_img))/sum(datanum_per_img);
    postsigma = sqrt(postsigma);
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
