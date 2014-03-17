% This is the configuration file of this project
function params = initParams()
% use a small portion of dataset for debugging
params.debug = 1;
params.portion = 0.001;

% Set the data folder here
params.dataFolder = 'E:/wuhan/';

% The Whole OSM file
if params.debug
%     params.osmFile = 'E:/wuhan/cug.osm';
    params.osmFile = 'E:/wuhan/hust.osm';
else
    params.osmFile = 'E:/wuhan/wuhan.osm';
end

% The whole OSM area
% if params.debug
%     params.lat_south = 30.4969053;
%     params.lat_north = 30.562261;
%     params.lon_west = 114.3463898;
%     params.lon_east = 114.4294739;
% else
%     params.lat_south = 29.9828;
%     params.lat_north = 31.3703;
%     params.lon_west = 113.6994;
%     params.lon_east = 115.0869;
% end
[params.lat_south, params.lat_north, params.lon_west, params.lon_east]=...,
    loadOSMRange(params.osmFile);

params.z = 19;

params.filegroup = 10000;

% cache file
if params.debug
%     params.cacheOSM = 'E:/wuhan/cugOSM.mat';
%     params.cacheOSMMat = 'E:/wuhan/cugOSMMat.mat';
%     params.cacheExtendOSM = 'E:/wuhan/cugExtendOSMMat.mat';
    params.cacheOSM = fullfile(params.dataFolder, 'hustOSM.mat');
    params.cacheOSMMat = fullfile(params.dataFolder, 'hustOSMMat.mat');
    params.cacheExtendOSM = fullfile(params.dataFolder, 'hustExtendOSMMat.mat');
    params.cacheBingMapMat = fullfile(params.dataFolder, 'BingMap_HUST.mat');
    params.cacheBingImage = fullfile(params.dataFolder, 'BingMap_HUST.jpg');
    params.cacheOSMImage = fullfile(params.dataFolder, 'osmMap_HUST.jpg');
    params.cacheRawXMat = fullfile(params.dataFolder, 'RawX.mat');
    params.cacheRawYMat = fullfile(params.dataFolder, 'RawY.mat');
    
else
    params.cacheOSM = 'E:/wuhan/wuhanOSM.mat';
    params.cacheOSMMat = 'E:/wuhan/wuhanOSMMat.mat';
    params.cacheExtendOSM = 'E:/wuhan/wuhanExtendOSMMat.mat';
end

params.threshold = 1e-4;
% The image size with same width and height.
params.imgSize = 256;

% x,y stands for the min-max lontitude pixel(column) and m,n stands for
% the min-max latitude pixel(row).
[params.x, params.m] = latlon2p(params.lat_north, params.lon_west,params.z);
[params.y, params.n] = latlon2p(params.lat_south, params.lon_east,params.z);
% To let the area pixel coordinate start with the multiple of imgSize
% index.
params.x = params.x + (params.imgSize - mod(params.x, params.imgSize));
params.m = params.m + (params.imgSize - mod(params.m, params.imgSize));
params.y = params.y - mod(params.y, params.imgSize);
params.n = params.n - mod(params.n, params.imgSize);
% The area pixel Matrix size.
params.numRows = params.n - params.m;
params.numColumns = params.y - params.x;

% % set the data split to train and test on 
% params.split = 2;
% 
% % set the number of first layer CNN filters
% params.numFilters = 128;
% 
% % set the number of RNN to use
% params.numRNN = 64;
% 
% % use depth or rgb information
% params.depth = false;
% 
% % use extra features from segmentation mask
% params.extraFeatures = true;
