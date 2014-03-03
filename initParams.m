% This is the configuration file of this project
function params = initParams()
% use a small portion of dataset for debugging
params.debug = 1;
params.portion = 0.001;

% Set the data folder here
params.dataFolder = 'E:/wuhan/';

% The Whole OSM file
if params.debug
    params.osmFile = 'E:/wuhan/cug.osm';
else
    params.osmFile = 'E:/wuhan/wuhan.osm';
end

% The whole OSM area
if params.debug
    params.lat_south = 30.4969053;
    params.lat_north = 30.562261;
    params.lon_west = 114.3463898;
    params.lon_east = 114.4294739;
else
    params.lat_south = 29.9828;
    params.lat_north = 31.3703;
    params.lon_west = 113.6994;
    params.lon_east = 115.0869;
end

params.z = 19;

params.filegroup = 10000;

% cache file
if params.debug
    params.cacheOSM = 'E:/wuhan/cugOSM.mat';
    params.cacheOSMMat = 'E:/wuhan/cugOSMMat.mat';
    params.cacheExtendOSM = 'E:/wuhan/cugExtendOSMMat.mat';
else
    params.cacheOSM = 'E:/wuhan/wuhanOSM.mat';
    params.cacheOSMMat = 'E:/wuhan/wuhanOSMMat.mat';
    params.cacheExtendOSM = 'E:/wuhan/wuhanExtendOSMMat.mat';
end

params.threshold = 1e-4;

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
