function [ imgMat ] = loadTileXY( x, y, params )
%LOADTILEXY Summary of this function goes here
%   This is the function to load tile x,y img matrix.

fpath = params.dataFolder;
quadkey = tileXY2quadkey(x, y, params.z);
filegroup = floor(base2dec(quadkey, 4)/params.filegroup);
filegroup = strcat(int2str(filegroup), '_', int2str(params.filegroup));
quadkey = strcat(quadkey, '.jpg');
fpath = fullfile(fpath, int2str(params.z), filegroup, quadkey);
imgMat = imread(fpath);


end

