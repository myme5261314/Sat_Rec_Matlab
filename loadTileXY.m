function [ imgMat ] = loadTileXY( x, y, params )
%LOADTILEXY Summary of this function goes here
%   This is the function to load tile x,y img matrix.

fpath = params.dataFloder;
quadkey = tileXY2quadkey(x, y, params.z);
filegroup = floor(base2dec(quadkey, 4)/params.filegroup);
% This line of the code runs too slow than the below one
% filegroup = strcat(int2str(filegroup), '_', int2str(params.filegroup));
filegroup = sprintf('%d_%d', filegroup, params.filegroup);
quadkey = sprintf('%s.jpg', quadkey);
% This line of the code runs too slow than the below one
% fpath = fullfile(fpath, int2str(params.z), filegroup, quadkey);
fpath = sprintf('%s%d/%s/%s', fpath, params.z, filegroup, quadkey);
imgMat = imread(fpath);


end

