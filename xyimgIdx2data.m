function [ x, y ] = xyimgIdx2data( params, xyimgcell, imgIdx, dataIdx )
%XYIMGIDX2DATA Summary of this function goes here
%   Detailed explanation goes here
% xy = cell(1,2);

if nargin==3
    dataIdx = 1:params.data_per_img;
end

x = ximg2data(xyimgcell{imgIdx,1},...
    params.WindowSize, params.StrideSize, dataIdx);
y = yimg2data(xyimgcell{imgIdx,2},...
    params.WindowSize, params.StrideSize, dataIdx);


end

