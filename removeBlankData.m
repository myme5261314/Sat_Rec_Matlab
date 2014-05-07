function [ data ] = removeBlankData( data, windowsize, stridesize )
%REMOVEBLANKDATA Summary of this function goes here
%   Detailed explanation goes here
    BlankIdx = findUselessData(data, windowsize, stridesize);
    data(BlankIdx,:) = [];

end

function [idx] = findUselessData(data, WindowSize, StrideSize)
    blank = (WindowSize - StrideSize)/2;
    assert(rem(blank,1)==0, 'The blank should be integer, which actual is %f'...
    , blank);
%     rowIdx = blank+1:blank+StrideSize;
%     colIdx = blank+1:blank+StrideSize;
%     newrowIdx = repmat(rowIdx, [1 StrideSize]);
%     newColIdx = zeros(1, StrideSize^2);
%     newColIdx(1,:) = colIdx(1,ceil((1:StrideSize^2)/StrideSize));
%     subIdx = zeros(1, 3*StrideSize^2);
%     matSize = [WindowSize WindowSize 3];
%     channelIdx = ones(1, StrideSize^2);
%     subIdx(1,1:StrideSize^2) = sub2ind(matSize, newrowIdx, newColIdx, 1*channelIdx);
%     subIdx(1, StrideSize^2+1:2*StrideSize^2) = sub2ind(matSize, newrowIdx, newColIdx, 2*channelIdx);
%     subIdx(1, 2*StrideSize^2+1:3*StrideSize^2) = sub2ind(matSize, newrowIdx, newColIdx, 3*channelIdx);
%     idx = ~any(data(:,subIdx),2);
    idx = ~all(data, 2);
end