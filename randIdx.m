function [ imgIdx, imgDataIdx ] = randIdx( params )
%RANDIDX Summary of this function goes here
%   Detailed explanation goes here
imgIdx = randperm(params.trainImgNum);
imgDataIdx = zeros(params.trainImgNum, prod(params.datasize_per_img));
for i=1:params.trainImgNum
    imgDataIdx(i,:) = randperm( prod(params.datasize_per_img) );
end

end

