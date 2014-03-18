function [ output_args ] = runRec_mem( input_args )
%RUNREC_MEM Summary of this function goes here
%   This is the function of the main project while loading the total data
%   into memory in the process.

params = initParams();

% Load OSM xml File
[params.nodeMap, params.wayMap] = loadOSM(params);
% Rasterize the OSM Matrix
params.osmMat = rasterizeOSM(params);
% Extend the OSM Matrix
params.extendMat = extendOSM(params);
% Save OSM illustration data to image
imwrite(full(params.extendMat), params.cacheOSMImage);
% Load Bing Map Generated Map Data
if exist(params.cacheBingMapMat, 'file')
    load(params.cacheBingMapMat);
    params.BingMat = BingMat;
else
    BingMat = getImgMat(1,1,[params.numColumns params.numRows], params);
    save(params.cacheBingMapMat, 'BingMat');
    params.BingMat = BingMat;
end
clear BingMat;
% Write Bing Map illustration data to image.
imwrite(params.BingMat, params.cacheBingImage);


% [row, col, v] = find(extendMat);
% mixBingOSMMat = BingMat;
% for i=1:size(row,1)
%     mixBingOSMMat(row(i), col(i), 1) = 255*v(i);
%     mixBingOSMMat(row(i), col(i), 2) = 0;
%     mixBingOSMMat(row(i), col(i), 3) = 0;
% end
% imwrite(mixBingOSMMat, 'e:/wuhan/BingOSM_HUST.jpg');

% Generate X from the BingMap data.
params.rawXmem = genRawX(params);
% Generate Y from the OSM data.
params.rawYmem = genY(params);
% Restore the osmMat from the generated Y.
% osmMat = y2osmMat(params.rawYmem, params);
% figure(1);
% imshow(full(osmMat));
% figure(2);
% imshow(full(params.extendMat));
assert_v = osmMat(25:end-40,25:end-40) == params.extendMat(25:end-40,25:end-40);
assert(all(all(assert_v)));

predY = NNTrain_mem(params);


[precision, recall] = calAccuracy(params.extendMat, predY, params);
precision
recall

if ~exist(params.cachePredOSMMat, 'file')
    predOSMMat = y2osmMat(predY, params);
    save(params.cachePredOSMMat, 'predOSMMat');
end

whos;
% imgSize = 1024;
% xrange = randi([1 floor((params.y-params.x-imgSize)/256)], 100000, 1)*256;
% yrange = randi([1 floor((params.n-params.m-imgSize)/256)], 100000, 1)*256;
% count = 0;
% for i = 1:100000
%     tempMat = params.osmMat(yrange(i):yrange(i)+imgSize-1, xrange(i):xrange(i)+imgSize-1);
%     per = numel(find(tempMat))/numel(tempMat);
%     if per > 0.5
%         x = xrange(i);
%         y = yrange(i);
%         x
%         y
%         count = count + 1;
%         figure(1);
%         subplot(2,4,count*2-1);
%         imshow(round(full(tempMat)*255),[0 255]);
% %         extendMat = full(tempMat);
%         subplot(2,4,count*2);
%         imshow(getImgMat(x,y,[imgSize imgSize],params));
%         if count == 4
%             break;
%         end
%     end
% end
% x = 18944+1;
% y = 25088+2;
% x = 9472 + 1;
% y = 16896 + 2;
% tempMat = params.osmMat(y:y+imgSize-1, x:x+imgSize-1);
% figure(1);
% subplot(1,2,1);
% imshow(full(tempMat));
% extendMat = full(tempMat);
% subplot(1,2,2);
% imshow(getImgMat(x,y,[imgSize imgSize],params));


disp(params);

end

