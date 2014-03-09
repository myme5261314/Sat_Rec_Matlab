function [ extendMat ] = runRec(  )
%RUNREC Summary of this function goes here
%   This is the main script of this project.

params = initParams();


[nodeMap, wayMap] = loadOSM(params);
params.nodeMap = nodeMap;
params.wayMap = wayMap;

osmMat = rasterizeOSM(params);
params.osmMat = osmMat;

extendMat = extendOSM(params);
params.osmMat = extendMat;

imgSize = 1024;
xrange = randi([1 floor((params.y-params.x-imgSize)/256)], 100000, 1)*256;
yrange = randi([1 floor((params.n-params.m-imgSize)/256)], 100000, 1)*256;
count = 0;
for i = 1:100000
    tempMat = params.osmMat(yrange(i):yrange(i)+imgSize-1, xrange(i):xrange(i)+imgSize-1);
    per = numel(find(tempMat))/numel(tempMat);
    if per > 0.5
        x = xrange(i);
        y = yrange(i);
        x
        y
        count = count + 1;
        figure(1);
        subplot(2,4,count*2-1);
        imshow(round(full(tempMat)*255),[0 255]);
%         extendMat = full(tempMat);
        subplot(2,4,count*2);
        imshow(getImgMat(x,y,[imgSize imgSize],params));
        if count == 4
            break;
        end
    end
end
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

