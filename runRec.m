function [ extendMat ] = runRec(  )
%RUNREC Summary of this function goes here
%   This is the main script of this project.

params = initParams();

% x,y stands for the min-max lontitude pixel(column) and m,n stands for
% the min-max latitude pixel(row).
[x, m] = latlon2p(params.lat_north, params.lon_west,params.z);
[y, n] = latlon2p(params.lat_south, params.lon_east,params.z);
x = floor(x);
m = floor(m);
y = ceil(y);
n = ceil(n);
% osmMat = sparse(y-x, n-m);
params.x = x;
params.y = y;
params.m = m;
params.n = n;
% params.osmMat = osmMat;

% output_args = osmMat;

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

