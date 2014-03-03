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
% xrange = randi([1 floor((params.y-params.x)/256)], 100000, 1)*256;
% yrange = randi([1 floor((params.n-params.m)/256)], 100000, 1)*256;
% for i = 1:100000
%     tempMat = params.osmMat(xrange(i):xrange(i)+imgSize-1, yrange(i):yrange(i)+imgSize-1);
%     per = find(tempMat>0.1)/prod(size(tempMat));
%     if per > 0.2
%         x = xrange(i);
%         y = yrange(i);
%         x
%         y
%         figure(1);
%         imshow(floor(full(tempMat)*255));
%         figure(2);
%         imshow(getImgMat(x,y,[imgSize imgSize],params));
%         break;
%     end
% end
x = 18944+1;
y = 25088+2;
tempMat = params.osmMat(x:x+imgSize-1, y:y+imgSize-1);
figure(1);
imshow(floor(full(tempMat)*255));
figure(2);
imshow(flipdim(imrotate(getImgMat(x,y,[imgSize imgSize],params), -90), 2));


disp(params);

end

