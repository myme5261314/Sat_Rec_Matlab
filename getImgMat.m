function [ ImgMat ] = getImgMat( xStart, yStart, MatSize, params )
%GETIMGMAT Summary of this function goes here
%   This is the function to generate the response img matrix.
xStart = xStart + params.x - 1;
yStart = yStart + params.m - 1;
xEnd = xStart + MatSize(1) - 1;
yEnd = yStart + MatSize(2) - 1;
TileXStart = floor(xStart/256);
TileYStart = floor(yStart/256);
TileXEnd = floor(xEnd/256);
TileYEnd = floor(yEnd/256);

ImgMat = [];
for i = TileYStart:TileYEnd
    RowImg = [];
    for j = TileXStart:TileXEnd
        xrange = max(xStart,j*256):min(xEnd,(j+1)*256-1);
        yrange = max(yStart,i*256):min(yEnd,(i+1)*256-1);
        xrange = xrange - j*256 + 1;
        yrange = yrange - i*256 + 1;
        partImg = loadTileXY(j, i, params);
        partImg = partImg(yrange, xrange, :);
%         partImg = imrotate(partImg, -90);
%         partImg = imrotate(partImg, -90);
%         partImg = permute(partImg, [2 1 3]);
        RowImg = [RowImg partImg];
%         figure(j);
%         imshow(partImg);
    end
    ImgMat = [ImgMat; RowImg];
end

end

