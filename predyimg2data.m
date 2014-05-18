function [ dataX, dataY ] = predyimg2data( predyimg, yimg, WindowSize, StrideSize, ifrotate )
%PREDYIMG2DATA Summary of this function goes here
%   Detailed explanation goes here
if nargin==4
    ifrotate = 1;
end

fullpredimg = zeros(size(yimg), 'single');
blank = (WindowSize-StrideSize)/2;
fullpredimg( ( 1:size(predyimg,1) )+blank, ( 1:size(predyimg,2) )+blank )...
    = predyimg;
r = floor( (size(yimg,1)-blank)/StrideSize );
c = floor( (size(yimg,2)-blank)/StrideSize );
dataX = zeros(r*c, WindowSize^2);
dataY = zeros(r*c, StrideSize^2);
for i=1:r*c
    randangle = 360*rand;
    ri = floor(i/c);
    ci = mod(i, c);
    temp = fullpredimg( (1:WindowSize)+ri*StrideSize, (1:WindowSize)+ci*StrideSize );
    if ifrotate
        temp = imrotate(temp, randangle, 'crop');
    end
    temp = temp(:);
    dataX(i,:) = temp';
    temp = yimg( (1:StrideSize)+blank+ri*StrideSize, (1:StrideSize)+blank+ci*StrideSize);
    if ifrotate
        temp = imrotate(temp, randangle, 'crop');
    end
    temp = temp(:);
    dataY(i,:) = temp';
    clear temp;
end

end

