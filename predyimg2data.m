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
s = img2matsize(size(yimg), WindowSize, StrideSize);
r = s(1);
c = s(2);
m = r*c;
dataX = zeros(r*c, WindowSize^2);
dataY = zeros(r*c, StrideSize^2);
Ximgcell = cell(m,1);
Yimgcell = cell(m,1);
for i=1:m
    ci = floor( (i-1)/c);
    ri = mod(i-1, c);
    temp = fullpredimg( (1:WindowSize)+ri*StrideSize, (1:WindowSize)+ci*StrideSize );
    Ximgcell{i,1} = temp;
    temp = yimg( (1:StrideSize)+blank+ri*StrideSize, (1:StrideSize)+blank+ci*StrideSize);
    Yimgcell{i,1} = temp;
%     clear temp;
end
if ifrotate
    randangle = 360*rand(m,1);
end
parfor i=1:m
    temp1 = Ximgcell{i,1};
    if ifrotate
        temp1 = imrotate(temp1, randangle(i), 'crop');
    end
    dataX(i,:) = temp1(:)';
    temp1 = Yimgcell{i,1};
    if ifrotate
        temp1 = imrotate(temp1, randangle(i), 'crop');
    end
    dataY(i,:) = temp1(:)';
end

end

