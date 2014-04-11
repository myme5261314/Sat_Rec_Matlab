function [ s ] = img2matsize( imgSize, windowsize, stridesize )
%IMG2MATSIZE Summary of this function goes here
%   This is the function to calculate the generated data num.
blank = (windowsize-stridesize)/2;
assert(rem(blank,1)==0, 'The blank should be integer, which actual is %f'...
    , blank);
width = imgSize(2);
height = imgSize(1);
r = floor( (width-blank*2)/stridesize );
c = floor( (height-blank*2)/stridesize );
s = [r c];

end

