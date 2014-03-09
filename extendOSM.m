function [ extendMat ] = extendOSM( params )
%EXTENDOSM Summary of this function goes here
%   This function extend the points near the 1's in the result of
%   rasterizeOSM.

memMat = memmapfile('OSMMat.dat', 'format', {'double', [params.numRows params.numColumns], 'mat' });
memMat.Writable = true;

% if exist(params.cacheExtendOSM, 'file')
if exist('OSMMat.dat','file')
%     load(params.cacheExtendOSM);
    extendMat = memMat;
    return
end

lat = mean([params.lat_south params.lat_north]);
resolution = cos(lat*pi/180)*2*pi*6378137/(256*2^params.z);
feet = 0.3048;
sig = 24*feet;
%   In our experiment ¦Ò was set such that the distance equivalent to 2¦Ò+1
%   pixels roughly corresponds to the width of a typical two-lane road.
sig = (sig-1)/2/resolution;
% sig
d = 1:10000;
v = exp(-(d/sig).^2);
% % plot(d(1:200),-log10(v(1:200)));
threshold = find(v);
threshold = length(threshold) + 1;
vF = @(mat) exp(-(bwdist(mat)/sig).^2);
memMat.Data.mat(:,:) = vF(memMat.Data.mat);
clear memMat;
memMat = memmapfile('OSMMat.dat', 'format', {'double', [params.numRows params.numColumns], 'mat' });
extendMat = memMat.Data.mat;
% nnz(memMat.Data.mat)
% memMat.Data.mat(memMat.Data.mat<threshold) = 0;
% % threshold = 10 * threshold;
% [h, w] = size(params.osmMat);
% % extendMat = sparse(h,w);
% % Split partition to use bwdist, because the bwdist doesn't support sparse.
% hCount = ceil(h/threshold);
% wCount = ceil(w/threshold);
% count = 0;
% per = 0.01;
% rTotal = [];
% cTotal = [];
% vTotal = [];
% for i = 1:hCount
%     for j = 1:wCount
%         count = count + 1;
%         if count/hCount/wCount > per
%             fprintf('%d/%d\n', count, hCount*wCount);
%             per = per + 0.01;
% %             temp = sparse(rTotal, cTotal, vTotal, h, w);
% %             extendMat = extendMat + temp;
% %             clear temp rTotal cTotal vTotal;
% %             rTotal = [];
% %             cTotal = [];
% %             vTotal = [];
%             whos
%         end
% %         if i == 1
% %             top = 1;
% %             bottom = 2*threshold;
% %         elseif i == hCount
% %             top = (i - 1) * threshold + 1;
% %             bottom = h;
% %         else
% %             top = (i - 1) * threshold + 1;
% %             bottom = (i + 1) * threshold;
% %         end
%         top = max(i - 2, 0) * threshold + 1;
%         bottom = min( (i + 1) * threshold, h);
%         left = max(j - 2, 0) * threshold + 1;
%         right = min( (j + 1) * threshold, w);
%         
%         tempPartition = params.osmMat(top:bottom, left:right);
%         tempPartition = bwdist(full(tempPartition));
%         
%         t = (i - 1) * threshold + 1 - top + 1;
%         b = t + min(threshold - 1, h - top - threshold);
%         l = (j - 1) * threshold + 1 - left + 1;
%         r = l + min(threshold - 1, w - left - threshold);
%         tempPartition = double(tempPartition(t:b, l:r));
%         tempPartition = exp(-(tempPartition/sig).^2);
%         tempPartition = sparse(tempPartition);
% %         i,j
% %         tempRow = [tempRow tempPartition];
%         rStart = (i - 1) * threshold + 1;
%         cStart = (j - 1) * threshold + 1;
%         rRange = rStart:rStart + size(tempPartition,1) - 1;
%         cRange = cStart:cStart + size(tempPartition,2) - 1;
% %         extendMat(rRange,cRange) = tempPartition;
%         idx = find(tempPartition > params.threshold);
%         v = full(tempPartition(idx));
%         [rv, cv] = ind2sub(size(tempPartition),idx);
%         if ~isempty(v)
%             rv = rv + rStart - 1;
%             cv = cv + cStart - 1;
%             rTotal = [rTotal; rv];
%             cTotal = [cTotal; cv];
%             vTotal = [vTotal; v];
%         end
% %         idx = sub2ind(size(extendMat), rv, cv);
% %         extendMat(idx) = v;
%         if i == hCount
%             assert(rRange(end) == h);
%         end
%         if  j == wCount
%             assert(cRange(end) == w);
%         end
% %             
% %             
% %         top = max(i-1, 1);
% %         bottom = min(i+1, hCount);
% %         left = max(j-1, 1);
% %         right = min(j+1, wCount);
% %         tempPartition = params.osmMat((top:bottom)*threshold, (left:right)*threshold);
% %         tempPartition = bwdist(full(tempPartition));
% %         extendRow = [extendRow sparse(tempPartition)];
%     end
% %     extendMat = [extendMat; tempRow];
% end
% 
% extendMat = sparse(rTotal, cTotal, vTotal, h, w);
% clear rTotal cTotal vTotal;
% whos
% save(params.cacheExtendOSM,'extendMat','-v7.3');

        
        

% size(params.osmMat)
% extendMat = bwdist(params.osmMat);
% extendMat = e.^(-(extendOSM/sig).^2);
% spdiags

end

