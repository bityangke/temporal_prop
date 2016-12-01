function new_feature = temporal_max_pooling(feature, sub_division_size)
% Input:
%         feature       : WHxTxC single array containing spatial featurestc
%         sub_division_size   : number of temporal max pooling grids: ex) 2,3,4,10,etc
% Output:
%         new_feature   : WHxNTxC cell array containing temporal max pooled
%                         spatial features. (NT=int((M-filter_size)/stride)+1) < M

T = size(feature,2);
stride = floor(T/sub_division_size);
if stride == 0
    stride =T;
end

for i=1:sub_division_size
    new_feature(:,i,:) = max(feature(:,(i-1)*stride+1:i*stride,:),[],2);
end

% function new_feature = temporal_max_pooling(feature, filter_size, stride)
% % Input:
% %         feature       : Mx1 cell array containing spatial features
% %                         feature{i} is one spatial feature map of any ConvNet
% %                         layers: ex) conv5, pool5, etc
% %         filter_size   : temporal max pooling filter size: ex) 2,3,4,etc
% %         stride        : temporal max pooling stride: ex) 2,3,4,etc 
% % Output:
% %         new_feature   : Nx1 cell array containing temporal max pooled
% %                         spatial features. (N=int((M-filter_size)/stride)+1) < M
% 
% M = size(feature,1);
% 
% for i=1:stride:M-filter_size+1 % loop over original feature
%     new_ind = (i-1)/stride + 1;
%     new_feature{new_ind,1} = feature{i};
%     for j = i+1:i+filter_size-1
%         new_feature{new_ind,1} = max(new_feature{new_ind,1}, feature{j});
%     end
% end
% 
% % M = size(feature,1);
% % N = uint32(floor((M-filter_size)/stride) + 1);
% % 
% % for i=1:N
% %     new_feature{i,1} = feature{2*i-1};  % this should be modified
% %     for j=2*i:(2*i-1)+filter_size-1     % this should be modified
% %         new_feature{i,1} = max(new_feature{i}, feature{j});
% %     end
% % end