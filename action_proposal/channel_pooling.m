function new_feature = channel_pooling(feature)
% Input:
%         feature       : Mx1 cell array containing spatial features
%                         feature{i} is one spatial feature map of any ConvNet
%                         layers: ex) conv5, pool5, etc
% Output:
%         new_feature   : Nx1 cell array containing temporal max pooled
%                         spatial features. (N=int((M-filter_size)/stride)+1) < M

C = size(feature{1},3);
pool_factor = 8;

for i=1:length(feature)
    new_feature{i,1} = feature{i}(:,:,1:pool_factor:C);
end
