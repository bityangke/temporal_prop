function converted_feature = convert_2dfeat_to_1dfeat(feature)
% converts 2D frame-wise feature maps to 1D video-wise feature maps
%
% Input:
%         feature: 2D feature of dimension HxWxCxT
%                  H: height, W: width, C: channels, T: frames
% Output:
%         converted_feature: 1D feature of dimension HW x T x C

T = size(feature,1);
[H,W,C] = size(feature{1});

numGrid = 7;
gridSize = H/numGrid;

converted_feature = zeros(H*W, T, C);

for i=1:T
    frame = feature{i};
    dividedFrame = mat2cell(frame, gridSize*ones(numGrid,1), gridSize*ones(1,numGrid), ones(C,1));
    reshaped = cellfun(@(x) reshape(x, [4 1 1]), dividedFrame,'UniformOutput', false);
    converted_feature(:,i,:) = reshape(cell2mat(reshaped), [H*W C]);

%     ch_stacked_feature = [];    % 196 x C dimension
%     for j=1:C
%         dividedFrame = mat2cell(frame(:,:,j), [gridSize*ones(numGrid,1)], [gridSize*ones(1,numGrid)]);
%         reshaped = cellfun(@(x) reshape(x, [4 1 1]), dividedFrame,'UniformOutput', false);
%         frame_vector=reshape(cell2mat(reshaped), [H*W 1] );
% %         frame_vector = [];
% %         for k=1:49
% %             frame_vector = [frame_vector; reshape(dividedFrame{k}, [4 1])];
% %         end
% %         ch_stacked_feature = [ch_stacked_feature frame_vector]; 
%         converted_feature(:,i,j) = frame_vector;
%     end
%     converted_feature{i} = ch_stacked_feature;
end