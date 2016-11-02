function new_rois = transform_rois(rois, activation_width, activation_height)
% Input
%          rois     : Nx2 matrix, each row contains start frame index and
%                     durations of an action proposal
%          activation_width  : width of the activation map
%          activation_height : height of the activation map
% Output:
%          new_rois : 5xN single matrix, each column contains left-top corner
%                     (x,y) coordinate and width and height of an action
%                     proposal in a horizontally concatenated features
%                     feature dimension: WTxHx512x1, where W,H are width and
%                     height of the activation map and T is the number of
%                     frames of a input video

N = size(rois,1);
new_rois = zeros(1+4, N);

new_rois(1,:) = 1;  % batch numbers are always 1 because we use only 1 batch per video
new_rois(3,:) = 1;  % y coordinates are always 1
new_rois(5,:) = activation_height;  % h are always H
new_rois(2,:) = rois(:,1)';  % x coordinates
new_rois(4,:) = rois(:,2)';      % w coordinates
new_rois = single(new_rois);

% function new_rois = transform_rois(rois, activation_width, activation_height)
% % Input
% %          rois     : Nx2 matrix, each row contains start frame index and
% %                     durations of an action proposal
% %          activation_width  : width of the activation map
% %          activation_height : height of the activation map
% % Output:
% %          new_rois : 5xN single matrix, each column contains left-top corner
% %                     (x,y) coordinate and width and height of an action
% %                     proposal in a horizontally concatenated features
% %                     feature dimension: WTxHx512x1, where W,H are width and
% %                     height of the activation map and T is the number of
% %                     frames of a input video
% 
% N = size(rois,1);
% new_rois = zeros(1+4, N);
% 
% new_rois(1,:) = 1;  % batch numbers are always 1 because we use only 1 batch per video
% new_rois(3,:) = 1;  % y coordinates are always 1
% new_rois(5,:) = activation_height;  % h are always H
% new_rois(2,:) = activation_width*(rois(:,1)-1)'+1;
% new_rois(4,:) = activation_width*(rois(:,2))';
% new_rois = single(new_rois);