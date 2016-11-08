function new_rois = transform_rois2(rois, activation_height)
% Input
%          rois     : Nx2 matrix, each row contains start frame index and
%                     durations of an action proposal
%          activation_height : height of the activation map
% Output:
%          new_rois : 5xN single matrix, each column contains left-top corner
%                     (x1,y1) coordinate and bottom-right corner (x2,y2) coordinate
%                     feature dimension: WTxHx512x1, where W,H are width and
%                     height of the activation map and T is the number of
%                     frames of a input video

N = size(rois,1);
new_rois = zeros(1+4, N);

new_rois(1,:) = 1;  % batch numbers are always 1 because we use only 1 batch per video
new_rois(3,:) = 1;  % y1 coordinates are always 1
new_rois(2,:) = rois(:,1)';  % x1 coordinates
new_rois(4,:) = rois(:,1)' + rois(:,2)' - 1;      % x2 coordinates
new_rois(5,:) = activation_height;  % y2 are always H
new_rois = single(new_rois);
