function [matched_starts, matched_durations] = match_gt_proposal(labels, starts, durations)
% Input:
%         labels: temporal ground truth label of actions
%         labels.gt_start_frames: Nx1 start_frames vector, N
%         is the number of labels of i-th video
%         labels.gt_end_frames  : Nx1 end_frames vector, N
%         is the number of labels of i-th video
%         starts: 1xK start frames of the temporal proposal
%         ends  : 1xK end frames of the temporal proposal
% Output:
%         matched_starts   : Nx1 vector of assigned proposals to GT of each
%         index - ex) matched_starts(1) is the starting point of the proposal assigned to GT(1)
%         matched_durations: Nx1 vector of assigned proposals to GT of each
%         index - ex) matched_durations(1) is the duration of the proposal assigned to GT(1)

proposals = [starts', starts'+durations'-1]; % Nx2 proposal matrix

% loop over GT labels
for j=1:length(labels.gt_start_frames)
    current_gt = [labels.gt_start_frames(j), labels.gt_end_frames(j)];
    current_IoU = [];
    % loop over proposals
    for i=1:length(proposals)
        current_IoU(i) = calculateIoU(proposals(i,:), current_gt);
    end
    [maxIoU, maxInd] = max(current_IoU);
    if maxIoU > 0.1
        matched_starts(j)    = proposals(maxInd,1);
        matched_durations(j) = proposals(maxInd,2) - matched_starts(j) + 1;
        proposals = proposals([1:maxInd-1,maxInd+1:size(proposals,1)], :);
    else
        matched_starts(j)    = NaN; % not assigned GT
        matched_durations(j) = NaN; % not assigned GT
    end
end

% -------------------------------------------------------------------------
function IoU = calculateIoU(range_window, range_GT_segment)
% -------------------------------------------------------------------------
% Input:
%        range_window: integer range vector [start end]
%        range_GT_segment: integer range vector [start end]
% Output: 
%        IoU: intersection over union value

inter_range = range_intersection(range_window, range_GT_segment);
union_range = range_union(range_window, range_GT_segment);

if ~isempty(inter_range)
    inter_length = range_length(inter_range);
else
    inter_length = 0;
end
union_length = range_length(union_range);

IoU = double(inter_length)/double(union_length);

% -------------------------------------------------------------------------
function output_length = range_length(input_range)
% -------------------------------------------------------------------------
output_length = input_range(2) - input_range(1) + 1;