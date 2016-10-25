function [tp, fn] = evaluate_temporal_proposal(labels, starts, durations, thIoU)
% Input:
%         labels: temporal label of actions
%         labels.gt_start_frames: Nx1 start_frames vector, N
%         is the number of labels of i-th video
%         labels.gt_end_frames  : Nx1 end_frames vector, N
%         is the number of labels of i-th video
%         starts: 1xN start frames of the temporal proposal
%         ends  : 1xN end frames of the temporal proposal
%         thIoU : threshold of intersection over union
% Output:
%         tp: True positive predictions
%         fn: False negative predictions

tp = 0;
fn = 0;

proposals = [starts', starts'+durations'-1]; % Nx2 proposal matrix
% loop over labels
for j=1:length(labels.gt_start_frames)
    current_gt = [labels.gt_start_frames(j), labels.gt_end_frames(j)];
    current_IoU = [];
    % loop over proposals
    for i=1:length(proposals)
        current_IoU(i) = calculateIoU(proposals(i,:), current_gt);
    end
    [maxIoU, maxInd] = max(current_IoU);
    if maxIoU >= thIoU
        tp = tp + 1;
        proposals = proposals([1:maxInd-1,maxInd+1:size(proposals,1)], :);
    else
        fn = fn + 1; 
    end
end

if tp + fn ~= length(labels.gt_start_frames)
    fprintf('mismatch between tp + fn and # of GTs! tp+fn=%d, # of GTs=%d\n', tp+fn, length(labels.gt_start_frames));
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