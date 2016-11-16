function [proposals, targets] = get_training_proposal(labels, starts, durations, numPositive, height)
% Input:
%         labels: temporal ground truth label of actions
%         labels.gt_start_frames: Nx1 start_frames vector, N
%         is the number of labels of i-th video
%         labels.gt_end_frames  : Nx1 end_frames vector, N
%         is the number of labels of i-th video
%         starts: 1xK start frames of the temporal proposal
%         ends  : 1xK end frames of the temporal proposal
%         numPositive: number of positive examples per video
%         height : height of the activation map
% Output:
%         proposals.rois   : Px2 matrix containing N temporal (start, duration) pairs
%                            with multiple grid_sizes and strides
%         proposals.labels : Px1 vector containing label for each pair - 0
%                            is negative (background), 1 is action
%         targets          : Px2 target coordinate matrix containing
%                            regression target values

thIoU_pos = 0.5;
thIoU_neg = 0.3;
numTotalSamples = 256;
numNegative = numTotalSamples - numPositive;

% shuffle candidate proposals
numCandidates = size(starts,2);
% load('seed_20161030.mat');
% rng(s);
randInd = randperm(numCandidates);

candidates = [starts(randInd)', starts(randInd)'+durations(randInd)'-1]; % Nx2 proposal matrix

N = size(labels.gt_start_frames, 1);    % Nobj
K = size(candidates, 1);                % Nprop

matched_starts    = cell(N,1);
matched_ends      = cell(N,1);
matched_labels    = cell(N,1);
matched_piou      = cell(N,1);
matched_pgtidx    = cell(N,1);

numPos            = zeros(N,1);
numNeg            = zeros(N,1);

% loop over candidates
for i=1:K
    % loop over GT labels: find the maximum overlap GT for the current proposal
    current_candidate_IoU = [];
    for j=1:N
        current_gt = [ labels.gt_start_frames(j), labels.gt_end_frames(j) ];
        current_candidate_IoU(j) = calculateIoU(candidates(i,:), current_gt);
    end
    [maxIoU, maxInd] = max(current_candidate_IoU);

    %%% positive sampling
    if maxIoU > thIoU_pos && numPos(maxInd,1) < numPositive
        matched_starts   {maxInd,1} = [matched_starts{maxInd,1}; candidates(i,1)];
        matched_ends{maxInd,1} = [matched_ends{maxInd,1}; candidates(i,2)];
        matched_labels   {maxInd,1} = [matched_labels{maxInd,1}; 1];
        numPos(maxInd,1) = numPos(maxInd,1)  + 1;
        matched_piou{maxInd,1}      = [matched_piou{maxInd,1}; maxIoU];
        matched_pgtidx{maxInd,1}    = [matched_pgtidx{maxInd,1}; maxInd];
    end
    
    %%% negative sampling
    %%%% maybe we should sample negative examples randomly rather than
    %%%% sampling maximum IoU negative examples
    if sum(current_candidate_IoU < thIoU_neg) == N && numNeg(maxInd,1) < numNegative
        matched_starts   {maxInd,1} = [matched_starts{maxInd,1}; candidates(i,1)];
        matched_ends{maxInd,1} = [matched_ends{maxInd,1}; candidates(i,2)];
        matched_labels   {maxInd,1} = [matched_labels{maxInd,1}; 0];
        numNeg(maxInd,1) = numNeg(maxInd,1) + 1;
        matched_piou{maxInd,1}      = [matched_piou{maxInd,1}; maxIoU];
        matched_pgtidx{maxInd,1}    = [matched_pgtidx{maxInd,1}; maxInd];
    end
end

% check the number of positive/negative samples
for i=1:size(numNeg,1);
    if numNeg(i) <  numNegative
        fprintf('Not enough negative samples for %d-th GT: %d/%d\n', i, numNeg(i), numNegative);
    end
    if numPos(i) < numPositive
        fprintf('Not enough positive samples for %d-th GT: %d/%d\n', i, numPos(i), numPositive);
    end
    if numNeg(i) + numPos(i) ~= numTotalSamples
        fprintf('Number of total samples = %d, it should be =%d\n', numNeg(i) + numPos(i), numTotalSamples);
    end
end

N_prop = size(cell2mat(matched_starts),1);
proposals.rois   = [cell2mat(matched_starts), ones(N_prop,1), cell2mat(matched_ends), height*ones(N_prop,1)]; %[Xmin, Ymin, Xmax, Ymax];
proposals.labels = cell2mat(matched_labels);
proposals.numPos = sum(numPos);
proposals.numNeg = sum(numNeg);
proposals.piou   = cell2mat(matched_piou);
proposals.pgtidx = cell2mat(matched_pgtidx);

targets = extract_regression_target_values_frcnn(labels, matched_starts, matched_ends, matched_labels, height);

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