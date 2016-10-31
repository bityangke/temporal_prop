function [targets] = extract_regression_target_values2(labels, starts, durations, proposal_labels)
% Input:
%         labels: temporal ground truth label of actions
%         labels.gt_start_frames: Nx1 start_frames vector, N
%         is the number of labels of i-th video
%         labels.gt_end_frames  : Nx1 end_frames vector, N
%         is the number of labels of i-th video

%         starts        : Nx1 cell array of assigned proposals to GT of each
%         index - ex) matched_starts(1) is the starting point of all proposals assigned to GT(1)
%         N is the number of GTs
%
%         durations     : Nx1 cell array of assigned proposals to GT of each
%         index - ex) matched_durations(1) is the duration of all proposals assigned to GT(1)
%         N is the number of GTs

%         proposal_labels : Nx1 cell array containing array of proposal labels - 
%                           the information whether each proposal is an action or a background
%
% Output:
%         targets : LNx2 matrix containing regression target pair (ts, tl) 
%                   where, ts = (Gs(i) - Ps(i))/Pl(i)
%                          tl = log(Gl(i)/Pl(i))
%             (L=sum of number of proposals per each GT, N=number of GTs)

N = size(labels.gt_start_frames,1);
if (N ~= size(starts,1)) || (N ~= size(durations,1))
    fprintf('# of labels and # of proposals mismatch! \n');
    return;
end

LN = 0;
for i=1:N
    LN = LN + size(starts{i,1},1);
end

targets = zeros(LN,2);

% loop over GT labels
for i=1:N
    K = size(starts{i,1},1);
    Gs = double(labels.gt_start_frames(i));
    Gl = double(labels.gt_end_frames(i) - labels.gt_start_frames(i) + 1);       
    
    current_starts    = starts{i,1};
    current_durations = durations{i,1};
    current_proposal_labels = proposal_labels{i,1};
    % loop over proposals
    for j=1:K
        if ~isnan(current_starts(j)) && (current_proposal_labels(j) == 1)
            Ps = double(current_starts(j));
            Pl = double(current_durations(j));
            targets((i-1)*K+j, :) = [double(Gs-Ps)/double(Pl), log(double(Gl)/double(Pl))];
        end
    end
end
