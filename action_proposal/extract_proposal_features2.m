function [proposal_features] = extract_proposal_features2(org_feature, starts, durations)
% Input:
%         org_feature   : WHxNTxC single array containing temporal max pooled
%         spatial features
%
%         starts        : Nx1 cell array of assigned proposals to GT of each
%         index - ex) matched_starts(1) is the starting point of all proposals assigned to GT(1)
%         N is the number of GTs
%
%         durations     : Nx1 cell array of assigned proposals to GT of each
%         index - ex) matched_durations(1) is the duration of all proposals assigned to GT(1)
%         N is the number of GTs
%
%
% Output:
%         proposal_features: LxP vector which is reshaped from
%         proposal_features cell array, L is the number of features 
%         (L=sum of number of proposals per each GT), 
%         P is a feature dimension

N = size(starts,1);

proposal_features = [];

% loop over GT
for n=1:N
    K = size(starts{n,1},2);
    current_starts = starts{n,1};
    current_durations = durations{n,1};
    current_proposal_features = [];
    % loop over proposals
    for i=1:K
        if ~isnan(current_starts(i))
            s = current_starts(i);
            e = s + current_durations(i) - 1;
%             current_proposal_features{i,1} = org_feature(:,s:e,:);
            tmp = org_feature(:,s:e,:);
            temp_pooled = temporal_max_pooling(tmp, 1);
            current_proposal_features(i,:) = reshape(temp_pooled, 1,numel(temp_pooled));
%             current_proposal_features(i,:) = reshape(org_feature(:,s:e,:), 1,numel(org_feature(:,s:e,:)));
        end
    end
    proposal_features = [proposal_features; current_proposal_features];
end
