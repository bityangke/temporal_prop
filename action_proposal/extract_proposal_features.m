function [proposal_features, proposal_features_reshaped] = extract_proposal_features(org_feature, shrink_factor, starts, durations)
% Input:
%         org_feature   : Dx1 cell array containing temporal max pooled
%         spatial features
%
%         shrink_factor : temporal maxpooling scaling factor. ex) 2 means
%         temporal max pooled with filter size 2 and stride 2
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
%         proposal_features : Kx1 cell array containing features from
%         proposals - probabily not used in practice
%
%         proposal_features_reshaped: LxP vector which is reshaped from
%         proposal_features cell array, L is the number of features 
%         (L=sum of number of proposals per each GT), 
%         P is a feature dimension

N = size(starts,1);

proposal_features = {};
proposal_features_reshaped = [];

% loop over GT
for n=1:N
    K = size(starts{n,1},2);
    current_starts = starts{n,1};
    current_durations = durations{n,1};
    current_proposal_features = {};
    current_proposal_features_reshaped = [];
    % loop over proposals
    for i=1:K
        if ~isnan(current_starts(i))
            s = double(ceil(current_starts(i)/shrink_factor));
            e = s + double(ceil(current_durations(i)/shrink_factor)) - 1;
            current_proposal_features{i,1} = org_feature(s:e);
            tmp_feat = [];
            for j=s:e
                tmp_feat = [tmp_feat, org_feature{j}(:)'];
            end
            current_proposal_features_reshaped = [current_proposal_features_reshaped; tmp_feat];
        end
    end
    proposal_features = [proposal_features; current_proposal_features];
    proposal_features_reshaped = [proposal_features_reshaped; current_proposal_features_reshaped];
end

% function [proposal_features, proposal_features_reshaped] = extract_proposal_features(org_feature, shrink_factor, starts, durations)
% % Input:
% %         org_feature   : Nx1 cell array containing temporal max pooled
% %         spatial features
% %         shrink_factor : temporal maxpooling scaling factor. ex) 2 means
% %         temporal max pooled with filter size 2 and stride 2
% %         starts        : 1xK vector containing K temporal action starting points
% %         durations     : 1xK vector containing K temporal action durations
% % Output:
% %         proposal_features : Kx1 cell array containing features from proposals
% %         proposal_features_reshaped: LxP vector which is reshaped from
% %         proposal_features cell array, L is the number of features, P is a
% %         feature dimension
% 
% K = size(starts,2);
% 
% % new_starts    = uint32(floor(starts/shrink_factor)+1);
% % new_durations = uint32(floor(durations/shrink_factor));
% proposal_features = {};
% proposal_features_reshaped = [];
% 
% for i=1:K
%     if ~isnan(starts(i))
%         s = uint32(ceil(starts(i)/shrink_factor));
%         e = s + uint32(ceil(durations(i)/shrink_factor)) - 1;
%         proposal_features{i,1} = org_feature(s:e);
%         tmp_feat = [];
%         for j=s:e
%             tmp_feat = [tmp_feat, org_feature{j}(:)'];
%             %proposal_features_reshaped = [proposal_features_reshaped; org_feature{j}(:)']; % LxP feature
%         end
%         proposal_features_reshaped = [proposal_features_reshaped; tmp_feat];
%     end
% end
% 
% % if size(proposal_features,1) > 1
% %     if size(proposal_features{end,1},1) < size(proposal_features{end-1,1},1)
% %         proposal_features = proposal_features(1:end-1,1);
% %     end
% % end