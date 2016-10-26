function [ts, tl] = extract_regression_target_values(labels, shrink_factor, starts, durations)
% Input:
%         labels: temporal ground truth label of actions
%         labels.gt_start_frames: Nx1 start_frames vector, N
%         is the number of labels of i-th video
%         labels.gt_end_frames  : Nx1 end_frames vector, N
%         is the number of labels of i-th video
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
% Output:
%         ts: Lx1 vector containing regression target ts = (Gs(i) - Ps(i))/Pl(i)
%             (L=sum of number of proposals per each GT)
%         tl: Lx1 vector containing regression target tl = log(Gl(i)/Pl(i))
%             (L=sum of number of proposals per each GT)

ts = [];
tl = [];

N = size(labels.gt_start_frames,1);
if (N ~= size(starts,1)) || (N ~= size(durations,1))
    fprintf('# of labels and # of proposals mismatch! \n');
    return;
end

% loop over GT labels
for i=1:N
    K = size(starts{i,1},2);
    Gs = double(ceil(double(labels.gt_start_frames(i))/double(shrink_factor)));
    Gl = double(ceil(double(labels.gt_end_frames(i) - labels.gt_start_frames(i) + 1)/double(shrink_factor)));       
    
    current_starts    = starts{i,1};
    current_durations = durations{i,1};
    % loop over proposals
    for j=1:K
        if ~isnan(current_starts(j))
            Ps = double(ceil(double(current_starts(j))/double(shrink_factor)));
            Pl = double(ceil(double(current_durations(j))/double(shrink_factor)));

            ts = [ts; double(Gs-Ps)/double(Pl)];
            tl = [tl; log(double(Gl)/double(Pl))];
        end
    end
end

% function [ts, tl] = extract_regression_target_values(labels, shrink_factor, starts, durations)
% % Input:
% %         labels: temporal ground truth label of actions
% %         labels.gt_start_frames: Nx1 start_frames vector, N
% %         is the number of labels of i-th video
% %         labels.gt_end_frames  : Nx1 end_frames vector, N
% %         is the number of labels of i-th video
% %         shrink_factor : temporal maxpooling scaling factor. ex) 2 means
% %         temporal max pooled with filter size 2 and stride 2
% %         starts: 1xN start frames (matched to GT) of the temporal proposal
% %         ends  : 1xN end frames (matched to GT of the temporal proposal
% % Output:
% %         ts: Lx1 vector containing regression target ts = (Gs(i) - Ps(i))/Pl(i)
% %         tl: Lx1 vector containing regression target tl = log(Gl(i)/Pl(i))
% 
% ts = [];
% tl = [];
% 
% N = size(labels.gt_start_frames,1);
% 
% if N ~= size(starts,2)
%     fprintf('# of labels and # of proposals mismatch! \n');
%     return;
% end
% 
% for i=1:N
%     if ~isnan(starts(i))
%         Gs = uint32(ceil(double(labels.gt_start_frames(i))/double(shrink_factor)));
%         Gl = uint32(ceil(double(labels.gt_end_frames(i) - labels.gt_start_frames(i) + 1)/double(shrink_factor)));
%         Ps = uint32(ceil(double(starts(i))/double(shrink_factor)));
%         Pl = uint32(ceil(double(durations(i))/double(shrink_factor)));
% 
% %         ts = [ts; double(Gs-Ps)/double(Pl)*ones(Pl,1)];
% %         tl = [tl; log(double(Gl)/double(Pl))*ones(Pl,1)];
%         ts = [ts; double(Gs-Ps)/double(Pl)];
%         tl = [tl; log(double(Gl)/double(Pl))];
%     end
% end