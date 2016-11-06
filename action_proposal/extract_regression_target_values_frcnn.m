function [targets] = extract_regression_target_values_frcnn(labels, starts, durations, proposal_labels, activation_height)
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
%         activation_height : height of the activation map
% Output:
%         targets : LNx4 matrix containing regression target pair (ts, tl) 
%                   where, tx = (Gx(i) - Px(i))/Pw(i)
%                          ty = (Gy(i) - Py(i))/Pw(i)
%                          tw = log(Gw(i)/Pw(i))
%                          th = log(Gh(i)/Ph(i))
%             (L=sum of number of proposals per each GT, N=number of GTs)

N = size(labels.gt_start_frames,1);
if (N ~= size(starts,1)) || (N ~= size(durations,1))
    fprintf('# of labels and # of proposals mismatch! \n');
    return;
end

LN = 0;
for i=1:N
    LN = LN + size(starts{i,1},1);
    K(i) = size(starts{i,1},1);
end

targets = zeros(LN,4);

% loop over GT labels
for i=1:N
    Gw = double(labels.gt_end_frames(i) - labels.gt_start_frames(i) + 1);       
    Gh = activation_height;
    GCx = double(labels.gt_start_frames(i)) + 0.5*Gw;
    GCy = 1.0 + 0.5*Gh;
  
%     Gx = double(labels.gt_start_frames(i));
%     Gy = 1;
%     Gw = double(labels.gt_end_frames(i) - labels.gt_start_frames(i) + 1);       
%     Gh = activation_height;
    
    current_starts    = starts{i,1};
    current_durations = durations{i,1};
    current_proposal_labels = proposal_labels{i,1};
    % loop over proposals
    for j=1:K(i)
        if ~isnan(current_starts(j)) && (current_proposal_labels(j) == 1)
            Pw = double(current_durations(j));
            Ph = activation_height;
            PCx = double(current_starts(j)) + 0.5*Pw;
            PCy = 1.0 + 0.5*Ph;
%             Px = double(current_starts(j));
%             Py = 1;
%             Pw = double(current_durations(j));
%             Ph = activation_height;
            if i == 1
                offset = 0;
            else
                offset = sum(K(1:i-1));
            end
            targets(offset+j, :) = [double(GCx-PCx)/double(Pw), double(GCy-PCy)/double(Ph), log(double(Gw)/double(Pw)), log(double(Gh)/double(Ph))];
%             targets(offset+j, :) = [double(Gx-Px)/double(Pw), double(Gy-Py)/double(Ph), log(double(Gw)/double(Pw)), log(double(Gh)/double(Ph))];
        end
    end
end


% function [targets] = extract_regression_target_values_frcnn(labels, starts, durations, proposal_labels, activation_width, activation_height)
% % Input:
% %         labels: temporal ground truth label of actions
% %         labels.gt_start_frames: Nx1 start_frames vector, N
% %         is the number of labels of i-th video
% %         labels.gt_end_frames  : Nx1 end_frames vector, N
% %         is the number of labels of i-th video
% 
% %         starts        : Nx1 cell array of assigned proposals to GT of each
% %         index - ex) matched_starts(1) is the starting point of all proposals assigned to GT(1)
% %         N is the number of GTs
% %
% %         durations     : Nx1 cell array of assigned proposals to GT of each
% %         index - ex) matched_durations(1) is the duration of all proposals assigned to GT(1)
% %         N is the number of GTs
% 
% %         proposal_labels : Nx1 cell array containing array of proposal labels - 
% %                           the information whether each proposal is an action or a background
% %
% %          activation_width  : width of the activation map
% %          activation_height : height of the activation map
% % Output:
% %         targets : LNx4 matrix containing regression target pair (ts, tl) 
% %                   where, tx = (Gx(i) - Px(i))/Pw(i)
% %                          ty = (Gy(i) - Py(i))/Pw(i)
% %                          tw = log(Gw(i)/Pw(i))
% %                          th = log(Gh(i)/Ph(i))
% %             (L=sum of number of proposals per each GT, N=number of GTs)
% 
% N = size(labels.gt_start_frames,1);
% if (N ~= size(starts,1)) || (N ~= size(durations,1))
%     fprintf('# of labels and # of proposals mismatch! \n');
%     return;
% end
% 
% LN = 0;
% for i=1:N
%     LN = LN + size(starts{i,1},1);
%     K(i) = size(starts{i,1},1);
% end
% 
% targets = zeros(LN,4);
% 
% % loop over GT labels
% for i=1:N
% %     K = size(starts{i,1},1);
%     Gx = activation_width*double(labels.gt_start_frames(i)-1)+1;
%     Gy = 1;
%     Gw = activation_width*double(labels.gt_end_frames(i) - labels.gt_start_frames(i) + 1);       
%     Gh = activation_height;
%     
%     current_starts    = starts{i,1};
%     current_durations = durations{i,1};
%     current_proposal_labels = proposal_labels{i,1};
%     % loop over proposals
%     for j=1:K(i)
%         if ~isnan(current_starts(j)) && (current_proposal_labels(j) == 1)
%             Px = activation_width*double(current_starts(j)-1)+1;
%             Py = 1;
%             Pw = activation_width*double(current_durations(j));
%             Ph = activation_height;
%             if i == 1
%                 offset = 0;
%             else
%                 offset = sum(K(1:i-1));
%             end
%             targets(offset+j, :) = [double(Gx-Px)/double(Pw), double(Gy-Py)/double(Ph), log(double(Gw)/double(Pw)), log(double(Gh)/double(Ph))];
%         end
%     end
% end
