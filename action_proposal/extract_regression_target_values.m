function [ts, tl] = extract_regression_target_values(labels, shrink_factor, starts, durations)
% Input:
%         labels: temporal ground truth label of actions
%         labels.gt_start_frames: Nx1 start_frames vector, N
%         is the number of labels of i-th video
%         labels.gt_end_frames  : Nx1 end_frames vector, N
%         is the number of labels of i-th video
%         shrink_factor : temporal maxpooling scaling factor. ex) 2 means
%         temporal max pooled with filter size 2 and stride 2
%         starts: 1xN start frames (matched to GT) of the temporal proposal
%         ends  : 1xN end frames (matched to GT of the temporal proposal
% Output:
%         ts: Lx1 vector containing regression target ts = (Gs(i) - Ps(i))/Pl(i)
%         tl: Lx1 vector containing regression target tl = log(Gl(i)/Pl(i))

ts = [];
tl = [];

N = size(labels.gt_start_frames,1);

if N ~= size(starts,2)
    fprintf('# of labels and # of proposals mismatch! \n');
    return;
end

for i=1:N
    if ~isnan(starts(i))
        Gs = uint32(ceil(double(labels.gt_start_frames(i))/double(shrink_factor)));
        Gl = uint32(ceil(double(labels.gt_end_frames(i) - labels.gt_start_frames(i) + 1)/double(shrink_factor)));
        Ps = uint32(ceil(double(starts(i))/double(shrink_factor)));
        Pl = uint32(ceil(double(durations(i))/double(shrink_factor)));

%         ts = [ts; double(Gs-Ps)/double(Pl)*ones(Pl,1)];
%         tl = [tl; log(double(Gl)/double(Pl))*ones(Pl,1)];
        ts = [ts; double(Gs-Ps)/double(Pl)];
        tl = [tl; log(double(Gl)/double(Pl))];
    end
end