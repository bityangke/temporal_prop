function [new_imdb] = load_partial_imdb_THUMOS(imdb, featDir)
% Input : 
%         imdb    : original imdb which contains set, images_path,
%         video_name, and labels(start_frame, end_frame, num_frames)
%         featDir : CNN feature directory path
% Output: imdb with CNN feature path informations

new_imdb.meta = imdb.meta;
list_1D = dir(strcat(featDir,'/*.mat'));
existing_feat = zeros(size(list_1D,2),1);

for i=1:size(list_1D,1)
    tmp = strsplit(list_1D(i).name, '_');
    existing_feat(i,1) = str2double(tmp{5});
end

existing_feat = sort(existing_feat);

% new_imdb.images.set    = imdb.images.set(existing_feat);
new_imdb.images.name   = imdb.images.name(existing_feat);
new_imdb.images.path   = imdb.images.path(existing_feat);
new_imdb.images.labels = imdb.images.labels(existing_feat);
new_imdb = splitTrainVal(new_imdb, 0.2);

for i=1:size(existing_feat,1)
    new_imdb.images.feature_path{i} = fullfile(featDir, sprintf('imagenet-vgg_relu5_on_THUMOS14val_%d_1D.mat',i));
end

new_imdb = split_video_features(new_imdb,fullfile(featDir,'split'));

% -------------------------------------------------------------------------
function [imdb] = splitTrainVal(imdb, val_ratio)
% -------------------------------------------------------------------------
N = length(imdb.images.name);

% randome split THUMOS-14 "validation" set to train and val 
% (THUMOS-14 does not provide any official train set for action detection)
rand_split_ind = randperm(N);
numVal   = floor(val_ratio*N);
valset_ind   = rand_split_ind(1:numVal);
trainset_ind = rand_split_ind(numVal+1:N);
imdb.images.set(valset_ind)   = 2;    % setCode+1 is val set
imdb.images.set(trainset_ind) = 1;    % setCode is train set

% -------------------------------------------------------------------------
function [new_imdb] = split_video_features(imdb, splitFeatDir)
% -------------------------------------------------------------------------
% Input : 
%         imdb    : original imdb which contains set, images_path,
%         video_name, and labels(start_frame, end_frame, num_frames)
%         splitFeatDir : dir path for splitted feature files to be stored
% Output: 
%         new_imdb: imdb with video features. Each video contain only one action

if ~exist(splitFeatDir, 'dir')
    mkdir(splitFeatDir);
    save_flag = 1;
else
    fprintf('\nsplitted feature files already exist!\n');
    save_flag = 0;
end

new_imdb.meta = imdb.meta;

count = 1;
% loop over all videos
for i=1:size(imdb.images.labels,2)
% for i=20:20%size(imdb.images.labels,2)
    fprintf('Splitting %d-th video... out of %d total videos\n', i, size(imdb.images.labels,2));
    N = size(imdb.images.labels{i}.gt_start_frames,1);
    load(imdb.images.feature_path{i});   % load features for current frame
    tmp = strsplit(imdb.images.feature_path{i}, '/');
    filename = strsplit(tmp{end}, '.');
    filename = filename{1};
    % loop over all GTs, split feature
    for j=1:N
        s = imdb.images.labels{i}.gt_start_frames(j);
        e = imdb.images.labels{i}.gt_end_frames(j);
        num_frames = imdb.images.labels{i}.num_frames;

        if j<N % not last GT
            next_s = imdb.images.labels{i}.gt_start_frames(j+1);
            next_e = imdb.images.labels{i}.gt_end_frames(j+1);
        else   % for last GT 
            next_s = num_frames;
            next_e = num_frames;
        end
        
        if j==1 % first GT
            prev_s = 1;
            prev_e = 1;
        else    % not first GT
            prev_s = imdb.images.labels{i}.gt_start_frames(j-1);
            prev_e = imdb.images.labels{i}.gt_end_frames(j-1);
        end
        
        if (s > prev_e) && (e < next_s)
            frame_s = prev_e+1;
            frame_e = next_s-1;
            num_frames = frame_e - frame_s + 1;
            new_imdb.images.labels{count}.gt_start_frames = s-prev_e;
            new_imdb.images.labels{count}.gt_end_frames   = e-prev_e;
        elseif (s <= prev_e) && (e < next_s)
            frame_s = s;
            frame_e = next_s-1;
            num_frames = frame_e - frame_s + 1;
            new_imdb.images.labels{count}.gt_start_frames = max(s-prev_e, 1);
            new_imdb.images.labels{count}.gt_end_frames   = e-prev_e;
        elseif (s > prev_e) && (e >= next_s)
            frame_s = prev_e+1;
            frame_e = e;
            num_frames = frame_e - frame_s + 1;
            new_imdb.images.labels{count}.gt_start_frames = s-prev_e;
            new_imdb.images.labels{count}.gt_end_frames   = e-prev_e;
        elseif (s <= prev_e) && (e >= next_s)
            frame_s = s;
            frame_e = e;
            num_frames = frame_e - frame_s + 1;
            new_imdb.images.labels{count}.gt_start_frames = 1;
            new_imdb.images.labels{count}.gt_end_frames   = num_frames;
        else
            fprintf('exception! Video splitting is not completed for video=%d, GT=%d \n', i, j);
        end
            
        % exception handling: for a long clip - more than 5000 frames
        num_frames_th = 5000;
        if num_frames > num_frames_th
            old_s = imdb.images.labels{i}.gt_start_frames(j);
            old_e = imdb.images.labels{i}.gt_end_frames(j);
            gt_duration = old_e - old_s + 1;
            if 3*gt_duration <= num_frames_th
                fprintf('3*gt_duration <= num_frames_th\n');
                if old_s - gt_duration < 1
                    frame_s_new = frame_s;
                    gt_s = old_s;
                else
                    frame_s_new = old_s - gt_duration;
                    gt_s = gt_duration+1;
                end
%                 frames_s_new = max(old_s - gt_duration, gt_duration - old_s + 1);
                if old_e + gt_duration >= next_s
                    frame_e_new = next_s - 1;
                else
                    frame_e_new = old_e + gt_duration;
                end
%                 frames_e_new = min (next_s - 1, old_e+gt_duration);
                current_GT_1D_feat = oneD_converted_feat(:,frame_s_new:frame_e_new,:);
                new_imdb.images.labels{count}.num_frames = frame_e_new - frame_s_new +1;
                new_imdb.images.labels{count}.gt_start_frames = gt_s;
                new_imdb.images.labels{count}.gt_end_frames   = new_imdb.images.labels{count}.gt_start_frames + gt_duration-1;
            else
                fprintf('3*gt_duration > num_frames_th\n');
                margin = floor(double(num_frames_th - gt_duration)/2.0);
                if old_s - frame_s < margin
                    frame_s_new = frame_s;
                    frame_e_new = frame_s + num_frames_th - 1;
                    gt_s = old_s - frame_s + 1;
                else
                    frame_e_new = frame_e;
                    frame_s_new = frame_e_new - num_frames_th + 1;
                    gt_s = old_s - frame_s + 1;
                end
                current_GT_1D_feat = oneD_converted_feat(:,frame_s_new:frame_e_new,:);
                new_imdb.images.labels{count}.num_frames = frame_e_new - frame_s_new +1;
                new_imdb.images.labels{count}.gt_start_frames = gt_s;
                new_imdb.images.labels{count}.gt_end_frames   = new_imdb.images.labels{count}.gt_start_frames + gt_duration-1;
            end
%             if 3*gt_duration <= num_frames_th
%                 fprintf('3*gt_duration <= num_frames_th\n');
%                 current_GT_1D_feat = current_GT_1D_feat(:,old_s-gt_duration:old_e+gt_duration,:);
%                 new_imdb.images.labels{count}.num_frames = 3*gt_duration;
%                 new_imdb.images.labels{count}.gt_start_frames = gt_duration+1;
%                 new_imdb.images.labels{count}.gt_end_frames   = gt_duration*2;
%             else
%                 fprintf('3*gt_duration > num_frames_th\n');
%                 margin = floor(double(num_frames_th - gt_duration)/2.0);
%                 frame_s 
%                 current_GT_1D_feat = current_GT_1D_feat(:,old_s-margin:old_e+margin,:);
%                 new_imdb.images.labels{count}.num_frames = gt_duration+2*margin;
%                 new_imdb.images.labels{count}.gt_start_frames = margin+1;
%                 new_imdb.images.labels{count}.gt_end_frames   = margin+gt_duration;
%             end
        else
            current_GT_1D_feat = oneD_converted_feat(:,frame_s:frame_e,:);
            new_imdb.images.labels{count}.num_frames     = num_frames;
        end

        new_imdb.images.set(count)  = imdb.images.set(i);
        new_imdb.images.path{count} = imdb.images.path{i};
        new_imdb.images.name{count} = imdb.images.name{i};
        new_imdb.images.feature_path{count} = fullfile(splitFeatDir, sprintf('%s_gt_%d.mat' , filename, j));
        
        if save_flag == 1
            save(new_imdb.images.feature_path{count}, 'current_GT_1D_feat','-v7.3');
        end
        
        count = count + 1;
    end
end