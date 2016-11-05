function imdb = setup_ap_THUMOS14(data_dir, isTest)
% Source images and classes
imdb.meta.sets = {'train','validation','test'};
imdb.images.set = [];

[gt, ~] = read_gt_THUMOS14(fullfile(data_dir, 'annotation'));

if isTest == 1
    load(fullfile(data_dir, 'test_set_meta.mat'));
    thumos14_videos = test_videos;
    clear test_videos;
    if ~exist(fullfile(data_dir, 'mats'), 'dir')
        read_frms_to_mat(fullfile(data_dir, 'TH14_test_set_frames'), gt, fullfile(data_dir, 'mats'));
    end
    imdb = addImageSet(imdb, fullfile(data_dir, 'mats'), 3) ;
else
    load(fullfile(data_dir, 'validation_set_meta', 'validation_set.mat'));
    thumos14_videos = validation_videos;
    clear validation_videos;
    if ~exist(fullfile(data_dir, 'mats'), 'dir')
        read_frms_to_mat(fullfile(data_dir, 'validation_frames'), gt, fullfile(data_dir, 'mats'));
    end
    imdb = addImageSet(imdb, fullfile(data_dir, 'mats'), 1) ;
end
imdb = mark_gt_THUMOS14(gt, thumos14_videos, imdb);

% Compress data types
imdb.images.set = uint8(imdb.images.set);

% -------------------------------------------------------------------------
function read_frms_to_mat(frms_dir, gt, mat_dir)
% -------------------------------------------------------------------------
% Input:
%         frms_dir: relative path to dataset frms dir (jpg files to read)
%         gt      : 1xN struct array contains GTs for N videos
%                   gt{k}.video_name -> int index
%                   gt{k}.start -> int frame number starts from 1
%                   gt{k}.end   -> int frame number starts from 1
%                   gt{k}.label
%                   gt{k}.label_name
%         mat_dir : relative path to dataset mat file dir (files to be stored)
% Output:
%         store the mat files to mat_dir

if ~exist(mat_dir)
    mkdir(mat_dir);
end

dir_list = clean_dir(frms_dir);

for i=1:length(dir_list)
    tmp = strsplit(dir_list{i}, '_');
    video_index = str2num(tmp{3});
    for k=1:length(gt)
        if video_index == gt{k}.video_name
            frms_list = clean_dir(fullfile(frms_dir,dir_list{i}));
            im = {};
            for j=1:length(frms_list)
                im{j} = imread(fullfile(frms_dir,dir_list{i},frms_list{j}));
            end
            save(fullfile(mat_dir,strcat(dir_list{i},'.mat')), 'im','-v7.3');
            break;
        end
    end
end

% -------------------------------------------------------------------------
function [imdb] = addImageSet(imdb, mat_dir, setCode)
% -------------------------------------------------------------------------
dir_list = clean_dir(mat_dir);
N = length(dir_list);

if setCode == 1
    % randome split THUMOS-14 "validation" set to train and val 
    % (THUMOS-14 does not provide any official train set for action detection)
    val_ratio = 0.1;
    rand_split_ind = randperm(N);
    numVal   = floor(val_ratio*N);
    valset_ind   = rand_split_ind(1:numVal);
    trainset_ind = rand_split_ind(numVal+1:N);
    imdb.images.set(valset_ind)   = setCode+1;    % setCode+1 is val set
    imdb.images.set(trainset_ind) = setCode;    % setCode is train set
elseif setCode == 3
    imdb.images.set(:) = setCode;
end

for i=1:N
    imdb.images.path{i} = fullfile(mat_dir, dir_list{i});
    tmp = strsplit(dir_list{i},'.');
    imdb.images.name{i} = tmp{1};
end

% % -------------------------------------------------------------------------
% function [imdb] = addImageSet(imdb, mat_dir, setCode)
% % -------------------------------------------------------------------------
% dir_list = clean_dir(mat_dir);
% for i=1:length(dir_list)
%     imdb.images.path{i} = fullfile(mat_dir, dir_list{i});
%     imdb.images.set(i) = setCode;
%     tmp = strsplit(dir_list{i},'.');
%     imdb.images.name{i} = tmp{1};
% end

% -------------------------------------------------------------------------
function imdb = mark_gt_THUMOS14(gt, thumos14_videos, imdb)
% -------------------------------------------------------------------------
% Input:
%         gt: ground truth data of test videos
%         thumos14_videos:  fps data of test videos
%         imdb: gt unmarked imdb
% Output:
%         imdb: gt marked imdb
%         imdb.images.labels{i}.gt_start_frames: Nx1 start_frames vector, N
%         is the number of labels of i-th video
%         imdb.images.labels{i}.gt_end_frames  : Nx1 end_frames vector, N
%         is the number of labels of i-th video

for i=1:length(imdb.images.name)
     filename = imdb.images.name{i};
     tmp = strsplit(filename, '_');
     file_index = str2num(tmp{3});

     % find index of gt matched to the file
     for j=1:length(gt)
         if gt{j}.video_name == file_index
             gt_index = j;
             gt_start_points = gt{j}.start;
             gt_end_points = gt{j}.end;
             break;
         end
     end

     % find fps of the video file
     fps = thumos14_videos(gt{gt_index}.video_name).frame_rate_FPS;
     gt_start_frames = uint32(gt_start_points*fps);
     gt_start_frames(find(gt_start_frames==0)) = 1;
     gt_end_frames = uint32(gt_end_points*fps);

     % filter out gt_end frame is not greater than num_frames
     ind = find(gt_end_frames<=thumos14_videos(file_index).number_of_frames);

     imdb.images.labels{i}.gt_start_frames = gt_start_frames(ind);
     imdb.images.labels{i}.gt_end_frames   = gt_end_frames(ind);
%      imdb.images.labels{i}.gt_start_frames = gt_start_frames;
%      imdb.images.labels{i}.gt_end_frames   = gt_end_frames;
     imdb.images.labels{i}.num_frames      = thumos14_videos(file_index).number_of_frames;
 end
