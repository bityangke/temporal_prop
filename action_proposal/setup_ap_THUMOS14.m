function imdb = setup_ap_THUMOS14(data_dir, isTest)
% Source images and classes
imdb.meta.sets = {'train','validation','test'} ;
imdb.images.set = [] ;

if isTest == 1
    [gt, ~] = read_gt_THUMOS14(fullfile(data_dir, 'annotation'));
    load(fullfile(data_dir, 'test_set_meta.mat'));
    thumos14_videos = test_videos;
    clear test_videos;
    read_frms_to_mat(fullfile(data_dir, 'frames_part'), fullfile(data_dir, 'mats'));
    [imdb] = addImageSet(imdb, fullfile(data_dir, 'mats'), 1) ; 
    imdb = mark_gt_THUMOS14(gt, thumos14_videos, imdb);
else
    [gt, ~] = read_gt_THUMOS14(fullfile(data_dir, 'annotation'));
    load(fullfile(data_dir, 'validation_set_meta', 'validation_set.mat'));
    thumos14_videos = validation_videos;
    clear validation_videos;
    read_frms_to_mat(fullfile(data_dir, 'frames'), fullfile(data_dir, 'mats'));
    [imdb] = addImageSet(imdb, fullfile(data_dir, 'mats'), 1) ; 
    imdb = mark_gt_THUMOS14(gt, thumos14_videos, imdb);
end

% Compress data types
imdb.images.set = uint8(imdb.images.set);

% -------------------------------------------------------------------------
function read_frms_to_mat(frms_dir, mat_dir)
% -------------------------------------------------------------------------
% Input:
%         frms_dir: relative path to dataset frms dir (jpg files to read)
%         mat_dir:  relative path to dataset mat file dir (files to be stored)
% Output:
%         store the mat files to mat_dir   
if ~exist(mat_dir)
    mkdir(mat_dir);
end

dir_list = clean_dir(frms_dir);
for i=1:length(dir_list)
    frms_list = clean_dir(fullfile(frms_dir,dir_list{i}));
    im = {};
    for j=1:length(frms_list)
        im{j} = imread(fullfile(frms_dir,dir_list{i},frms_list{j}));
%         im{j} = single(im{j});
%         im{j} = imresize(im{j}, net.meta.normalization.imageSize(1:2));
%         im{j} = im{j} - net.meta.normalization.averageImage;
    end
    save(fullfile(mat_dir,strcat(dir_list{i},'.mat')), 'im','-v7.3');
end

% -------------------------------------------------------------------------
function [imdb] = addImageSet(imdb, mat_dir, setCode) 
% -------------------------------------------------------------------------
dir_list = clean_dir(mat_dir);
for i=1:length(dir_list)
    imdb.images.path{i} = fullfile(mat_dir, dir_list{i});
    imdb.images.set(i) = setCode;
    tmp = strsplit(dir_list{i},'.');
    imdb.images.name{i} = tmp{1};
end
