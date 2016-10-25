function imdb = mark_gt_THUMOS14(gt, thumos14_videos, imdb)
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
     
     imdb.images.labels{i}.gt_start_frames = gt_start_frames;
     imdb.images.labels{i}.gt_end_frames   = gt_end_frames;
 end
