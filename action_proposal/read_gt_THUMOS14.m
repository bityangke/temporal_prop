function [gt, label_name_list] = read_gt_THUMOS14(gt_path)
    % read annotation files
    annotation_file_list = dir(strcat(gt_path,'/*.txt'));
    j=1;
    for i=1:length(annotation_file_list)
        tmp = strsplit(annotation_file_list(i).name, '_');
        if ~strcmp(tmp{1}, 'Ambiguous')
            label_name_list{j} = tmp{1};
            j = j+1;
        end
    end
    j=1;
    for i=1:length(annotation_file_list)
%         if ~strcmp(annotation_file_list(i).name, 'Ambiguous_val.txt')
        tmp = strsplit(annotation_file_list(i).name, '_');
        if ~strcmp(tmp{1}, 'Ambiguous')
            fid = fopen(fullfile(gt_path,annotation_file_list(i).name));
            while 1
                tline = fgetl(fid);
                if ~ischar(tline), break, end
                tmp = strsplit(tline, ' ');
                tmp_name = strsplit(tmp{1},'_');
                raw_gt(j,1) = str2num(tmp_name{3}); % video_number
                raw_gt(j,2) = str2num(tmp{2});  % start frame
                raw_gt(j,3) = str2num(tmp{3});  % end frame
                raw_gt(j,4) = i-1;               % action label
                j = j+1;
            end
            fclose(fid);
        end
    end
    new_gt = sortrows(raw_gt);
    % select the raw data containing the temporal annotations
    effective_videos = unique(new_gt(:,1));
    for i=1:length(effective_videos)
        indices = find(new_gt(:,1) == effective_videos(i));
        gt{i}.video_name = effective_videos(i);
        gt{i}.start = new_gt(indices,2);
        gt{i}.end = new_gt(indices,3);
        gt{i}.label = new_gt(indices,4); 
        for j=1:length(indices)
            gt{i}.label_name{j} = label_name_list{new_gt(indices(j),4)};
        end        
    end
    gt = remove_redundant_label_gt(gt);
end

function [new_gt] = remove_redundant_label_gt(gt)
    N = size(gt,2);
    for j=1:N
        new_gt{j}.video_name = gt{j}.video_name;
        M = size(gt{j}.start,1);
        count = 0;    
        for i=1:M
            s = gt{j}.start(i);
            e = gt{j}.end(i); 
            if i < M
                if (s ~= gt{j}.start(i+1)) || (e ~= gt{j}.end(i+1))
                    count = count + 1;
                    new_gt{j}.start(count,1) = s;
                    new_gt{j}.end(count,1)   = e;
                    new_gt{j}.label(count,1) = gt{j}.label(i+1);
                    new_gt{j}.label_name{count} = gt{j}.label_name{count};
                end
            elseif i == M
                count = count + 1;
                new_gt{j}.video_name = gt{j}.video_name;
                new_gt{j}.start(count,1) = s;
                new_gt{j}.end(count,1)   = e;
                new_gt{j}.label(count,1) = gt{j}.label(i);
                new_gt{j}.label_name{count} = gt{j}.label_name{count};
            end
        end
    end
end

% before 20161103 ver
% function [gt, label_name_list] = read_gt_THUMOS14(gt_path)
%     % read annotation files
%     annotation_file_list = dir(strcat(gt_path,'/*.txt'));
%     j=1;
%     for i=1:length(annotation_file_list)
%         tmp = strsplit(annotation_file_list(i).name, '_');
%         if ~strcmp(tmp{1}, 'Ambiguous')
%             label_name_list{j} = tmp{1};
%             j = j+1;
%         end
%     end
%     j=1;
%     for i=1:length(annotation_file_list)
% %         if ~strcmp(annotation_file_list(i).name, 'Ambiguous_val.txt')
%         tmp = strsplit(annotation_file_list(i).name, '_');
%         if ~strcmp(tmp{1}, 'Ambiguous')
%             fid = fopen(fullfile(gt_path,annotation_file_list(i).name));
%             while 1
%                 tline = fgetl(fid);
%                 if ~ischar(tline), break, end
%                 tmp = strsplit(tline, ' ');
%                 tmp_name = strsplit(tmp{1},'_');
%                 raw_gt(j,1) = str2num(tmp_name{3}); % video_number
%                 raw_gt(j,2) = str2num(tmp{2});  % start frame
%                 raw_gt(j,3) = str2num(tmp{3});  % end frame
%                 raw_gt(j,4) = i-1;               % action label
%                 j = j+1;
%             end
%             fclose(fid);
%         end
%     end
%     new_gt = sortrows(raw_gt);
%     % select the raw data containing the temporal annotations
%     effective_videos = unique(new_gt(:,1));
%     for i=1:length(effective_videos)
%         indices = find(new_gt(:,1) == effective_videos(i));
%         gt{i}.video_name = effective_videos(i);
%         gt{i}.start = new_gt(indices,2);
%         gt{i}.end = new_gt(indices,3);
%         gt{i}.label = new_gt(indices,4); 
%         for j=1:length(indices)
%             gt{i}.label_name{j} = label_name_list{new_gt(indices(j),4)};
%         end        
%     end
% end
