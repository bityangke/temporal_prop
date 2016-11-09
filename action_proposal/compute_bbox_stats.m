function [imdb] = compute_bbox_stats(imdb)
fprintf('Computing bounding box statistics ... \n');
ncls = 1;
sums = zeros(ncls,4);
squared_sums = zeros(ncls,4);
class_counts = zeros(ncls,1) + eps;

N = size(imdb.images.name,2);
for i=1:N
    fprintf('%d/%d\n',i,N);
    labels = imdb.images.labels{i};
    num_frames = labels.num_frames;
    [starts, durations] = generate_temporal_proposal2(num_frames); % with various filter sizes and strides
    [proposals{i}, my_ptargets{i}] = get_training_proposal(labels, starts, durations, 128, 196);
    imdb.boxes.proposals{i} = proposals{i};
    imdb.boxes.labels{i}    = proposals{i}.labels;
    imdb.boxes.ptargets{i}  = my_ptargets{i};
end

for i=1:numel(proposals)
    if imdb.images.set(i)<3
        pos =  (proposals{i}.labels>0) & (proposals{i}.labels<=ncls) ;
        labels = proposals{i}.labels(pos);
        targets = my_ptargets{i}(pos,:);
%         pos =  (imdb.boxes.plabel{i}>0) & (imdb.boxes.plabel{i}<=ncls) ;
%         labels = imdb.boxes.plabel{i}(pos);
%         targets = imdb.boxes.ptarget{i}(pos,:);
        for c=1:ncls
            cls_inds = (labels==c);
            if sum(cls_inds)>0
                class_counts(c) = class_counts(c) + sum(cls_inds);
                sums(c,:) = sums(c,:) + sum(targets(cls_inds,:));
                squared_sums(c,:) = squared_sums(c,:) + sum(targets(cls_inds,:).^2);
            end
        end
    end
end
means = bsxfun(@rdivide,sums,class_counts);
stds = sqrt(bsxfun(@rdivide,squared_sums,class_counts) - means.^2);

imdb.boxes.bboxMeanStd{1} = means;
imdb.boxes.bboxMeanStd{2} = stds;

% normalize proposal targets to have mean zero, stdev 1
% do not normalize proposal targets of y and height
for i=1:numel(imdb.boxes.ptargets)
    norm_ptargets{i} = bsxfun(@minus, imdb.boxes.ptargets{i}, means);
    norm_ptargets{i}(:,1) = bsxfun(@rdivide, norm_ptargets{i}(:,1), stds(:,1));
    norm_ptargets{i}(:,3) = bsxfun(@rdivide, norm_ptargets{i}(:,3), stds(:,3));
    imdb.boxes.ptargets{i} = norm_ptargets{i};
end


display('bbox target means:');
display(means);
display('bbox target stddevs:');
display(stds);