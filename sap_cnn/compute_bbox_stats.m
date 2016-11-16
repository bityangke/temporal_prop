function [imdb] = compute_bbox_stats(imdb)
fprintf('Computing bounding box statistics ... \n');
ncls = 1;
sums = zeros(ncls,4);
squared_sums = zeros(ncls,4);
class_counts = zeros(ncls,1) + eps;

N = numel(imdb.images.name);

%% Generate temporal action proposals
for i=1:N
    fprintf('%d/%d\n',i,N);
    labels = imdb.images.labels{i};
    num_frames = labels.num_frames;
    [starts, durations] = generate_temporal_proposal2(num_frames); % with various filter sizes and strides
    [proposals{i}, my_ptargets{i}] = get_training_proposal(labels, starts, durations, 128, imdb.images.size(i,2));
    imdb.boxes.proposals{i} = proposals{i};
    imdb.boxes.labels{i}    = proposals{i}.labels;
    
    imdb.boxes.pbox{i,1}      = proposals{i}.rois;
    imdb.boxes.plabel{i,1}    = proposals{i}.labels;
    imdb.boxes.piou{i,1}      = proposals{i}.piou;
    imdb.boxes.pgtidx{i,1}    = proposals{i}.pgtidx;
    imdb.boxes.ptarget{i,1}   = my_ptargets{i};
end

%% Compute bounding box statistics
for i=1:numel(proposals)
    if imdb.images.set(i)<3
        pos =  (proposals{i}.labels>0) & (proposals{i}.labels<=ncls) ;
        labels = proposals{i}.labels(pos);
        targets = my_ptargets{i}(pos,:);
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
for i=1:numel(imdb.boxes.ptarget)
    norm_ptargets{i} = bsxfun(@minus, imdb.boxes.ptarget{i}, means);
    norm_ptargets{i}(:,1) = bsxfun(@rdivide, norm_ptargets{i}(:,1), stds(:,1));
    norm_ptargets{i}(:,3) = bsxfun(@rdivide, norm_ptargets{i}(:,3), stds(:,3));
    imdb.boxes.ptarget{i} = norm_ptargets{i};
end

display('bbox target means:');
display(means);
display('bbox target stddevs:');
display(stds);