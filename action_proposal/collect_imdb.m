function new_imdb = collect_imdb(imdb)

N = numel(imdb.images.labels);
ncls = 1;
sums = zeros(ncls,4);
squared_sums = zeros(ncls,4);
class_counts = zeros(ncls,1) + eps;

new_imdb.meta = imdb.meta;
new_imdb.imageDir = imdb.imageDir;
count = 0;
for i=1:N
     duration = imdb.images.labels{i}.gt_end_frames - imdb.images.labels{i}.gt_start_frames + 1;
     if duration >= 100 && duration < 120
        count = count + 1;
        new_imdb.images.labels{count} = imdb.images.labels{i};
        new_imdb.images.set(count)    = imdb.images.set(i);
        new_imdb.images.path{count}   = imdb.images.path{i};
        new_imdb.images.name{count}   = imdb.images.name{i};
        new_imdb.images.feature_path{count} = imdb.images.feature_path{i};
        
        new_imdb.boxes.proposals{count} = imdb.boxes.proposals{i};
        new_imdb.boxes.labels{count} = imdb.boxes.labels{i};
        new_imdb.boxes.ptargets{count} = imdb.boxes.ptargets{i};
     end
end

for i=1:numel(new_imdb.boxes.proposals)
    if new_imdb.images.set(i)<3
        pos =  (new_imdb.boxes.proposals{i}.labels>0) & (new_imdb.boxes.proposals{i}.labels<=ncls) ;
        labels = new_imdb.boxes.proposals{i}.labels(pos);
        targets = new_imdb.boxes.ptargets{i}(pos,:);
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

new_imdb.boxes.bboxMeanStd{1} = means;
new_imdb.boxes.bboxMeanStd{2} = stds;

% normalize proposal targets to have mean zero, stdev 1
% do not normalize proposal targets of y and height
for i=1:numel(new_imdb.boxes.ptargets)
    norm_ptargets{i} = bsxfun(@minus, new_imdb.boxes.ptargets{i}, means);
    norm_ptargets{i}(:,1) = bsxfun(@rdivide, norm_ptargets{i}(:,1), stds(:,1));
    norm_ptargets{i}(:,3) = bsxfun(@rdivide, norm_ptargets{i}(:,3), stds(:,3));
    new_imdb.boxes.ptargets{i} = norm_ptargets{i};
end


display('bbox target means:');
display(means);
display('bbox target stddevs:');
display(stds);