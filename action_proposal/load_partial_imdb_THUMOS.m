function [new_imdb] = load_partial_imdb_THUMOS(imdb, expDir)

new_imdb = imdb;
list_1D = clean_dir(fullfile(expDir, '/1D_part/'));

for i=1:size(list_1D,2)
    tmp = strsplit(list_1D{i}, '_');
    existing_feat(i,1) = str2num(tmp{5});
end

existing_feat = sort(existing_feat);

new_imdb.images.set = imdb.images.set(existing_feat);
new_imdb.images.name = imdb.images.name(existing_feat);
new_imdb.images.path = imdb.images.path(existing_feat);
new_imdb.images.labels = imdb.images.labels(existing_feat);

for i=1:size(existing_feat,1)
    new_imdb.images.feature_path{i} = sprintf('../data/imagenet12-eval-vgg-f/1D_part/imagenet-vgg_relu5_on_THUMOS14val_%d_1D.mat',i);
end