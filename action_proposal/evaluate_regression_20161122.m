load(fullfile(expDir, 'ws_20161122.mat'));
load(fullfile(expDir, 'stat_s_20161122.mat'));
load(fullfile(expDir, 'wl_20161122.mat'));
load(fullfile(expDir, 'stat_l_20161122.mat'));

num_videos = 3;

for i=1:num_videos
    fprintf('extracting features from video ... %d/%d\n', i, num_videos);
    % loop over frames
    load(imdb.images.feature_path{i});
    cnn_feat = current_GT_1D_feat;
%     frames = load(imdb.images.path{i});
    labels = imdb.images.labels{i};

    % grid partitioning in temporal domain of a training video
    [starts{i}, durations{i}] = generate_temporal_proposal_for_lasso(size(cnn_feat,2)); % with various filter sizes and strides
 
    % channel subsampling
    cnn_feat_channel_subsampled = channel_subsampling(cnn_feat);

    % temporal max pooling or uniform sampling to generate boxes
%     temp_pool_cnn_feat = temporal_max_pooling(cnn_feat_pooled, tempPoolingFilterSize, tempPoolingStepSize);
    temp_pool_cnn_feat = cnn_feat_channel_subsampled;

    % extract features for each temporal proposal
    prop_feat_current = extract_proposal_features2(temp_pool_cnn_feat, {starts{i}}, {durations{i}});
    
    ds = ws'*prop_feat_current'; % ds is 1xN, where N is the number of proposals
    dl = wl'*prop_feat_current'; % dl is 1xN, where N is the number of proposals
    
    Gs_estimated = durations{i} .* ds + starts{i}; % durations{i} and starts{i} are 1xN
    Gl_estimated = durations{i} .* exp(dl);
    
    
end