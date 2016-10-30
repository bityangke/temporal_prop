load(fullfile(expDir, 'ws.mat'));
load(fullfile(expDir, 'stat_s.mat'));
load(fullfile(expDir, 'wl.mat'));
load(fullfile(expDir, 'stat_l.mat'));

num_videos = 3;

for i=1:num_videos
    fprintf('extracting features from video ... %d/%d\n', i, num_videos);
    % loop over frames
    cnn_feat = cnn_feat_total{1,i};
%     frames = load(imdb.images.path{i});
    labels = imdb.images.labels{i};

    % grid partitioning in temporal domain of a training video
%     [starts{i}, durations{i}] = generate_temporal_proposal(frames.im, 1);
%     [starts{i}, durations{i}] = generate_temporal_proposal2(frames.im); % with various filter sizes and strides
    [starts{i}, durations{i}] = generate_temporal_proposal2(cnn_feat); % with various filter sizes and strides

    % extract CNN features
%     cnn_feat = {};
%     for j=1:length(frames.im)
%         im = frames.im{j};
%         im_ = single(im) ; % note: 0-255 range
%         im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
%         im_ = im_ - net.meta.normalization.averageImage;
%         net.eval({'input', im_});
%         cnn_feat{j,1} = net.vars(net.getVarIndex('x15')).value;
%     end
    % channel subsampling
    cnn_feat_pooled = channel_pooling(cnn_feat);

    % temporal max pooling or uniform sampling to generate boxes
    temp_pool_cnn_feat = temporal_max_pooling(cnn_feat_pooled, tempPoolingFilterSize, tempPoolingStepSize);

    % extract features for each temporal proposal
    [~, prop_feat_current] = extract_proposal_features(temp_pool_cnn_feat, shrinkFactor, {starts{i}}, {durations{i}});
    
    ds = ws'*prop_feat_current'; % ds is 1xN, where N is the number of proposals
    dl = wl'*prop_feat_current'; % dl is 1xN, where N is the number of proposals
    
    Gs_estimated = durations{i} .* ds + starts{i}; % durations{i} and starts{i} are 1xN
    Gl_estimated = durations{i} .* exp(dl);
    
    
end