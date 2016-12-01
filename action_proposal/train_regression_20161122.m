function info = train_regression_20161122(varargin)
run ../matlab/vl_setupnn;
addpath('./Union');
addpath('./range_intersection/');

% -------------------------------------------------------------------------
%                                                   Paths & Params Setting
% -------------------------------------------------------------------------
%%%% paths
dataDir   = fullfile('..', '..','st-slice-cnn-tar','data', 'THUMOS14'); % modify this line to set up the data path
% dataDir   = fullfile('..', '..','..','..','dataset', 'action', 'THUMOS14', 'val'); % modify this line to set up the data path
expDir    = fullfile('..', 'data', 'exp_20161117_LinearRegression_256sample_128pos_denser_window_tempPool3') ;
imdbPath  = fullfile(expDir, 'imdb.mat');

%%%% params
tempPoolingFilterSize = 100;   % Temporal MaxPooling filter size
tempPoolingStepSize   = 100;   % Temporal MaxPooling step size (stride)
shrinkFactor          = tempPoolingStepSize;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
if exist(imdbPath, 'file')
    imdb = load(imdbPath) ;
    imdb.imageDir = fullfile(dataDir, 'images');
else
    imdb = setup_ap_THUMOS14(dataDir, 0);
    mkdir(expDir) ;
    save(imdbPath, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                    Feature extraction
% -------------------------------------------------------------------------
proposal_total_feature = [];
ts_total = [];
tl_total = [];
num_videos = numel(imdb.images.feature_path);

% loop over videos
for i=1:num_videos
    fprintf('extracting features from video ... %d/%d\n', i, num_videos);
    load(imdb.images.feature_path{i});
    cnn_feat = current_GT_1D_feat;
%     frames = load(imdb.images.path{i});
    labels = imdb.images.labels{i};

    % grid partitioning in temporal domain of a training video
    [starts{i}, durations{i}] = generate_temporal_proposal_for_lasso(size(cnn_feat,2)); % with various filter sizes and strides

    % match GT labels and proposals
    [matched_starts{i}, matched_durations{i}] = match_gt_proposal(labels, starts{i}, durations{i});

    % channel subsampling
    cnn_feat_channel_subsampled = channel_subsampling(cnn_feat);

    % temporal max pooling or uniform sampling to generate boxes
%     temp_pool_cnn_feat = temporal_max_pooling(cnn_feat_channel_subsampled, tempPoolingFilterSize);
    temp_pool_cnn_feat = cnn_feat_channel_subsampled;
    
    % extract features for each temporal proposal
    prop_feat_current = extract_proposal_features2(temp_pool_cnn_feat, matched_starts{i}, matched_durations{i});
    proposal_total_feature = [proposal_total_feature; prop_feat_current];

    % extract regression target values
    [ts_current, tl_current] = extract_regression_target_values(labels, 1, matched_starts{i}, matched_durations{i});
    ts_total = [ts_total; ts_current];
    tl_total = [tl_total; tl_current];
end
% save(fullfile(expDir, 'proposal_total_feature_20161122.mat'), 'proposal_total_feature','-v7.3');
% save(fullfile(expDir, 'ts_total_20161122.mat'), 'ts_total','-v7.3');
% save(fullfile(expDir, 'tl_total_20161122.mat'), 'tl_total','-v7.3');

% -------------------------------------------------------------------------
%                                      Perform Regularized L1 Regression
% -------------------------------------------------------------------------
% Regression Sanity Check!
if size(proposal_total_feature,2) > size(proposal_total_feature,1)
    fprintf('The feature matrix does not satisfy this condition!: N > D\n');
    fprintf('Current N=%d < D=%d\n', size(proposal_total_feature,1), size(proposal_total_feature,2));
    fprintf('You should collect more data points! \n');
    return;
end
% Regression with Lasso!
[ws, stat_s] = lasso(proposal_total_feature, ts_total, 'Lambda', 0.01);
[wl, stat_l] = lasso(proposal_total_feature, tl_total, 'Lambda', 0.01);

save(fullfile(expDir, 'ws_20161122.mat'), 'ws','-v7.3');
save(fullfile(expDir, 'stat_s_20161122.mat'), 'stat_s','-v7.3');
save(fullfile(expDir, 'wl_20161122.mat'), 'wl','-v7.3');
save(fullfile(expDir, 'stat_l_20161122.mat'), 'stat_l','-v7.3');

evaluate_regression_20161122