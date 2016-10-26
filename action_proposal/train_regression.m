function info = test_ap(varargin)
run ../matlab/vl_setupnn;
addpath('./Union');
addpath('./range_intersection/');

% -------------------------------------------------------------------------
%                                                   Paths & Params Setting
% -------------------------------------------------------------------------
%%%% paths
dataDir   = fullfile('..', '..','st-slice-cnn-tar','data', 'THUMOS14'); % modify this line to set up the data path
expDir    = fullfile('..', 'data', 'imagenet12-eval-vgg-f') ;
imdbPath  = fullfile(expDir, 'imdb.mat');
modelPath = fullfile('..', 'models', 'imagenet-alex.mat'); %'imagenet-vgg-f.mat');%'imagenet-resnet-50-dag.mat') ;
%%%% params
tempPoolingFilterSize = 1;   % Temporal MaxPooling filter size
tempPoolingStepSize   = 1;   % Temporal MaxPooling step size (stride)
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
%                                                    Network loading
% -------------------------------------------------------------------------
net = load(modelPath) ;
% remove the fc layers
net.layers = net.layers(1:end-6);
% convert simplenn to dagnn
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true)

% -------------------------------------------------------------------------
%                                                    Feature extraction
% -------------------------------------------------------------------------
proposal_total_feature = [];
ts_total = [];
tl_total = [];
num_videos = length(imdb.images.path);

num_videos = 10;
% loop over videos
for i=1:num_videos
    fprintf('extracting features from video ... %d/%d\n', i, num_videos);
    % loop over frames
    frames = load(imdb.images.path{i});
    labels = imdb.images.labels{i};

    % grid partitioning in temporal domain of a training video
    [starts{i}, durations{i}] = generate_temporal_proposal(frames.im, 1);

    % match GT labels and proposals
    [matched_starts{i}, matched_durations{i}] = match_gt_proposal(labels, starts{i}, durations{i});

    % extract CNN features
    cnn_feat = {};
    for j=1:length(frames.im)
        im = frames.im{j};
        im_ = single(im) ; % note: 0-255 range
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
        im_ = im_ - net.meta.normalization.averageImage;
        net.eval({'input', im_});
        cnn_feat{j,1} = net.vars(net.getVarIndex('x15')).value;
    end
    % temporal max pooling or uniform sampling to generate boxes
    temp_pool_cnn_feat = temporal_max_pooling(cnn_feat, tempPoolingFilterSize, tempPoolingStepSize);

    % extract features for each temporal proposal
    [~, prop_feat_current] = extract_proposal_features(temp_pool_cnn_feat, shrinkFactor, matched_starts{i}, matched_durations{i});
    proposal_total_feature = [proposal_total_feature; prop_feat_current];
    
    % extract regression target values
    [ts_current, tl_current] = extract_regression_target_values(labels, shrinkFactor, matched_starts{i}, matched_durations{i});
    ts_total = [ts_total; ts_current];
    tl_total = [tl_total; tl_current];
end

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
[ws, stat_s] = lasso(proposal_total_feature, ts_total);
[wl, stat_l] = lasso(proposal_total_feature, tl_total);
