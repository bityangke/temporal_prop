function info = train_regression_cnn(varargin)
run ../matlab/vl_setupnn;
addpath('./Union');
addpath('./range_intersection/');

% -------------------------------------------------------------------------
%                                                   Paths & Params Setting
% -------------------------------------------------------------------------
%%%% paths
dataDir   = fullfile('..', '..','st-slice-cnn-tar','data', 'THUMOS14'); % modify this line to set up the data path
% dataDir   = fullfile('..', '..','..','..','dataset', 'action', 'THUMOS14', 'val'); % modify this line to set up the data path
expDir    = fullfile('..', 'data', 'imagenet12-eval-vgg-f') ;
imdbPath  = fullfile(expDir, 'imdb.mat');
% modelPath = fullfile('..', 'models', 'imagenet-alex.mat'); %'imagenet-vgg-f.mat');%'imagenet-resnet-50-dag.mat') ;
modelPath = fullfile('..','models','imagenet-vgg-verydeep-16.mat');
%%%% params
tempPoolingFilterSize = 100;   % Temporal MaxPooling filter size
tempPoolingStepSize   = 100;   % Temporal MaxPooling step size (stride)
shrinkFactor          = tempPoolingStepSize;
piecewise = true;

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
net.layers = net.layers(1:end-7);

% convert simplenn to dagnn
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true)

% -------------------------------------------------------------------------
%                                                    Feature extraction
% -------------------------------------------------------------------------
proposal_total_feature = [];
ts_total = [];
tl_total = [];
num_videos = length(imdb.images.path);

% load(fullfile(expDir, 'imagenet-alex_pool5_on_THUMOS14val_1to4.mat'));
% num_videos = size(cnn_feat_total,2);

% loop over videos
for i=1:num_videos
    fprintf('extracting features from video ... %d/%d\n', i, num_videos);
    % loop over frames
%     cnn_feat = cnn_feat_total{1,i};
    frames = load(imdb.images.path{i});
    labels = imdb.images.labels{i};

    % grid partitioning in temporal domain of a training video
%     [starts{i}, durations{i}] = generate_temporal_proposal(frames.im, 100);
    [starts{i}, durations{i}] = generate_temporal_proposal2(frames.im); % with various filter sizes and strides
%     [starts{i}, durations{i}] = generate_temporal_proposal2(cnn_feat); % with various filter sizes and strides

    % match GT labels and proposals
    [matched_starts{i}, matched_durations{i}] = match_gt_proposal(labels, starts{i}, durations{i});

    % extract CNN features
    cnn_feat = {};
    for j=1:length(frames.im)
        im  = single(frames.im{j});
        im_ = imresize(im, net.meta.normalization.imageSize(1:2));
        im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;

        net.eval({'input', im_});
        cnn_feat{j,1} = net.vars(net.getVarIndex('x30')).value;
    end
    % channel subsampling
    cnn_feat_pooled = channel_pooling(cnn_feat);

    % temporal max pooling or uniform sampling to generate boxes
    temp_pool_cnn_feat = temporal_max_pooling(cnn_feat_pooled, tempPoolingFilterSize, tempPoolingStepSize);
    % temp_pool_cnn_feat = temporal_max_pooling(cnn_feat, tempPoolingFilterSize, tempPoolingStepSize);

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
modelPath =  fullfile('..','models','imagenet-vgg-verydeep-16.mat');
net = apcnn_init('piecewise', piecewise, 'modelPath', modelPath);

% Regression Sanity Check!
if size(proposal_total_feature,2) > size(proposal_total_feature,1)
    fprintf('The feature matrix does not satisfy this condition!: N > D\n');
    fprintf('Current N=%d < D=%d\n', size(proposal_total_feature,1), size(proposal_total_feature,2));
    fprintf('You should collect more data points! \n');
    return;
end
% Regression with Lasso!
% [ws, stat_s] = lasso(proposal_total_feature, ts_total);
% [wl, stat_l] = lasso(proposal_total_feature, tl_total);
[ws, stat_s] = lasso(proposal_total_feature, ts_total, 'Lambda', 0.01);
[wl, stat_l] = lasso(proposal_total_feature, tl_total, 'Lambda', 0.01);

save(fullfile(expDir, 'ws.mat'), 'ws','-v7.3');
save(fullfile(expDir, 'stat_s.mat'), 'stat_s','-v7.3');
save(fullfile(expDir, 'wl.mat'), 'wl','-v7.3');
save(fullfile(expDir, 'stat_l.mat'), 'stat_l','-v7.3');

evaluate_regression

% --------------------------------------------------------------------
function inputs = getBatch(opts, feature, gt_labels)
% --------------------------------------------------------------------
% Inputs:
%         feature: WxHxCxT single conv5/pool5/relu5 features
%         gt_labels: temporal ground truth label of actions
%         gt_labels.gt_start_frames: Nx1 start_frames vector, N
%         is the number of labels of i-th video
%         gt_labels.gt_end_frames  : Nx1 end_frames vector, N
%         is the number of labels of i-th video
% Outputs: 
%         inputs: 
opts.visualize = 0;

if isempty(batch)
  return;
end

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
opts.prefetch = (nargout == 0);

[im,rois,labels,btargets] = fast_rcnn_train_get_batch(images,imdb,...
  batch, opts);

if opts.prefetch, return; end

nb = numel(labels);
nc = numel(imdb.classes.name) + 1;

% regression error only for positives
instance_weights = zeros(1,1,4*nc,nb,'single');
targets = zeros(1,1,4*nc,nb,'single');

for b=1:nb
  if labels(b)>0 && labels(b)~=opts.bgLabel
    targets(1,1,4*(labels(b)-1)+1:4*labels(b),b) = btargets(b,:)';
    instance_weights(1,1,4*(labels(b)-1)+1:4*labels(b),b) = 1;
  end
end

rois = single(rois);

if opts.useGpu > 0
  im = gpuArray(im) ;
  rois = gpuArray(rois) ;
  targets = gpuArray(targets) ;
  instance_weights = gpuArray(instance_weights) ;
end

inputs = {'input', im, 'label', labels, 'rois', rois, 'targets', targets, ...
  'instance_weights', instance_weights} ;