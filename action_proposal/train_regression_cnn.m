function info = train_regression_cnn(varargin)
run ../matlab/vl_setupnn;
addpath('./Union');
addpath('./range_intersection/');

% -------------------------------------------------------------------------
%                                                   Paths & Params Setting
% -------------------------------------------------------------------------
%%%% paths
dataDir   = fullfile('..', '..','..','..','dataset', 'action', 'THUMOS14', 'val'); % modify this line to set up the data path
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
opts.train.gpus = [] ;
opts.piecewise = true;  % piecewise training (+bbox regression)
opts.train.batchSize = 1;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.prefetch = false ; % does not help for two images in a batch
opts.train.learningRate = 1e-3 / 64 * [ones(1,6) 0.1*ones(1,6)];
opts.train.weightDecay = 0.0005 ;
opts.train.numEpochs = 12;
opts.train.derOutputs = {'losscls', 1, 'lossbbox', 1} ;
opts.lite = false  ;
opts.numFetchThreads = 2 ;

opts = vl_argparse(opts, varargin) ;
display(opts);

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

imdb = load_partial_imdb_THUMOS(imdb, expDir);
% for i=1:num_videos
%     imdb.images.feature_path{i} = sprintf('../data/imagenet12-eval-vgg-f/1D_part/imagenet-vgg_relu5_on_THUMOS14val_%d_1D.mat',i);
% end

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

num_videos = 1;
% loop over videos
% for i=3:3
% for i=1:num_videos
%     fprintf('extracting features from video ... %d/%d\n', i, num_videos);
%     % loop over frames
%     load(fullfile(expDir, '../THUMOS14/vgg16_relu5_features/imagenet-vgg_relu5_on_THUMOS14val_1.mat'));
% %     cnn_feat = cnn_feat_total{1,i};
%     frames = load(imdb.images.path{i});
%     labels = imdb.images.labels{i};
%
%     % grid partitioning in temporal domain of a training video
% %     [starts{i}, durations{i}] = generate_temporal_proposal(frames.im, 100);
% %     [starts{i}, durations{i}] = generate_temporal_proposal2(frames.im); % with various filter sizes and strides
% %     [starts{i}, durations{i}] = generate_temporal_proposal2(cnn_feat); % with various filter sizes and strides
%
%     % match GT labels and proposals
% %     [proposals, targets] = get_training_proposal(labels, starts{i}, durations{i}, 64);
% %     [matched_starts{i}, matched_durations{i}] = match_gt_proposal(labels, starts{i}, durations{i});
% %     proposals.starts = matched_starts{i};
% %     proposals.durations = matched_durations{i};
% %
% %     % extract CNN features
% %     cnn_feat = {};
% %     for j=1:length(frames.im)
% %         if mod(j,100) == 0
% %             fprintf('features of frame %d/%d\n', j, length(frames.im));
% %         end
% %         im  = single(frames.im{j});
% %         im_ = imresize(im, net.meta.normalization.imageSize(1:2));
% %         im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
% %
% %         net.eval({'input', im_});
% %         cnn_feat{j,1} = net.vars(net.getVarIndex('x30')).value;
% %     end
% %     save(fullfile(expDir, sprintf('imagenet-vgg_relu5_on_THUMOS14val_%d.mat',i)), 'cnn_feat','-v7.3');
%
% %     % channel subsampling
% %     cnn_feat_pooled = channel_pooling(cnn_feat);
% %
% %     % temporal max pooling or uniform sampling to generate boxes
% %     temp_pool_cnn_feat = temporal_max_pooling(cnn_feat_pooled, tempPoolingFilterSize, tempPoolingStepSize);    temp_pool_cnn_feat = temporal_max_pooling(cnn_feat, tempPoolingFilterSize, tempPoolingStepSize);
% %
% %     % extract features for each temporal proposal
% %     [~, prop_feat_current] = extract_proposal_features(temp_pool_cnn_feat, shrinkFactor, matched_starts{i}, matched_durations{i});
% %     proposal_total_feature = [proposal_total_feature; prop_feat_current];
% %
% %     % extract regression target values
% %     [ts_current, tl_current] = extract_regression_target_values(labels, shrinkFactor, matched_starts{i}, matched_durations{i});
% %
% %     targets.ts = ts_current;
% %     targets.tl = tl_current;
% %
%     inputs = getBatch(opts, imdb, 1);
% end
% -------------------------------------------------------------------------
%                                      Perform Regularized L1 Regression
% -------------------------------------------------------------------------
modelPath =  fullfile('..','models','imagenet-vgg-verydeep-16.mat');
net = apcnn_init('piecewise', opts.piecewise, 'modelPath', modelPath);

% Regression Sanity Check!
% if size(proposal_total_feature,2) > size(proposal_total_feature,1)
%     fprintf('The feature matrix does not satisfy this condition!: N > D\n');
%     fprintf('Current N=%d < D=%d\n', size(proposal_total_feature,1), size(proposal_total_feature,2));
%     fprintf('You should collect more data points! \n');
%     return;
% end


% minibatch options
bopts = net.meta.normalization;
bopts.useGpu = numel(opts.train.gpus) >  0 ;
bopts.numFgRoisPerImg = 16;
bopts.numRoisPerImg = 64;
bopts.maxScale = 1000;
bopts.scale = 600;
bopts.bgLabel = 2;
bopts.visualize = 0;
bopts.interpolation = net.meta.normalization.interpolation;
bopts.numThreads = opts.numFetchThreads;
bopts.prefetch = opts.train.prefetch;

% Regression with Fast R-CNN!
[net,info] = cnn_train_dag(net, imdb, @(i,b) ...
                           getBatch(bopts,i,b), ...
                           opts.train) ;
% evaluate_regression

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
% Inputs:
%         feature: WxHxCxT single conv5/pool5/relu5 features
%         gt_labels: temporal ground truth label of actions
%         gt_labels.gt_start_frames: Nx1 start_frames vector, N
%         is the number of labels of i-th video
%         gt_labels.gt_end_frames  : Nx1 end_frames vector, N
%         is the number of labels of i-th video
%         proposals.starts   : 1xP vector containing N temporal action starting points
%                              with multiple grid_sizes and strides
%         proposals.durations: 1xP vector containing N temporal action durations
%                              with multiple grid_sizes and strides
% Outputs:
%         inputs: Inputs to CNN eval

opts.visualize = 0;
opts.prefetch = (nargout == 0);
if opts.prefetch, return; end

load(imdb.images.feature_path{batch});

labels = imdb.images.labels{batch};
[starts, durations] = generate_temporal_proposal2(size(oneD_converted_feat,2)); % with various filter sizes and strides
[proposals, my_targets] = get_training_proposal(labels, starts, durations, 64, size(oneD_converted_feat,1) );
fprintf('.');

nb = size(proposals.rois,1);
nc = 2;

% regression error only for positives
instance_weights = zeros(1,1,4*nc,nb,'single');
targets = zeros(1,1,4*nc,nb,'single');

for b=1:nb
    if proposals.labels(b)>0 && proposals.labels(b) ~= 0
        targets(1,1,4*(proposals.labels(b)-1)+1:4*proposals.labels(b),b) = my_targets(b,:)';
        instance_weights(1,1,4*(proposals.labels(b)-1)+1:4*proposals.labels(b),b) = 1;
    end
end

rois = transform_rois(proposals.rois, size(oneD_converted_feat,1));

clear cnn_feat;

if opts.useGpu > 0
  oneD_converted_feat = gpuArray(oneD_converted_feat) ;
  rois = gpuArray(rois) ;
  targets = gpuArray(targets) ;
  instance_weights = gpuArray(instance_weights) ;
end

inputs = {'input', oneD_converted_feat, 'label', proposals.labels, 'rois', rois, 'targets', targets, ...
  'instance_weights', instance_weights} ;
