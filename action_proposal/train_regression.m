function info = test_ap(varargin)
% CNN_IMAGENET_EVALUATE   Evauate MatConvNet models on ImageNet

run ../matlab/vl_setupnn;
addpath('./Union');
addpath('./range_intersection/');

opts.dataDir = fullfile('..','..', '..', '..', 'dataset','action', 'THUMOS14', 'val') ; % modify this line to set up the data path
opts.expDir = fullfile('..', 'data', 'imagenet12-eval-vgg-f') ;
opts.modelPath = fullfile('..', 'models', 'imagenet-alex.mat'); %'imagenet-vgg-f.mat');%'imagenet-resnet-50-dag.mat') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.networkType = [] ;
opts.lite = false ;
opts.numFetchThreads = 12 ;
opts.train.batchSize = 128 ;
opts.train.numEpochs = 1 ;
opts.train.gpus = [] ;
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;
display(opts);

tempPoolingFilterSize = 10;   % Temporal MaxPooling filter size
tempPoolingStepSize   = 10;   % Temporal MaxPooling step size (stride)

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
if exist(opts.imdbPath)
    imdb = load(opts.imdbPath) ;
    imdb.imageDir = fullfile(opts.dataDir, 'images');
else
    imdb = setup_ap_THUMOS14(opts.dataDir, 0);
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                    Network loading
% -------------------------------------------------------------------------
net = load(opts.modelPath) ;
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
% loop over videos
for i=1:5%length(imdb.images.path)
    % loop over frames
    frames = load(imdb.images.path{i});
    labels = imdb.images.labels{i};

    % grid partitioning in temporal domain of a training video
    [starts{i}, durations{i}] = generate_temporal_proposal(frames.im, 100);

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
    %temp_pool_cnn_feat{i,1} = temporal_max_pooling(cnn_feat, tempPoolingFilterSize, tempPoolingStepSize);
    % extract features for each temporal proposal
    [~, prop_feat_current] = extract_proposal_features(temp_pool_cnn_feat, 10, matched_starts{i}, matched_durations{i});
    proposal_total_feature = [proposal_total_feature; prop_feat_current];
    [ts_current, tl_current] = extract_regression_target_values(labels, 10, matched_starts{i}, matched_durations{i});
    ts_total = [ts_total; ts_current];
    tl_total = [tl_total; tl_current];
end

% -------------------------------------------------------------------------
%                                      Perform Regularized L1 Regression
% -------------------------------------------------------------------------
[ws, stat_s] = lasso(proposal_total_feature, ts_total);
[wl, stat_l] = lasso(proposal_total_feature, tl_total);


% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
if isfield(meta.normalization, 'keepAspect')
  keepAspect = meta.normalization.keepAspect ;
else
  keepAspect = true ;
end

if numel(meta.normalization.averageImage) == 3
  mu = double(meta.normalization.averageImage(:)) ;
else
  mu = imresize(single(meta.normalization.averageImage), ...
                meta.normalization.imageSize(1:2)) ;
end

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct(...
  'useGpu', useGpu, ...
  'numThreads', opts.numFetchThreads, ...
  'imageSize',  meta.normalization.imageSize(1:2), ...
  'cropSize', max(meta.normalization.imageSize(1:2)) / 256, ...
  'subtractAverage', mu, ...
  'keepAspect', keepAspect) ;

fn = @(x,y) getBatch(bopts,useGpu,lower(opts.networkType),x,y) ;

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;
else
  phase = 'test' ;
end
data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
  labels = imdb.images.label(batch) ;
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end
