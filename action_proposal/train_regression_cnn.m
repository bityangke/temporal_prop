function info = train_regression_cnn(varargin)
run ../matlab/vl_setupnn;
addpath('./Union');
addpath('./range_intersection/');

% -------------------------------------------------------------------------
%                                                   Paths & Params Setting
% -------------------------------------------------------------------------
%%%% paths for feature extraction
dataDir   = fullfile('..', '..', 'st-slice-cnn-tar', 'data', 'THUMOS14'); % for MacBook. modify this line to set up the data path
% dataDir   = fullfile('..', '..','..','..','dataset', 'action', 'THUMOS14', 'val'); % for cvmlp server. modify this line to set up the data path
expDir    = fullfile('..', 'data', 'imagenet12-eval-vgg-f') ;
imdbPath  = fullfile(expDir, 'imdb.mat');
modelPath = fullfile('..','models','imagenet-vgg-verydeep-16.mat'); % model path for CNN feature extraction

%%%% params for train regression MLP
opts.train.gpus = [] ;
opts.piecewise = true;  % piecewise training (+bbox regression)
opts.train.expDir = fullfile('.','data','exp_20161109_relu_removed_tempPool3');
opts.train.batchSize = 1;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.prefetch = false ; % does not help for two images in a batch
opts.train.learningRate = 1e-3 / 256 * [ones(1,6) 0.1*ones(1,6)];        % this should be modified
opts.train.weightDecay = 0.0005 ;
opts.train.numEpochs = 12;
opts.train.derOutputs = {'losscls', 1, 'lossbbox', 1} ;
opts.lite = false  ;
opts.numFetchThreads = 2 ;

opts = vl_argparse(opts, varargin) ;
display(opts);

% -------------------------------------------------------------------------
%                                                  Database initialization
% -------------------------------------------------------------------------
if exist(imdbPath, 'file')
    imdb = load(imdbPath) ;
    imdb.imageDir = fullfile(dataDir, 'images');
else
    imdb = setup_ap_THUMOS14(dataDir, 0);
    mkdir(expDir) ;
    imdb = load_partial_imdb_THUMOS(imdb, fullfile(expDir,'1D_part'));
    imdb = compute_bbox_stats(imdb);    
    save(imdbPath, '-struct', 'imdb') ;
end

% imdb = setup_ap_THUMOS14(dataDir, 0);
% new_imdb = load_partial_imdb_THUMOS(imdb, fullfile(expDir,'1D_part'));
% new_imdb = compute_bbox_stats(new_imdb);   

% -------------------------------------------------------------------------
%                                      Train MLP Regressor and Classifier
% -------------------------------------------------------------------------
modelPath =  fullfile('..','models','imagenet-vgg-verydeep-16.mat');
net = apcnn_init('piecewise', opts.piecewise, 'modelPath', modelPath);

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
                       
% --------------------------------------------------------------------
%                                                               Deploy
% --------------------------------------------------------------------
modelPath = fullfile(expDir, 'net-deployed.mat');
if ~exist(modelPath,'file')
    net = deployAPCNN(net, imdb);
    net_ = net.saveobj();
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end
% evaluate_regression


% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
% Inputs:
%         opts  : training options
%         imdb  : training imdb
%         batch : training batch number
% Outputs:
%         inputs: Inputs to CNN eval

opts.visualize = 0;
opts.prefetch = (nargout == 0);
if opts.prefetch, return; end

% 'input dim should be: HxWxCxb, 'label' dim: Nx1, 'rois' dim: 5xN,
% 'targets' dim: 1x1x4(K+1)xN, 'instance_weights' dim:  1x1x4(K+1)xN
% 
%
% in order to process multiple batches, we need to transfrom the video
% featuers to a canonical size

load(imdb.images.feature_path{batch});
fprintf('num_frames = %d\n', imdb.images.labels{batch}.num_frames);

% for i=1:numel(batch)
%     load(imdb.images.feature_path{batch(i)});
%     feat{i} = current_GT_1D_feat;
% end

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

labels = imdb.images.labels{batch};
% [starts, durations] = generate_temporal_proposal2(size(current_GT_1D_feat,2)); % with various filter sizes and strides
% [proposals, my_targets] = get_training_proposal(labels, starts, durations, 64, size(current_GT_1D_feat,1) );
proposals  = imdb.boxes.proposals{batch};
my_targets = imdb.boxes.ptargets{batch};

% ----------------------- new target calculation
% rois = transform_rois(proposals.rois, size(current_GT_1D_feat,1));
% ex_rois = rois(2:5,:)';
% ex_rois(:,3) = ex_rois(:,1) + ex_rois(:,3) - 1; 
% gt_roi = [labels.gt_start_frames, 1, labels.gt_end_frames, size(current_GT_1D_feat,1)];
% gt_rois = single(repmat(gt_roi, [size(rois,2) 1]));
% targets_new = bbox_transform(ex_rois, gt_rois);
% ----------------------- new target calculation end

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
% for b=1:nb
%     if proposals.labels(b)>0 && proposals.labels(b) ~= 0
%         targets(1,1,4*(proposals.labels(b)-1)+1:4*proposals.labels(b),b) = my_targets(b,:)';
%         instance_weights(1,1,4*(proposals.labels(b)-1)+1:4*proposals.labels(b),b) = 1;
%     end
% end

rois = transform_rois2(proposals.rois, size(current_GT_1D_feat,1));

if opts.useGpu > 0
  current_GT_1D_feat = gpuArray(current_GT_1D_feat) ;
  rois = gpuArray(rois) ;
  targets = gpuArray(targets) ;
  instance_weights = gpuArray(instance_weights) ;
end

inputs = {'input', current_GT_1D_feat, 'label', proposals.labels, 'rois', rois, 'targets', targets, ...
  'instance_weights', instance_weights} ;

