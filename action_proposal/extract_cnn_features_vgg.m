function info = extract_cnn_features(varargin)
run ../matlab/vl_setupnn;
addpath('./Union');
addpath('./range_intersection/');

% -------------------------------------------------------------------------
%                                                   Paths & Params Setting
% -------------------------------------------------------------------------
%%%% paths
dataDir   = fullfile('..', '..','..','..','dataset', 'action', 'THUMOS14', 'val'); % modify this line to set up the data path
expDir    = fullfile('..', 'data', 'imagenet12-eval-vgg-f') ;
imdbPath  = fullfile(expDir, 'imdb.mat');
% modelPath = fullfile('..', 'models', 'imagenet-alex.mat');
modelPath = fullfile('..','models','imagenet-vgg-verydeep-16.mat');
 %'imagenet-vgg-f.mat');%'imagenet-resnet-50-dag.mat') ;
opts.gpu = [8];
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

% myPool = parpool(32);
% num_videos = 32;
% cnn_feat = cell(num_videos,1);

% Evaluate network either on CPU or GPU.
if numel(opts.gpu) > 0
  gpuDevice(opts.gpu) ;
  net.move('gpu') ;
end

% loop over videos
start_index = 1; % 84,99,114,126 should be recomputed
end_index = 200;

for i=start_index:end_index
% for i=1:num_videos
    fprintf('extracting features from video ... %d/%d\n', i, end_index-start_index+1);
    % loop over frames
    frames = load(imdb.images.path{i});
    cnn_feat = {};
    % extract CNN features
    for j=1:length(frames.im)
        if mod(j,100) == 0
            fprintf('features of frame %d/%d\n', j, length(frames.im));
        end
        im  = single(frames.im{j});
        im_ = imresize(im, net.meta.normalization.imageSize(1:2));
        im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage);
        if numel(opts.gpu) > 0
          im_ = gpuArray(im_) ;
        end
        net.eval({'input', im_});
        clear im_;
        cnn_feat{j,1} = gather(net.vars(net.getVarIndex('x30')).value);
    end
    save(fullfile(expDir, sprintf('imagenet-vgg_relu5_on_THUMOS14val_%d.mat',i)), 'cnn_feat','-v7.3');
end
% save(fullfile(expDir, 'imagenet-alex_pool5_on_THUMOS14val.mat'), 'cnn_feat','-v7.3');
