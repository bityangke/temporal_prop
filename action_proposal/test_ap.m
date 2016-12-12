function info = test_ap(varargin)
% CNN_IMAGENET_EVALUATE   Evauate MatConvNet models on ImageNet

run ../matlab/vl_setupnn;
addpath('./Union');
addpath('./range_intersection/');

% opts.dataDir = fullfile('..', '..','st-slice-cnn-tar','data', 'THUMOS14') ;
% opts.dataDir = fullfile('..', '..','st-slice-cnn-tar','data', 'THUMOS14', 'test') ;
opts.dataDir = fullfile('..','..', '..', '..', 'dataset','action', 'THUMOS14', 'test') ; % modify this line to set up the data path
opts.expDir = fullfile('..', 'data', 'exp_20161208_STROIPool_fulldata') ;
opts.modelPath = fullfile('..', 'data', 'exp_20161208_STROIPool_fulldata', 'net-epoch-30.mat'); %'imagenet-vgg-f.mat');%'imagenet-resnet-50-dag.mat') ;
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


% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
    imdb = load(opts.imdbPath) ;
    imdb.imageDir = fullfile(opts.dataDir, 'images');
else
    imdb = setup_ap_THUMOS14(opts.dataDir, 1);
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                    Network loading
% -------------------------------------------------------------------------
res = [] ;


net = load(opts.modelPath) ;
% remove the output layer
net.layers = net.layers(1:end-6);
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true)

% -------------------------------------------------------------------------
%                                                    Feature extraction
% -------------------------------------------------------------------------
thIoU = 0.01:0.01:1;
% loop over videos
for i=1:length(imdb.images.path)
    % loop over frames
    frames = load(imdb.images.path{i});
    labels = imdb.images.labels{i};
%     fprintf('i=%d\n',i);
    [starts, durations] = generate_temporal_proposal(frames.im, 100);
    for j=1:100
        [tp(i,j), fn(i,j)] = evaluate_temporal_proposal(labels, starts, durations, thIoU(j));
    end
    % extract CNN features
%     cnn_feat = {};
%     for j=1:length(frames.im)
%         im = frames.im{j};
%         im_ = single(im) ; % note: 0-255 range
%         im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
%         im_ = im_ - net.meta.normalization.averageImage;
%         net.eval({'input', im_});
%         cnn_feat{j} = net.vars(net.getVarIndex('x15')).value;
%     end
    % temporal max pooling or uniform sampling to generate boxes

    % for each box, perform regression (perform? or train?)

end
recall = sum(tp)./(sum(tp)+sum(fn));
plot(thIoU, recall);
xlabel('thIoU');
ylabel('recall');
% title('recall-thIoU');

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
