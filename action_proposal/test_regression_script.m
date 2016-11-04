% testing regression and classification
clear

run ../matlab/vl_setupnn;
addpath('./Union');
addpath('./range_intersection/');

opts.classes = {'action'} ;
opts.confThreshold = 0.5;
imdbDir    = fullfile('..', 'data', 'imagenet12-eval-vgg-f') ;
modelPath =  fullfile('.','data','exp_20161103_0800', 'net-epoch-12.mat');
train_imdb_path = fullfile(imdbDir, 'imdb.mat');
expDir    = fullfile(imdbDir, '1D_part_test_cheat');
opts.gpu = [];

% Load train imdb
imdb = load(train_imdb_path) ;
% imdb.imageDir = fullfile(dataDir, 'images');

% Load the network and put it in test mode.
net = load(modelPath);
net = dagnn.DagNN.loadobj(net.net);
net = deployAPCNN(net, imdb);
net.mode = 'test' ;

% Load a test image(feature) and candidate bounding boxes.
load(fullfile(expDir, 'imagenet-vgg_relu5_on_THUMOS14val_60_1D.mat'));

% Load test GT imdb
dataDir   = fullfile('..', '..', 'st-slice-cnn-tar', 'data', 'THUMOS14'); % for MacBook. modify this line to set up the data path
imdb_test_GT = setup_ap_THUMOS14(dataDir, 0);
bbox_label = imdb_test_GT.images.labels{60};
if imdb_test_GT.images.labels{60}.num_frames ~= size(oneD_converted_feat,2)
    fprintf('Mismatch between gt bbox and the video feature!\n');
end

% construct a gt bbox
for i=1:numel(bbox_label.gt_start_frames)
    gt_bbox(i,1) = bbox_label.gt_start_frames(i); % x
    gt_bbox(i,2) = 1; % y
    gt_bbox(i,3) = bbox_label.gt_end_frames(i)-bbox_label.gt_start_frames(i)+1; % width
    gt_bbox(i,4) = size(oneD_converted_feat,1); % height
end

count = 1;
% construct proposal boxes around GT
for i=1:size(gt_bbox,1)
    d = double(gt_bbox(i,3));
    number = round(rand(1));
    if number==1
        direction = 1;
    else
        direction = -1;
    end
    shift = d*0.5*direction;
    
    scale = 2*rand(1);
    if scale > 1 
        sc_shift = round(-d*0.3);
    else
        sc_shift = round(d*0.3);
    end
    
    p_bbox(:,count) = [1 gt_bbox(i,1)+shift gt_bbox(i,2) gt_bbox(i,3) gt_bbox(i,4)]';   % shift
    p_bbox(:,count+1) = [1 gt_bbox(i,1)-shift gt_bbox(i,2) gt_bbox(i,3) gt_bbox(i,4)]'; % shift
    p_bbox(:,count+2) = [1 gt_bbox(i,1)+sc_shift gt_bbox(i,2) round(scale*gt_bbox(i,3)) gt_bbox(i,4)]'; % scale and shift
    p_bbox(:,count+3) = [1 gt_bbox(i,1)+sc_shift gt_bbox(i,2) round(scale*gt_bbox(i,3)) gt_bbox(i,4)]'; % scale and shift
    count = count + 4;
end
rois = single(p_bbox);

% Evaluate network either on CPU or GPU.
if numel(opts.gpu) > 0
    gpuDevice(opts.gpu) ;
    oneD_converted_feat = gpuArray(oneD_converted_feat) ;
    rois = gpuArray(rois) ;
    net.move('gpu') ;
end

net.conserveMemory = false ;
net.eval({'input', oneD_converted_feat, 'rois', rois});

% Extract class probabilities and  bounding box refinements
probs = squeeze(gather(net.vars(net.getVarIndex('probcls')).value));
deltas = squeeze(gather(net.vars(net.getVarIndex('predbbox')).value));

% evaluate (and visualize) results 
eval_rois = rois(2:5,:)';
c = 1; %find(strcmp(opts.classes{1}, net.meta.classes.name)) ;
cprobs = probs(c,:) ;
cdeltas = deltas(4*(c-1)+(1:4),:)' ;
cboxes = bbox_transform_inv(rois(2:5,:)', cdeltas);
cls_dets = [cboxes cprobs'] ;

% loop over GT
for i=1:size(gt_bbox,1)
    current_gt = [bbox_label.gt_start_frames(i), bbox_label.gt_end_frames(i)];
    for j=1:4
        rois_ind = (i-1)*4+j;
        org  = [eval_rois(rois_ind,1), eval_rois(rois_ind,1)+eval_rois(rois_ind,3)-1];
        pred = [cboxes(rois_ind,1), cboxes(rois_ind,1)+cboxes(rois_ind,3)-1];
        iou_org(i,j) = calculateIoU(org, current_gt);
        iou_pred(i,j) = calculateIoU(pred, current_gt);
    end
end

fprintf('mean IoU of original Proposal: %f\n', mean(iou_org(:)));
fprintf('mean IoU of regressed Proposal: %f\n', mean(iou_pred(:)));
