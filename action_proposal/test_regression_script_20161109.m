% testing regression and classification
clear

run ../matlab/vl_setupnn;
addpath('./Union');
addpath('./range_intersection/');

opts.classes = {'action'} ;
opts.confThreshold = 0.5;
imdbDir    = fullfile('..', 'data', 'exp_20161109_256sample_128pos_denser_window_tempPool3') ;
modelPath =  fullfile('..','data','exp_20161109_256sample_128pos_denser_window_tempPool3', 'net-epoch-7.mat');
train_imdb_path = fullfile(imdbDir, 'imdb.mat');
opts.gpu = [];
use_norm = 0;

% Load train imdb
imdb = load(train_imdb_path) ;

% Load the network and put it in test mode.
net = load(modelPath);
net = dagnn.DagNN.loadobj(net.net);
net = deployAPCNN(net, imdb);
net.mode = 'test' ;

% Load training IMDB
imdb_test_GT = imdb;

targets_total = [];
cdeltas_total = [];
cumm_rois = [];
cumm_targets = [];
cprobs_total = [];
labels = [];

loss = zeros(numel(imdb.images.feature_path),1);
for test_video=1:numel(imdb.images.feature_path) %60:79
    close all;
    fprintf('Testing... %d/%d\n',test_video, numel(imdb.images.feature_path));

    if imdb.images.set(test_video) == 1    
        load(imdb.images.feature_path{test_video});
        oneD_converted_feat = current_GT_1D_feat;
    else
        continue;
    end
    
    proposals  = imdb.boxes.proposals{test_video};
    targets = imdb.boxes.ptargets{test_video};
    
    rois = transform_rois2(proposals.rois, size(current_GT_1D_feat,1));

    net.conserveMemory = false ;
    net.eval({'input', oneD_converted_feat, 'rois', rois});
    
    % Extract class probabilities and  bounding box refinements
    probs = squeeze(gather(net.vars(net.getVarIndex('probcls')).value));
    deltas = squeeze(gather(net.vars(net.getVarIndex('predbbox')).value));
    
    % evaluate (and visualize) results 
    c = 1; 
    cprobs = probs(c,:) ;
    cdeltas = deltas(4*(c-1)+(1:4),:)' ;
    cboxes = bbox_transform_inv(rois(2:5,:)', cdeltas);
    cls_dets = [cboxes cprobs'] ;
    
    targets_total = vertcat(targets_total, targets);
    cdeltas_total = vertcat(cdeltas_total, cdeltas);
    cprobs_total  = vertcat(cprobs_total, cprobs(:));
    labels = vertcat(labels, proposals.labels);
    
    loss(test_video) = calculate_smoothL1(cdeltas(proposals.labels>0,:), targets(proposals.labels>0,:));
end
pos=labels>0;

figure
scatter(targets_total(pos,1), targets_total(pos,3), 'r');
hold on;
scatter(cdeltas_total(pos,1), cdeltas_total(pos,3), 'g');
xlabel('dx');
ylabel('dw');
legend('target', 'prediction');
axis([min([targets_total(pos,1);cdeltas_total(pos,1)])-0.1 max([targets_total(pos,1); cdeltas_total(pos,1)])+0.1 min([targets_total(pos,3);cdeltas_total(pos,3)])-0.1 max([targets_total(pos,3); cdeltas_total(pos,3)])+0.1]);
title('test result using training data');

fprintf('average loss is %.4f\n', mean(loss(loss>0)));