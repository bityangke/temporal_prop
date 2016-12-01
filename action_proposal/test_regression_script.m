% testing regression and classification
clear

run ../matlab/vl_setupnn;
addpath('./Union');
addpath('./range_intersection/');

opts.classes = {'action'} ;
opts.confThreshold = 0.5;
imdbDir    = fullfile('..', 'data', 'exp_20161117_LinearRegression_256sample_128pos_denser_window_tempPool3') ;
modelPath =  fullfile('..','data','exp_20161117_LinearRegression_256sample_128pos_denser_window_tempPool3', 'net-epoch-1.mat');
% imdbDir    = fullfile('..', 'data', 'exp_20161109_256sample_128pos_denser_window_tempPool3') ;
% modelPath =  fullfile('.','data','exp_20161104_1207', 'net-epoch-12.mat');
train_imdb_path = fullfile(imdbDir, 'imdb.mat');
% expDir    = fullfile(imdbDir, '1D_part');
% expDir    = fullfile(imdbDir, '1D_part_test_cheat');
opts.gpu = [];
use_norm = 0;

% Load train imdb
imdb = load(train_imdb_path) ;

% Load the network and put it in test mode.
net = load(modelPath);
net = dagnn.DagNN.loadobj(net.net);
net = deployAPCNN(net, imdb);
net.mode = 'test' ;

% % Load test GT imdb
% dataDir   = fullfile('..', '..', 'st-slice-cnn-tar', 'data', 'THUMOS14'); % for MacBook. modify this line to set up the data path
% imdb_test_GT = setup_ap_THUMOS14(dataDir, 0);

% Load training IMDB
imdb_test_GT = imdb;

targets_total = [];
cdeltas_total = [];
cumm_rois = [];
cumm_targets = [];
cprobs_total = [];

targets_total2 = [];
cdeltas_total2 = [];
cumm_rois2 = [];
cumm_targets2 = [];
cprobs_total2 = [];
loss = zeros(numel(imdb.images.feature_path),1);
loss2 = zeros(numel(imdb.images.feature_path),1);

for test_video=1:numel(imdb.images.feature_path) 
    close all;
    fprintf('Testing... %d/%d\n',test_video, numel(imdb.images.feature_path));
    % Load a test image(feature) and candidate bounding boxes.
    %     load(fullfile(expDir, sprintf('imagenet-vgg_relu5_on_THUMOS14val_%d_1D.mat',test_video)));
    if imdb.images.set(test_video) == 2    
        load(imdb.images.feature_path{test_video});
        oneD_converted_feat = current_GT_1D_feat;
    else
        continue;
    end
    
    bbox_label = imdb_test_GT.images.labels{test_video};
    if imdb_test_GT.images.labels{test_video}.num_frames ~= size(oneD_converted_feat,2)
        fprintf('Mismatch between gt bbox and the video feature!\n');
    end

    % construct a gt bbox: (x,y,w,h)
    gt_bbox = [];
    for i=1:numel(bbox_label.gt_start_frames)
        gt_bbox(i,1) = bbox_label.gt_start_frames(i); % x
        gt_bbox(i,2) = 1; % y
        gt_bbox(i,3) = bbox_label.gt_end_frames(i)-bbox_label.gt_start_frames(i)+1; % width
        gt_bbox(i,4) = size(oneD_converted_feat,1); % height
    end
    gt_bbox2 = convert_xywh_to_x1y1x2y2(gt_bbox);

    count = 1;
    % construct proposal boxes around GT
    p_bbox = [];    targets = [];
    p_bbox2 = [];   targets2 = [];
    for i=1:size(gt_bbox,1)
        d = double(gt_bbox(i,3));
        number = round(rand(1));
        if number==1
            direction = 1;
        else
            direction = -1;
        end
%         shift = d*0.5*direction;
        shift = d*0.05;

%         scale = 2*rand(1);
        scale  = 0.9;
        scale2 = 1.1;
        if scale > 1 
            sc_shift = round(-d*0.3);
        else
            sc_shift = round(d*0.3);
        end

        p_bbox(:,count)   = [1 gt_bbox(i,1)       gt_bbox(i,2) gt_bbox(i,3) gt_bbox(i,4)]'; % original GT
        p_bbox(:,count+1) = [1 gt_bbox(i,1)-shift gt_bbox(i,2) gt_bbox(i,3) gt_bbox(i,4)]'; % shift
        p_bbox(:,count+2) = [1 gt_bbox(i,1)+shift gt_bbox(i,2) gt_bbox(i,3) gt_bbox(i,4)]'; % shift opposite direction
                
        p_bbox(2:5,count:count+2) = convert_xywh_to_x1y1x2y2(p_bbox(2:5,count:count+2)')';
        targets = vertcat(targets, get_targets(p_bbox(2:5,count:count+2)', gt_bbox2(i,:)));
        
        p_bbox2(:,count)   = [1 gt_bbox(i,1) gt_bbox(i,2) gt_bbox(i,3)        gt_bbox(i,4)]'; % original GT
        p_bbox2(:,count+1) = [1 gt_bbox(i,1) gt_bbox(i,2) gt_bbox(i,3)*scale  gt_bbox(i,4)]'; % enlarge
        p_bbox2(:,count+2) = [1 gt_bbox(i,1) gt_bbox(i,2) gt_bbox(i,3)*scale2 gt_bbox(i,4)]'; % shrink
        
        p_bbox2(2:5,count:count+2) = convert_xywh_to_x1y1x2y2(p_bbox2(2:5,count:count+2)')';
        targets2 = vertcat(targets2, get_targets(p_bbox2(2:5,count:count+2)', gt_bbox2(i,:)));
        
        count = count + 3;
    end
    rois = single(p_bbox);
    cumm_rois = vertcat(cumm_rois, rois(2:5,:)');
    cumm_targets = vertcat(cumm_targets, targets);
    
    rois2 = single(p_bbox2);
    cumm_rois2 = vertcat(cumm_rois2, rois2(2:5,:)');
    cumm_targets2 = vertcat(cumm_targets2, targets2);
    
    % roi normalization
    if use_norm == 1
        means = imdb.boxes.bboxMeanStd{1};
        stds  = imdb.boxes.bboxMeanStd{2};
        norm_rois = [];
        for i=1:size(rois,1);
            norm_rois = bsxfun(@minus, rois(2:5,:)', means);
            norm_rois(:,1) = bsxfun(@rdivide, norm_rois(:,1), stds(:,1));
            norm_rois(:,3) = bsxfun(@rdivide, norm_rois(:,3), stds(:,3));
        end
        rois(2:5,:) = norm_rois';
        norm_rois2 = [];
        for i=1:size(rois,1);
            norm_rois2 = bsxfun(@minus, rois2(2:5,:)', means);
            norm_rois2(:,1) = bsxfun(@rdivide, norm_rois2(:,1), stds(:,1));
            norm_rois2(:,3) = bsxfun(@rdivide, norm_rois2(:,3), stds(:,3));
        end
        rois2(2:5,:) = norm_rois2';
    end
    
    % Evaluate network either on CPU or GPU.
    if numel(opts.gpu) > 0
        gpuDevice(opts.gpu) ;
        oneD_converted_feat = gpuArray(oneD_converted_feat) ;
        rois = gpuArray(rois) ;
        rois2 = gpuArray(rois2) ;
        net.move('gpu') ;
    end

    % network evaluation !
    net.conserveMemory = false ;
    net.eval({'input', oneD_converted_feat, 'rois', rois});

    % Extract class probabilities and  bounding box refinements
    probs = squeeze(gather(net.vars(net.getVarIndex('probcls')).value));
    deltas = squeeze(gather(net.vars(net.getVarIndex('predbbox')).value));

    % network evaluation !
    net.conserveMemory = false ;
    net.eval({'input', oneD_converted_feat, 'rois', rois2});

    % Extract class probabilities and  bounding box refinements
    probs2 = squeeze(gather(net.vars(net.getVarIndex('probcls')).value));
    deltas2 = squeeze(gather(net.vars(net.getVarIndex('predbbox')).value));
    
    % evaluate (and visualize) results 
    eval_rois = rois(2:5,:)';
    c = 1; 
    cprobs = probs(c,:) ;
    cdeltas = deltas(4*(c-1)+(1:4),:)' ;
    cboxes = bbox_transform_inv(rois(2:5,:)', cdeltas);
    cls_dets = [cboxes cprobs'] ;

    eval_rois2 = rois2(2:5,:)';
    c = 1; 
    cprobs2 = probs2(c,:) ;
    cdeltas2 = deltas2(4*(c-1)+(1:4),:)' ;
    cboxes2 = bbox_transform_inv(rois2(2:5,:)', cdeltas2);
    cls_dets2 = [cboxes2 cprobs2'] ;
    
%     scatter(targets(:,1), targets(:,3), 'r');
%     hold on;
%     scatter(cdeltas(:,1), cdeltas(:,3), 'g');
%     xlabel('dx');
%     ylabel('dw');
%     legend('target', 'prediction');
%     axis([min([targets(:,1);cdeltas(:,1)])-0.1 max([targets(:,1); cdeltas(:,1)])+0.1 min([targets(:,3);cdeltas(:,3)])-0.1 max([targets(:,3); cdeltas(:,3)])+0.1]);

    targets_total = vertcat(targets_total, targets);
    cdeltas_total = vertcat(cdeltas_total, cdeltas);
    cprobs_total  = vertcat(cprobs_total, cprobs(:));
    
    loss(test_video) = calculate_smoothL1(cdeltas, targets);
    
    targets_total2 = vertcat(targets_total2, targets2);
    cdeltas_total2 = vertcat(cdeltas_total2, cdeltas2);
    cprobs_total2  = vertcat(cprobs_total2, cprobs2(:));
    
    loss2(test_video) = calculate_smoothL1(cdeltas2, targets2);

%     % loop over GT
%     for i=1:size(gt_bbox,1)
%         current_gt = [bbox_label.gt_start_frames(i), bbox_label.gt_end_frames(i)];
%         for j=1:4
%             rois_ind = (i-1)*4+j;
%             org  = [eval_rois(rois_ind,1), eval_rois(rois_ind,3)];
%             pred = [cboxes(rois_ind,1), cboxes(rois_ind,3)];
%             iou_org{test_video}(i,j) = calculateIoU(org, current_gt);
%             iou_pred{test_video}(i,j) = calculateIoU(pred, current_gt);
%         end
%     end
%     
%     fprintf('test video %d\n', test_video);
%     fprintf('mean IoU of original Proposal: %f\n', mean(iou_org{test_video}(:)));
%     fprintf('mean IoU of regressed Proposal: %f\n', mean(iou_pred{test_video}(:)));
end

figure
scatter(targets_total(:,1), targets_total(:,3), 'r');
hold on;
scatter(cdeltas_total(:,1), cdeltas_total(:,3), 'g');
xlabel('dx');
ylabel('dw');
legend('target', 'prediction');
axis([min([targets_total(:,1);cdeltas_total(:,1)])-0.1 max([targets_total(:,1); cdeltas_total(:,1)])+0.1 min([targets_total(:,3);cdeltas_total(:,3)])-0.1 max([targets_total(:,3); cdeltas_total(:,3)])+0.1]);
title('shift only');

fprintf('average loss is %.4f\n', mean(loss((loss>0))));


figure
scatter(targets_total2(:,1), targets_total2(:,3), 'r');
hold on;
scatter(cdeltas_total2(:,1), cdeltas_total2(:,3), 'g');
xlabel('dx');
ylabel('dw');
legend('target', 'prediction');
axis([min([targets_total2(:,1);cdeltas_total2(:,1)])-0.1 max([targets_total2(:,1); cdeltas_total2(:,1)])+0.1 min([targets_total2(:,3);cdeltas_total2(:,3)])-0.1 max([targets_total2(:,3); cdeltas_total2(:,3)])+0.1]);
title('scale only');

fprintf('average loss is %.4f\n', mean(loss((loss>0))));