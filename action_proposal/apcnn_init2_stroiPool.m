function net = apcnn_init2(varargin)
% Action Proposal (AP) Initialization NN: modifed from the original FAST_RCNN_INIT 
% modified by Jinwoo Choi, 2016.
%%% AP CNN using the output of roi_pool layer's out as 
%%% an input of pred_bbox layer
%%%
%FAST_RCNN_INIT  Initialize a Fast-RCNN
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.piecewise = 1;
opts.modelPath = fullfile('..','models','imagenet-vgg-verydeep-16.mat');
opts = vl_argparse(opts, varargin) ;
display(opts) ;

% Load an imagenet pre-trained cnn model.
net = load(opts.modelPath);
net = vl_simplenn_tidy(net);

% Add drop-out layers.
relu6p = find(cellfun(@(a) strcmp(a.name, 'relu6'), net.layers)==1);
relu7p = find(cellfun(@(a) strcmp(a.name, 'relu7'), net.layers)==1);

drop6 = struct('type', 'dropout', 'rate', 0.5, 'name','drop6');
drop7 = struct('type', 'dropout', 'rate', 0.5, 'name','drop7');
net.layers = [net.layers(1:relu6p) drop6 net.layers(relu6p+1:relu7p) drop7 net.layers(relu7p+1:end)];

% Change loss for FC layers.
nCls = 2;   % action + background
fc8p = find(cellfun(@(a) strcmp(a.name, 'fc8'), net.layers)==1);
net.layers{fc8p}.name = 'predcls';
net.layers{fc8p}.weights{1} = 0.01 * randn(1,1,size(net.layers{fc8p}.weights{1},3),nCls,'single');
net.layers{fc8p}.weights{2} = zeros(1, nCls, 'single');

% Skip pool5.
pPool5 = find(cellfun(@(a) strcmp(a.name, 'pool5'), net.layers)==1);
net.layers = net.layers([1:pPool5-1,pPool5+1:end-1]);

% Convert to DagNN.
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% Add ROIPooling layer.  -> should be modified or removed
vggdeep = false;
pRelu5 = find(arrayfun(@(a) strcmp(a.name, 'relu5'), net.layers)==1);
if isempty(pRelu5)
  vggdeep = true;
  pRelu5 = find(arrayfun(@(a) strcmp(a.name, 'relu5_3'), net.layers)==1);
  if isempty(pRelu5)
    error('Cannot find last relu before fc');
  end
end

%----------------------------------------------------------------------%
% remove and construct a new FC6 layer
%----------------------------------------------------------------------%
tempPoolSize = 3;
% store fc6 parameters
pFc6f = (arrayfun(@(a) strcmp(a.name, 'fc6f'), net.params)==1);
pFc6b = (arrayfun(@(a) strcmp(a.name, 'fc6b'), net.params)==1);
fc6f_params_pre = net.params(pFc6f).value;
fc6b_params_pre = net.params(pFc6b).value;
fc6f_params_pre = reshape(fc6f_params_pre, [49 1 512 4096]);

% remove the current fc6 layer
net.removeLayer('fc6');

% add a customized fc6 layer
net.addLayer('fc6', dagnn.Conv('size', [49 tempPoolSize 512 4096], 'hasBias', true, 'stride', [1 1], 'pad', [0 0 0 0], 'dilate', [1 1]), {'xRP'}, {'x31'}, {'fc6f', 'fc6b'});

pRelu6 = (arrayfun(@(a) strcmp(a.name, 'relu6'), net.layers)==1);
pFc6 = (arrayfun(@(a) strcmp(a.name, 'fc6'), net.layers)==1);
net.layers(pRelu6).inputs{1} = net.layers(pFc6).outputs{1};

pFc6f = (arrayfun(@(a) strcmp(a.name, 'fc6f'), net.params)==1);
pFc6b = (arrayfun(@(a) strcmp(a.name, 'fc6b'), net.params)==1);
% duplicate parameters for temporal dimensions
for i=1:tempPoolSize
    tmp(:,i,:,:) = fc6f_params_pre;
end
net.params(pFc6f).value = tmp;
net.params(pFc6b).value = fc6b_params_pre;
%----------------------------------------------------------------------%
%    constructing a new FC6 layer done
%----------------------------------------------------------------------%

%----------------------------------------------------------------------%
% add a customized roi pooling layer
%----------------------------------------------------------------------%
pFc6 = (arrayfun(@(a) strcmp(a.name, 'fc6'), net.layers)==1);
if vggdeep
      net.addLayer('stroipool', dagnn.TROIPooling('method','max','transform',1/16,...
    'subdivisions',[49,tempPoolSize],'flatten',0), ...
    {'input','rois'}, 'xRP');
else
  net.addLayer('stroipool', dagnn.ROIPooling('method','max','transform',1/16,...
    'subdivisions',[6,6],'flatten',0), ...
    {'input','rois'}, 'xRP');
end
%----------------------------------------------------------------------%
% add a customized roi pooling layer done
%----------------------------------------------------------------------%
% pRP = (arrayfun(@(a) strcmp(a.name, 'stroipool'), net.layers)==1);
% net.layers(pFc6).inputs{1} = net.layers(pRP).outputs{1};

% Feed output of the drop6 to input of the predcls
pPredcls = (arrayfun(@(a) strcmp(a.name, 'predcls'), net.layers)==1);
pDrop6 = (arrayfun(@(a) strcmp(a.name, 'drop6'), net.layers)==1);
net.layers(pPredcls).inputs{1} = net.layers(pDrop6).outputs{1};

% Add softmax loss layer.
pFc8 = (arrayfun(@(a) strcmp(a.name, 'predcls'), net.layers)==1);
net.addLayer('losscls',dagnn.Loss(), ...
  {net.layers(pFc8).outputs{1},'label'}, ...
  'losscls',{});

% Add bbox regression layer.
if opts.piecewise
%   pparFc8 = (arrayfun(@(a) strcmp(a.name, 'predclsf'), net.params)==1);
%   pdrop6  = (arrayfun(@(a) strcmp(a.name, 'drop6'), net.layers)==1);
  pparRP  = (arrayfun(@(a) strcmp(a.name, 'predclsf'), net.params)==1);
  pRP = (arrayfun(@(a) strcmp(a.name, 'stroipool'), net.layers)==1);

  net.addLayer('predbbox',dagnn.Conv('size',[49 tempPoolSize 512 8],'hasBias', true), ...
    net.layers(pRP).outputs{1},'predbbox',{'predbboxf','predbboxb'});

  net.params(end-1).value = 0.001 * randn(49,tempPoolSize,512, 8,'single');
  net.params(end).value = zeros(1, 8,'single');

  net.addLayer('lossbbox',dagnn.LossSmoothL1(), ...
    {'predbbox','targets','instance_weights'}, ...
    'lossbbox',{});
end

% remove Conv layers
net.removeLayer('conv1_1');
net.removeLayer('conv1_2');
net.removeLayer('conv2_1');
net.removeLayer('conv2_2');
net.removeLayer('conv3_1');
net.removeLayer('conv3_2');
net.removeLayer('conv3_3');
net.removeLayer('conv4_1');
net.removeLayer('conv4_2');
net.removeLayer('conv4_3');
net.removeLayer('conv5_1');
net.removeLayer('conv5_2');
net.removeLayer('conv5_3');

net.removeLayer('relu1_1');
net.removeLayer('relu1_2');
net.removeLayer('relu2_1');
net.removeLayer('relu2_2');
net.removeLayer('relu3_1');
net.removeLayer('relu3_2');
net.removeLayer('relu3_3');
net.removeLayer('relu4_1');
net.removeLayer('relu4_2');
net.removeLayer('relu4_3');
net.removeLayer('relu5_1');
net.removeLayer('relu5_2');
net.removeLayer('relu5_3');

net.removeLayer('pool1');
net.removeLayer('pool2');
net.removeLayer('pool3');
net.removeLayer('pool4');

% remove FC7, relu7, drop7 layers
net.removeLayer('drop7');
net.removeLayer('relu7');
net.removeLayer('fc7');

net.rebuild();

% No decay for bias and set learning rate to 2
for i=2:2:numel(net.params)
  net.params(i).weightDecay = 0;
  net.params(i).learningRate = 2;
end

% slower learning rate for pred_box layer
net.params(end-1).learningRate = 0.5;

% Change image-mean as in fast-rcnn code
net.meta.normalization.averageImage = ...
  reshape([122.7717 102.9801 115.9465],[1 1 3]);

net.meta.normalization.interpolation = 'bilinear';

net.meta.classes.name = {'action', 'background' };
  
net.meta.classes.description = {};