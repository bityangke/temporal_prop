classdef TROIPooling < dagnn.Layer
    % DAGNN.ROIPOOLING  Temporal Region of interest pooling layer

    % Copyright (C) 2016 Hakan Bilen.
    % All rights reserved.
    %
    % This file is part of the VLFeat library and is made available under
    % the terms of the BSD license (see the COPYING file).

    properties
      method = 'max'
      subdivisions = [7 30]
      transform = 1
      flatten = false
    end

    methods
        function outputs = forward(obj, inputs, params)
            numROIs = numel(inputs{2})/3;

%             outputs{1} = vl_nnroipool(...
%               inputs{1}, inputs{2}, ...
%               'subdivisions', obj.subdivisions, ...
%               'transform', obj.transform, ...
%               'method', obj.method) ;
%             outputs{1} = temporal_roi_pooling(inputs{1}, inputs{2}, obj.subdivisions);
            
            if obj.flatten
                outputs{1} = reshape(outputs{1},1,1,[],numROIs) ;
            end
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            numROIs = numel(inputs{2}) / 5 ;
            if obj.flatten
              % unflatten
                derOutputs{1} = reshape(...
                  derOutputs{1},obj.subdivisions(1),obj.subdivisions(2),[],numROIs) ;
            end
            derInputs{1} = vl_nnroipool(...
              inputs{1}, inputs{2}, derOutputs{1}, ...
              'subdivisions', obj.subdivisions, ...
              'transform', obj.transform, ...
              'method', obj.method) ;
            derInputs{2} = [];
            derParams = {} ;
        end

        function outputSizes = getOutputSizes(obj, inputSizes)
            if isempty(inputSizes{1})
                n = 0 ;
            else
                n = prod(inputSizes{2})/5 ;
            end
            outputSizes{1} = [obj.subdivisions, inputSizes{1}(3), n] ;
        end

        function output = temporal_roi_pooling(feature_maps, rois, subdiv)
            stride = round(size(feature_maps,1)/subdiv(1)); % non-integer rounding issue should be resolved
            poolsize = stride;
            for i=1:size(feature_maps,4)
                %%% just sub-sample over temporal axis
                % one simple version: just using random subsampling
                cur_rois = rois(2:3,rois(1,:)==i);
                sub_sampled = arrayfun(@(x,y)(get_feature(feature_maps(:,:,:,i), x, y, subdiv(2))), cur_rois(1,:), cur_rois(2,:));
               
                %%% max pooling over spatial axis
                output(:,:,:,i) = vl_nnpool(sub_sampled, [poolsize 1], ...
                                     'pad', [0 1 0 0], ...
                                     'stride', [stride 1], ...
                                     'method', 'max') ;
            end
        end
        
        function output = get_feature(feature_map, point_start, point_end, numGrid)
            out_feat = feature_map(:, point_start:point_end, :);
            N = size(out_feat,2);
            randInd = randi(N, [numGrid, 1]);
            randInd = sort(randInd);
            output = out_feat(:,randInd,:);
        end
        
%         function output = column_max_pooling(feature_maps)
%             
%         end
        
        function obj = TROIPooling(varargin)
            obj.load(varargin) ;
        end
    end
end
