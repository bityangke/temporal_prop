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
      subsample_indice = {}
    end

    methods
        function outputs = forward(obj, inputs, params)
            numROIs = numel(inputs{2})/3;
            obj.subsample_indice = cell(size(inputs{1},4),1);
            
            outputs{1} = obj.temporal_roi_pooling(inputs{1}, inputs{2}, obj.subdivisions);
            
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
            % TO DO: let the gradient is bypassed only for the maximum and subsampled locations
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

        function output = temporal_roi_pooling(obj, feature_maps, rois, subdiv)
            stride = round(size(feature_maps,1)/subdiv(1)); % non-integer rounding issue should be resolved
            poolsize = stride;
            output = [];
            for i=1:size(feature_maps,4)
                %%% just sub-sample over temporal axis
                % one simple version: just using random subsampling
                cur_rois = rois([2 4],rois(1,:)==i);
                sub_sampled = cell(1,size(cur_rois,2));
                [sub_sampled, randInd] = arrayfun(@(x,y)(obj.get_feature(feature_maps(:,:,:,i), x, y, subdiv(2))), ...
                                                            cur_rois(1,:), cur_rois(2,:));
%                 [sub_sampled, randInd] = arrayfun(@(x,y)(obj.get_feature(feature_maps(:,:,:,i), x, y, subdiv(2))), ...
%                                        cur_rois(1,:), cur_rois(2,:), 'UniformOutput', false);
                obj.subsample_indice{i} = randInd;
                %%% max pooling over spatial axis
                % TO DO: store the max indice
                pooled = cellfun(@(x)(vl_nnpool(x, [poolsize 1],'pad', [0 1 0 0], 'stride', [stride 1], 'method', 'max')), ...
                                                  sub_sampled, 'UniformOutput', false);
                cur_output = cat(4,pooled{:});
                output = cat(4, output, cur_output);
            end
        end
        
%           function [output] = get_feature(obj, feature_map, point_start, point_end, numGrid)
        function [output, randInd] = get_feature(obj, feature_map, point_start, point_end, numGrid)
            out_feat = feature_map(:, point_start:point_end, :);
            N = size(out_feat,2);
            selected = randi(N, [numGrid, 1]);
            selected = sort(selected);  % TO DO: convert this to cell output
            
            dim = size(out_feat(:,selected,:));
            output = mat2cell(out_feat(:,selected,:), dim(1), dim(2), dim(3)); % TO DO: check this is working or not
            randInd = mat2cell(selected, numGrid, 1);
%             output = out_feat(:,randInd,:);
        end
        
        function obj = TROIPooling(varargin)
            obj.load(varargin) ;
        end
    end
end
