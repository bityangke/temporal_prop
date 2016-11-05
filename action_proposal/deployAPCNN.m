% --------------------------------------------------------------------
function net = deployAPCNN(net,imdb)
% --------------------------------------------------------------------
for l = numel(net.layers):-1:1
    if isa(net.layers(l).block, 'dagnn.Loss') || ...
        isa(net.layers(l).block, 'dagnn.DropOut')
    layer = net.layers(l);
    net.removeLayer(layer.name);
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
    end
end

net.rebuild();

pfc8 = net.getLayerIndex('predcls') ;
net.addLayer('probcls',dagnn.SoftMax(),net.layers(pfc8).outputs{1},...
  'probcls',{});

net.vars(net.getVarIndex('probcls')).precious = true ;

idxBox = net.getLayerIndex('predbbox');
if ~isnan(idxBox)
    net.vars(net.layers(idxBox).outputIndexes(1)).precious = true;
    % incorporate mean and std to bbox regression parameters
    blayer = net.layers(idxBox) ;
    filters = net.params(net.getParamIndex(blayer.params{1})).value;
    biases = net.params(net.getParamIndex(blayer.params{2})).value;

    boxMeans = single(imdb.boxes.bboxMeanStd{1}');
    boxStds = single(imdb.boxes.bboxMeanStd{2}');

    net.params(net.getParamIndex(blayer.params{1})).value = ...
    bsxfun(@times,filters,...
    reshape([boxStds(:)' zeros(1,4,'single')]',...
    [1 1 1 4*numel(net.meta.classes.name)]));

    biases = biases .* [boxStds(:)' zeros(1,4,'single')];

    net.params(net.getParamIndex(blayer.params{2})).value = ...
    bsxfun(@plus,biases, [boxMeans(:)' zeros(1,4,'single')]);
end

net.mode = 'test' ;