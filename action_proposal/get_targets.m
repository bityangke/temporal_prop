function targets = get_targets(rois, gt)
% Inputs: 
%          rois : Nx4 matrix, each row contains bounding box proposal coordinate (Px1,Py1,Px2,Py2)
%          gt   : 1x4 (Gx1,Gy1,Gx2,Gy2)
% Outputs: 
%          targets: Nx4 regression target matrix, each row contains, 
%                          tx = (Gx(i) - Px(i))/Pw(i)
%                          ty = (Gy(i) - Py(i))/Pw(i)
%                          tw = log(Gw(i)/Pw(i))
%                          th = log(Gh(i)/Ph(i))

N = size(rois,1);

Gw = double(gt(3)-gt(1)+1);
Gh = double(gt(4)-gt(2)+1);
Gcx = double(gt(1) + 0.5*Gw);
Gcy = double(gt(2) + 0.5*Gh);

for i=1:N
    Pw = double(rois(i,3)-rois(i,1)+1);
    Ph = double(rois(i,4)-rois(i,2)+1);
    Pcx = double(rois(i,1) + 0.5*Pw);
    Pcy = double(rois(i,2) + 0.5*Ph);
    targets(i,:) = [double(Gcx-Pcx)/double(Pw), double(Gcy-Pcy)/double(Ph), log(double(Gw)/double(Pw)), log(double(Gh)/double(Ph))];
end