function loss_val = calculate_smoothL1(pred_delta, gt_delta)
%Input:
%       pred_delta : Nx4 prediction target vector
%       gt_delta   : Nx4 ground truth target vector
%Output: 
%       loss_val : smooth L1 loss value

abs_diff  = abs(pred_delta-gt_delta);
case1vals = abs_diff(abs_diff<1 );
case2vals = abs_diff(abs_diff>=1);
case1L1val = 0.5*case1vals.^2;
case2L1val = case2vals - 0.5;
loss_val =  sum(case1L1val(:)) + sum(case2L1val(:));


% N = size(pred_delta,1);
% loss_val = 0;
% for j=1:N
%     for i=1:size(pred_delta,2)
%         abs_diff = abs(pred_delta(j,i) - gt_delta(j,i));
%         if abs_diff < 1
%             loss_val = loss_val + 0.5*abs_diff^2;
%         else
%             loss_val = loss_val - abs_diff - 0.5;
%         end
%     end
% end