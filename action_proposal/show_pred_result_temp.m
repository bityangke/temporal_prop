predMap = uint8(zeros(300, length(cnn_feat), 3));

for i=1:3
    predMap(1:50, labels.gt_start_frames(i):labels.gt_end_frames(i), :) = 255;
end

Gs_estimated = uint32(Gs_estimated);
Gl_estimated = uint32(Gl_estimated);

predMap( 70:100, Gs_estimated(2100):Gs_estimated(2100)+Gl_estimated(2100)-1, 1) = 255;
predMap( 130:160,  Gs_estimated(2800):Gs_estimated(2800)+Gl_estimated(2800)-1, 1) = 255;
predMap( 190:220,  Gs_estimated(4500):Gs_estimated(4500)+Gl_estimated(4500)-1, 1) = 255;