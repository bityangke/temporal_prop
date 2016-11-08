function [new_coord] = convert_xywh_to_x1y1x2y2(coord)
% Inputs:
%          coord: nx4 vector containing a bounding box coordinates x,y,w,h
% Outputs:
%          new coord: nx4 vector containing a bounding box coordinates
%          x1,y1,x2,y2. (x1,y1) is a top-left coordinate and (x2,y2) is a
%          bottom-right coordinate of the bounding box

new_coord(:,1) = coord(:,1); % x1
new_coord(:,2) = coord(:,2); % y1

width  = double(coord(:,3));
height = double(coord(:,4));

new_coord(:,3) = new_coord(:,1) + width - 1;
new_coord(:,4) = new_coord(:,2) + height - 1;