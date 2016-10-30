function [starts, durations] = generate_temporal_proposal(video, grid_size)
% Input:
%         video    : 1xT cell array contains video frames
%         grid_size: length of grid - ex) 100frames, 200frames
% Output:
%         starts   : 1xN vector containing N temporal action starting points
%         durations: 1xN vector containing N temporal action durations

num_frms = length(video);

N = ceil(num_frms/grid_size);

starts    = 1:grid_size:num_frms;
durations = ones(1,N)*grid_size;
% durations(end) = mod(num_frms, grid_size);

% ignore the last grid if it's size is differnt to the other grids
if mod(num_frms, grid_size) ~= 0
    starts = starts(1:end-1);
    durations = durations(1:end-1);
end