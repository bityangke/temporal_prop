function [starts, durations] = generate_temporal_proposal2(numFrames)
% Input:
%         numFrames    : number of video frames
% Output:
%         starts   : 1xN vector containing N temporal action starting points with multiple %                    grid_sizes and strides
%         durations: 1xN vector containing N temporal action durations with multiple %                    grid_sizes and strides

% numFrames = length(video);
grid_bank   = [5, 10, 20, 30, 50, 70, 100, 200, 300, 500, 1000]; %[1, 2, 3];    % unit in frames
stride_bank = [1, 5, 10]; %[1, 2, 3];    % unit in frames

starts = [];
durations = [];

% loop over grid sizes
for i=1:length(grid_bank)
    grid_size = grid_bank(i);
    for j=1:length(stride_bank)
        stride = stride_bank(j);
%         M = floor((numFrames-grid_size)/stride)+1;

        current_starts    = uint32(1:stride:(numFrames-grid_size+1));
        M = numel(current_starts);
        current_durations = uint32(ones(1,M)*grid_size);

        if size(current_starts,2) ~= size(current_durations,2)
            fprintf('mismatch between size(current_starts,2) and size(current_durations,2) in F=%d, stride=%d \n', grid_size, stride);
            return
        end

        % attach the current grid size proposals array to the total proposals array
        starts = [starts, current_starts];
        durations = [durations, current_durations];
    end
end
