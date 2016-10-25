function files = clean_dir(base)
	files = dir(base);
	files_tmp = {};
	for i = 1:length(files)
		if strncmpi(files(i).name, '.',1) == 0
			files_tmp{length(files_tmp)+1} = files(i).name;
		end
	end
	files = files_tmp;
end
