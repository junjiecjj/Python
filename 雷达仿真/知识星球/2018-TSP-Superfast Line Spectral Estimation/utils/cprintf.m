% Prints in specified column
function cprintf(column, string, varargin)
	% Move to column
	fprintf('\033[200D\033[%iC', column);
	% Print string
	fprintf(string, varargin{:});
end

