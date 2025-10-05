classdef Logger < handle
    % A logging class for MATLAB.
 
    properties
        name = 'root';
        level = 10;  % Default: DEBUG
        % Maps log level names to numerical values.
        levels = struct('TRACE',5,'DEBUG',10,'INFO',20,'SUCCESS',25,...
            'WARNING',30,'ERROR',40,'CRITICAL',50);
        sinks = {}; % Stores all log destinations (console or file)
        extra_context; % A map to store extra key-value contextual data
    end

    properties (Constant, Access = private)
        % Maps log levels to cprintf colors
        COLORS = containers.Map(...
            {'TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'},...
            {'black', 'blue', '_green', '*green', '_yellow', 'red', '*red'});
    end

    methods
        function obj = Logger(name)
	        % Initializes the logger and adds a default console handler
            if nargin > 0 
                obj.name = name;
            end  
            % Creates a new, unique map to store contextual data 
            obj.extra_context = containers.Map();
        end

        function setLevel(obj, levelName)
	        % Sets the logger's minimum logging level.
            levelName = upper(levelName);
            if isfield(obj.levels, levelName)
                obj.level = obj.levels.(levelName);
            else
                warning('Invalid log level: %s. Level not changed.', levelName);
            end
        end

        function addConsoleHandler(obj, varargin) 
	        % Configures a console sink. 
            p = inputParser; % To handle optional parameters
            addParameter(p, 'level', 'DEBUG', @ischar);
            addParameter(p, 'use_colors', false, @islogical);
            parse(p, varargin{:}); % Match the incoming arguments against the defined parameters
            
            % Warn the user if color output is requested but cprintf is not available
            if p.Results.use_colors && isempty(which('cprintf'))
                warning('Logger:cprintf_missing', ...
                    ['''use_colors'' is enabled but cprintf was not found on the MATLAB path.\n' ...
                     'Colored output will be disabled. Download cprintf from the MATLAB File Exchange for this functionality.']);
            end

	        % Creates a struct to represent the console sink
            h.type = 'console';
            h.level = obj.levels.(upper(p.Results.level));
            h.use_colors = p.Results.use_colors;
	        % Appends the new sink to the list of sinks
            obj.sinks{end+1} = h;
        end
        
        function addFileHandler(obj, filename, varargin)
	        % Configures a file sink
            p = inputParser;
            addParameter(p, 'level', 'DEBUG', @ischar);
            addParameter(p, 'rotation', [], @(x) ischar(x) || isnumeric(x));
            parse(p, varargin{:});
	    
	        % Creates a struct to represent the file sink
            h.type = 'file';
            h.level = obj.levels.(upper(p.Results.level));
            h.filename = filename;
            h.rotation = p.Results.rotation;

            % Tries to open the file in 'append' mode
            [fh, msg] = fopen(filename, 'a');
	        % Throws an error if the file can't be opened
            if fh == -1
                error('Cannot open log file: %s. Reason: %s', filename, msg);
            end
            h.fh = fh;
            obj.sinks{end+1} = h;
        end

        function log(obj, levelName, msg, varargin)
            % Processes the message and sends it to all configured sinks
            levelName = upper(levelName);
            % Validate the log level name
            if ~isfield(obj.levels, levelName)
                error('Unknown log level: %s', levelName);
            end

            lvlNum = obj.levels.(levelName);

            % Format the message with sprintf if additional arguments are provided
            if ~isempty(varargin)
                try
                    message_in = sprintf(msg, varargin{:});
                catch ME
                    % If formatting fails, issue a warning and use the raw message
                    warning('Logger:sprintf_error', 'Failed to format log message! Reason: [%s] %s', ME.identifier, ME.message);
                    message_in = msg; 
                end
            else
                message_in = msg;
            end

            
            % Retrieves info where the log call originated
            caller = obj.getCallerInfo();
            % Retrieves the context and formats it
            ctx = obj.formatContext();
           
            % Appends context to the message if it exists
            if strlength(ctx) > 0
                message_out = message_in + " | " + ctx;
            else
                message_out = message_in;
            end

            % Iterates through each configured sink
            for k = 1:numel(obj.sinks)
                sink = obj.sinks{k};

                % If the message level is lower than the
                % sink's minimum level, skip this sink
                if lvlNum < sink.level
                    continue;
                end

                % Console and file formatting
                out = sprintf('%s | %-8s | %s - %s', ...
                      string(datetime('now'), 'yyyy-MM-dd HH:mm:ss.SSS'), ...
                      levelName, caller, message_out);
                
                if strcmp(sink.type, 'console')
                    
                    % Use cprintf only if use_colors is true and cprintf is available 
                    if sink.use_colors && ~isempty(which('cprintf'))
                        color = obj.COLORS(levelName);
                        % Substitute '%' with '%%' and force cprinf to print %
                        out_for_cprintf = strrep(out, '%', '%%');
                        cprintf(color, out_for_cprintf);
    
                    else
                        % Print without colors
                        fprintf(out);
                        
                    end
                    % Add a newline character after printing
                    fprintf('\n');

                elseif strcmp(sink.type, 'file')

                    % Write the formatted string to the file, adding a newline
                    fprintf(sink.fh, '%s\n', out);
                    
                    % Handles file rotation logic immediately after writing
                    if ~isempty(sink.rotation)
                        obj.handleRotation(sink);
                    end
                end
            end
        end

        % Wrapper methods for each log level
        function trace(obj, msg, varargin), obj.log('TRACE', msg, varargin{:}); end
        function debug(obj, msg, varargin), obj.log('DEBUG', msg, varargin{:}); end
        function info(obj, msg, varargin), obj.log('INFO', msg, varargin{:}); end
        function success(obj, msg, varargin), obj.log('SUCCESS', msg, varargin{:}); end
        function warn(obj, msg, varargin), obj.log('WARNING', msg, varargin{:}); end
        function error(obj, msg, varargin), obj.log('ERROR', msg, varargin{:}); end
        function critical(obj, msg, varargin), obj.log('CRITICAL', msg, varargin{:}); end

        % Methods for adding and clearing contextual data
        function addContext(obj, key, value)
            % Adds or updates a key-value pair in the context map
            obj.extra_context(key) = value;
        end

        function clearContext(obj)
            % Resets the entire context map
            obj.extra_context = containers.Map();
        end

        % Class destructor
        function delete(obj)
            % Iterates through all sinks to ensure 
            % file handles are closed properly
            for k = 1:numel(obj.sinks)
                h = obj.sinks{k};
                if isfield(h, 'fh') && h.fh ~= -1
                    fclose(h.fh);
                end
            end
        end
    end
    
    % Private methods
    methods (Access = private)
        function callerStr = getCallerInfo(~)
            % Gets the caller's file, function, and line 
            % number from the call stack. 
            st = dbstack(3);  % It ignores this method, the log method and 
                              % the wrapper method, pointing to user's code. 
            
            
            % Handle cases where the call stack is not deep enough 
            % (e.g., called from command window)
            if numel(st) < 1
                callerStr = '__main__:<module>:?';
                return;
            end
            [~, fname, ~] = fileparts(st(1).file);
            func = st(1).name;
            if isempty(func)
                func = '<module>';
            end
            callerStr = sprintf('%s:%s:%d', fname, func, st(1).line);
        end
        
        function ctx = formatContext(obj)
            % Formats the extra_context map into a single string
            keys = obj.extra_context.keys;
            % Return an empty string if there is no context to format
            if isempty(keys)
                ctx = "";
                return;
            end
            % Pre-allocate a cell array to hold the formatted key-value strings
            parts = cell(1, length(keys));
            % Iterate through the keys and format each key-value pair
            for i = 1:length(keys)
                k = keys{i};
                v = obj.extra_context(k);
                parts{i} = sprintf("%s=%s", k, string(v));
            end
            % Join all parts with a separator
            ctx = strjoin(parts, " | ");
        end

        function handleRotation(~, sink)
            % Manages file rotation based on file size
            if isnumeric(sink.rotation)
                file_info = dir(sink.filename);
                % Checks if the file exists and if its size exceeds the rotation limit
                if ~isempty(file_info) && file_info.bytes > sink.rotation
                    % Closes the current file handle
                    fclose(sink.fh);
                    % Renames the old file adding a timestamp
                    timestamp = string(datetime('now'),'yyyyMMdd_HHmmss');
                    [path, name_only, ext] = fileparts(sink.filename);
                    new_name = fullfile(path, sprintf('%s_%s%s', name_only, timestamp, ext));
                    movefile(sink.filename, new_name);
                    
                    % Re-opens a new, empty file with the original name
                    [fh, ~] = fopen(sink.filename, 'a');
                    if fh == -1
                        warning('Failed to re-open log file after rotation.');
                    end
                    % Update the sink with the new file handle
                    sink.fh = fh;
                end
            end
        end

    end

end