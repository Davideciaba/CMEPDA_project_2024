classdef Logger < handle
% LOGGER A logging class for MATLAB.
%   Supports multiple log levels, console and file outputs,
%   colored output (if cprintf is available in MATLAB), 
%   and contextual data.
%
% KEY FEATURES:
%   - Log Levels: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR,
%     and CRITICAL.
%   - Multiple Handlers: Configure output to the console, files, or both
%     simultaneously.
%   - Colored Output: Provides colored logs in the MATLAB Desktop (requires
%     'cprintf.p').
%     If colors look incorrect (e.g., red "bleeding"), it may depend on the MATLAB version.
%     We observed bleeding in MATLAB R2024b, but not in R2025b.
%   - File Rotation: Automatically manages the size of log files,
%     archiving old ones to prevent them from growing indefinitely.
%   - Contextual Data: Adds metadata (e.g., user ID, session name) to all
%     log messages.
    
    properties (Access = public)
        name string = "root";
        level (1,1) double = 10;  % Default: DEBUG
        % Maps log level names to numerical values.
        levels = struct('TRACE',5,'DEBUG',10,'INFO',20,'SUCCESS',25,...
            'WARNING',30,'ERROR',40,'CRITICAL',50);
        sinks cell = {}; % Stores all log destinations (console or file)
        extraContext; % A map to store extra key-value contextual data
    end

    properties (Constant, Access = private)
        % Maps log levels to cprintf colors
        COLORS = containers.Map(...
            {'TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'},...
            {'*cyan', '*blue', '*black', '*green', '*yellow', '*errors', '*red'});
        
    end
    
    methods (Access = public)
        
        function obj = Logger(name)
             % PURPOSE: Initializes a new Logger instance.
            arguments
                name (1,1) string = "root"
            end
            obj.name = name;
            % Creates a new, unique map to store contextual data 
            obj.extraContext = containers.Map();
        end
        
        function setLevel(obj, levelName)
	        % PURPOSE: Adjusts the global minimum severity threshold.
            arguments
                obj (1,1) Logger
                levelName (1,1) string
            end
            levelName = upper(levelName);
            if isfield(obj.levels, char(levelName))
                obj.level = obj.levels.(char(levelName));
            else
                warning('Invalid log level: %s. Level not changed.', levelName);
            end
        end
        
        function addConsoleHandler(obj, options) 
	        % PURPOSE: Attaches the Command Window as a log sink.
            arguments
                obj (1,1) Logger
                options.level (1,1) string = "DEBUG"
                options.useColors (1,1) {mustBeA(options.useColors, 'logical')} = false
            end

            % Warn the user if color output is requested but cprintf is not available
            if options.useColors && exist('cprintf', 'file') == 0
                warning('Logger:cprintf_missing', ...
                    ['''useColors'' is enabled but cprintf was not found on the MATLAB utils directory.\n' ...
                     'Colored output will be disabled. Download cprintf from the MATLAB File Exchange for this functionality.']);
                
            end

	        % Creates a struct to represent the console sink
            h.type = 'console';
            h.level = obj.levels.(upper(options.level));
            h.useColors = options.useColors;
	        % Appends the new sink to the list of sinks
            obj.sinks{end+1} = h;
        end
        
        function addFileHandler(obj, filename, options)
	        % PURPOSE: Attaches a file on disk as a log sink.
            arguments
                obj (1,1) Logger
                filename (1,1) string {mustBeNonempty}
                options.level (1,1) string = "DEBUG"
                options.rotation {mustBeA(options.rotation, 'double'), mustBeScalarOrEmpty, mustBePositive} = []
            end

	        % Creates a struct to represent the file sink
            h.type = 'file';
            h.level = obj.levels.(char(upper(options.level)));
            h.filename = filename;
            h.rotation = options.rotation;

            % Tries to open the file in 'append' mode
            [fh, msg] = fopen(filename, 'a');
	        % Throws an error if the file can't be opened
            if fh == -1
                error('Cannot open log file: %s. Reason: %s', filename, msg);
            end
            h.fh = fh;
            obj.sinks{end+1} = h;
        end
        
        function log(obj, levelName, msg, args)
            % PURPOSE: Processes the message and sends it to all configured sinks
            arguments
                obj (1,1) Logger
                levelName (1,1) string
                msg (1,1) string
            end
            arguments (Repeating)
                args
            end
            levelName = upper(levelName);
            % Validate the log level name
            if ~isfield(obj.levels, char(levelName))
                error('Unknown log level: %s', levelName);
            end

            lvlNum = obj.levels.(char(levelName));

            % Format the message with sprintf if additional arguments are provided
            if ~isempty(args)
                try                    
                    % Escape '\' to prevent misinterpretation in sprintf
                    msg = strrep(msg, '\', '\\');
                    message_in = sprintf(msg, args{:});
                catch ME
                    % If formatting fails, issue a warning and use the raw message
                    warning('Logger:sprintf_error', 'Failed to format log message! Reason: [%s] %s', ME.identifier, ME.message);
                    message_in = msg; 
                end
            else
                message_in = msg;
            end
            
            
            % Retrieves info where the log call originated
            caller = string(obj.getCallerInfo());
            % Retrieves the context and formats it
            ctx = obj.formatContext();
           
            % Appends context to the message if it exists
            if strlength(ctx) > 0
                message_out = message_in + " | " + ctx;
            else
                message_out = message_in;
            end
            
            timestamp = string(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss.SSS"));

            % Iterates through each configured sink
            for k = 1:numel(obj.sinks)
                sink = obj.sinks{k};

                % If the message level is lower than the
                % sink's minimum level, skip this sink
                if lvlNum < sink.level
                    continue;
                end
                
                if sink.type == "console"
                     
                     if sink.useColors
                         % Print with colors using cprintf (if available)
                         
                         color = obj.COLORS(char(levelName));                         
                         cprintf('Text', '%s | ', timestamp);
                            
                         cprintf(color, ' %-7s ', char(levelName));
                            
                         cprintf('Text', ' | %s - ', caller);
                            
                         cprintf(color, '%s \n', message_out);

                         drawnow('update');
                         
                         
                    else
                         % Print without colors
                         fprintf('%s | %-8s | %s - %s\n', timestamp, levelName, caller, message_out);
                         drawnow('update');
                    end

                elseif sink.type == "file"

                    % Write to the file
                    fprintf(sink.fh, '%s | %-8s | %s - %s\n', timestamp, levelName, caller, message_out);

                    % Handles file rotation logic immediately after writing
                    if ~isempty(sink.rotation)
                        obj.handleRotation(k);
                    end
                end
            end
        end

        % --- Wrappers for each log level ---
        function trace(obj, msg, args)
            arguments
                obj, msg (1,1) string
            end
            arguments (Repeating)
                args
            end
            obj.log("TRACE", msg, args{:}); 
        end
        
        function debug(obj, msg, args)
            arguments
                obj, msg (1,1) string
            end
            arguments (Repeating)
                args
            end
            obj.log("DEBUG", msg, args{:}); 
        end
        
        function info(obj, msg, args)
            arguments
                obj, msg (1,1) string
            end
            arguments (Repeating)
                args
            end
            obj.log("INFO", msg, args{:}); 
        end

        function success(obj, msg, args)
            arguments
                obj, msg (1,1) string
            end
            arguments (Repeating)
                args
            end
            obj.log("SUCCESS", msg, args{:}); 
        end

        function warning(obj, msg, args)
            arguments
                obj, msg (1,1) string
            end
            arguments (Repeating)
                args
            end
            obj.log("WARNING", msg, args{:}); 
        end

        function error(obj, msg, args)
            arguments
                obj, msg (1,1) string
            end
            arguments (Repeating)
                args
            end
            obj.log("ERROR", msg, args{:}); 
        end

        function critical(obj, msg, args)
            arguments
                obj, msg (1,1) string
            end
            arguments (Repeating)
                args
            end
            obj.log("CRITICAL", msg, args{:}); 
        end

        function addContext(obj, key, value)
            % PURPOSE: Manages the context
            arguments
                obj (1,1) Logger
                key (1,1) string
                value
            end
            % Adds or updates a key-value pair in the context map
            obj.extraContext(key) = value;
        end

        function clearContext(obj)
            % PURPOSE: Resets the entire context map
            obj.extraContext = containers.Map();
        end

        % --- Class destructor ---
        function delete(obj)
            arguments
                obj (1,1) Logger
            end
            % PURPOSE: Iterates through all sinks to ensure 
            %   file handles are closed properly
            for k = 1:numel(obj.sinks)
                h = obj.sinks{k};
                % We only close actual file identifiers created by fopen
                if isfield(h, 'fh') && h.fh >=3
                    fclose(h.fh);
                    
                    % Garbage Collection: Check for 0-byte abandoned files
                    if h.type == "file"
                        file_info = dir(h.filename);
                        
                        if ~isempty(file_info) && file_info.bytes == 0
                            try
                                % Explicitly use the builtin delete
                                builtin('delete', h.filename);
                            catch ME
                                % We intentionally catch the exception without throwing it
                                warning('Logger:GarbageCollectionFailed', ...
                                    'Failed to delete empty log file: %s. OS Reason: %s', ...
                                    h.filename, ME.message);
                            end
                        end
                    end
                end
            end
        end
    end
    
    methods (Access = private)
        
        function callerStr = getCallerInfo(~)
            % PURPOSE: Gets the caller's file, function, and line 
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
            % PURPOSE: Formats the extra_context map into a single string
            keys = obj.extraContext.keys;
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
                v = obj.extraContext(k);
                parts{i} = sprintf("%s = %s", k, string(v));
            end
            % Join all parts with a separator
            ctx = strjoin(string(parts), " | ");
        end
        
        function handleRotation(obj, sinkIndex)
            % PURPOSE: Manages file rotation based on file size
            sink = obj.sinks{sinkIndex};
            if isnumeric(sink.rotation) && ~isempty(sink.rotation)
                file_info = dir(sink.filename);
                % Checks if the file exists and if its size exceeds the rotation limit
                if ~isempty(file_info) && file_info.bytes > sink.rotation
                    % Closes the current file handle
                    fclose(sink.fh);
                    % Renames the old file adding a timestamp
                    timestamp = string(datetime('now'),'yyyyMMdd_HHmmss_SSS');
                    [path, name_only, ext] = fileparts(sink.filename);
                    new_name = fullfile(path, sprintf('%s_%s%s', name_only, timestamp, ext));
                    movefile(sink.filename, new_name);
                    
                    % Re-opens a new, empty file with the original name
                    [fh, ~] = fopen(sink.filename, 'a');
                    if fh == -1
                        warning('Failed to re-open log file after rotation.');
                    end
                    % Update the sink with the new file handle
                    obj.sinks{sinkIndex}.fh = fh;
                end
            end
        end
    end
end

