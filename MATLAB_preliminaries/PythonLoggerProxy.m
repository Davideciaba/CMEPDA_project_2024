classdef PythonLoggerProxy < handle
% PythonLoggerProxy forwards log messages to a Python logger.
% This version is optimized to ONLY forward messages to Python,
% without printing a copy to the MATLAB Command Window.

    properties (Access = private)
        pythonLogHandle % Handle to the Python 'py_logger.log' function
    end

    methods
        function obj = PythonLoggerProxy(pythonHandle)
            % The constructor receives the handle to the Python logging function.
            if nargin < 1 || ~isa(pythonHandle, 'function_handle')
                error('PythonLoggerProxy requires a valid function handle as an argument.');
            end
            obj.pythonLogHandle = pythonHandle;
        end

        % Public wrapper methods for each log level
        function trace(obj, msg, varargin), obj.log('TRACE', msg, varargin{:}); end
        function debug(obj, msg, varargin), obj.log('DEBUG', msg, varargin{:}); end
        function info(obj, msg, varargin), obj.log('INFO', msg, varargin{:}); end
        function success(obj, msg, varargin), obj.log('SUCCESS', msg, varargin{:}); end
        function warn(obj, msg, varargin), obj.log('WARNING', msg, varargin{:}); end
        function error(obj, msg, varargin), obj.log('ERROR', msg, varargin{:}); end
        function critical(obj, msg, varargin), obj.log('CRITICAL', msg, varargin{:}); end
    end

    methods (Access = private)
        function log(obj, levelName, msg, varargin)
            % The central method that handles logging.
            
            % 1. Format the message using sprintf if additional arguments are provided
            formattedMsg = sprintf(msg, varargin{:});
            
            % 2. Forward the log to the Python logger
            try
                obj.pythonLogHandle(levelName, formattedMsg);
            catch ME
                % Print a warning to the Command Window only if forwarding fails
                warning('Failed to forward log to Python. Error [%s]: %s', ME.identifier, ME.message);
            end
        end
    end
end