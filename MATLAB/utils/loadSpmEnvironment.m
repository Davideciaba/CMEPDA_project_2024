function spmPath = loadSpmEnvironment()
    % LOADSPMENVIRONMENT Resolves SPM directory path and sanitizes MATLAB path.
    %
    % Purpose:
    %   Finds the SPM installation directory using the SPM_DIR environment variable
    %   or an upwards directory to locate config.json in the project root.
    %   It then removes older SPM paths from the current MATLAB session before 
    %   injecting the targeted SPM path at the top of the search path.
    %
    % Returns:
    %   spmPath (char): The absolute validated path to the SPM directory.

    ENV_VAR_NAME = 'SPM_DIR';
    CONFIG_FILENAME = 'config.json';

    % Attempt to read from environment variable
    spmPath = getenv(ENV_VAR_NAME);

    % Fallback to local configuration
    if isempty(spmPath)
        % Get the absolute path of the directory containing this script
        scriptDir = fileparts(mfilename('fullpath'));
        currentScanDir = scriptDir;
        configFilePath = '';
        
        % Go upwards until config.json is found or root directory is reached
        while true
            potentialPath = fullfile(currentScanDir, CONFIG_FILENAME);
            if isfile(potentialPath)
                configFilePath = potentialPath;
                break;
            end
            
            parentDir = fileparts(currentScanDir);
            if strcmp(currentScanDir, parentDir) 
                % Root directory reached, stop searching
                break;
            end
            currentScanDir = parentDir;
        end

        % If the file was found during traversal, parse its contents
        if ~isempty(configFilePath)
            try
                fileContent = fileread(configFilePath);
                configData = jsondecode(fileContent);
                if isfield(configData, ENV_VAR_NAME)
                    spmPath = configData.(ENV_VAR_NAME);
                end
            catch ME
                error('loadSpmEnvironment:JsonParseError', ...
                    'Failed to parse %s: %s', configFilePath, ME.message);
            end
        end
    end

    % Validate existence
    if isempty(spmPath)
        error('loadSpmEnvironment:ConfigurationMissing', ...
            'SPM directory is undefined. Set %s environment variable or place %s in the project root.', ...
            ENV_VAR_NAME, CONFIG_FILENAME);
    end

    % Validate directory
    if ~isfolder(spmPath)
        error('loadSpmEnvironment:PathNotFound', ...
            'The resolved SPM directory does not exist on disk: %s', spmPath);
    end

    % Absolute path for string comparisons
    spmPath = char(fullfile(spmPath));
    spmPath = char(string(spmPath));

    % Find all existing registered spm.m files in the current environment
    existingSpmInstances = which('spm.m', '-all');
    
    if ~isempty(existingSpmInstances)
        for idx = 1:length(existingSpmInstances)
            oldRoot = fileparts(existingSpmInstances{idx});
            
            % If the existing SPM path is not the one we are trying to load
            if ~strcmpi(oldRoot, spmPath)
                fprintf('WARNING: Conflicting SPM version detected at: %s\n', oldRoot);
                fprintf('Initiating path sanitization...\n');
                
                % Extract the current MATLAB path split into a cell array
                rawPath = path;
                pathCells = strsplit(rawPath, pathsep);
                
                % Identify all paths that start with the old SPM root directory
                len = length(oldRoot);
                isPollutingPath = strncmp(pathCells, oldRoot, len);
                pathsToRemove = pathCells(isPollutingPath);
                
                if ~isempty(pathsToRemove)
                    % Remove all identified paths
                    rmpath(pathsToRemove{:});
                end
            end
        end
    end

    % Inject the SPM target at the top of the search path
    addpath(spmPath, '-begin');
end