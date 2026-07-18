function spmPath = loadSpmEnvironment()
    % LOADSPMENVIRONMENT Resolves SPM directory path and sanitizes MATLAB path.
    %
    % Purpose:
    %   Finds the SPM installation directory using the SPM_DIR environment variable
    %   or an upwards directory search to locate config.json in the CMEPDA_project_2024 root.
    %
    % Returns:
    %   spmPath (char): The absolute validated path to the SPM directory.

    ENV_VAR_NAME = 'SPM_DIR';
    CONFIG_FILENAME = 'config.json';
    PROJECT_BOUNDARY = 'CMEPDA_project_2024';
    SPM_SIGNATURE = 'spm.m';

    % Attempt to read from environment variable
    spmPath = getenv(ENV_VAR_NAME);

    % Fallback to local configuration
    if isempty(spmPath)
        
        % Search upwards from pwd
        configFilePath = findConfigFile(pwd, CONFIG_FILENAME, PROJECT_BOUNDARY);
        
        % Search upwards from script dir
        if isempty(configFilePath)
            scriptDir = fileparts(mfilename('fullpath'));
            configFilePath = findConfigFile(scriptDir, CONFIG_FILENAME, PROJECT_BOUNDARY);
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
            'SPM directory is undefined. Set %s env variable or provide %s in %s.', ...
            ENV_VAR_NAME, CONFIG_FILENAME, PROJECT_BOUNDARY);
    end

    % Validate directory
    if ~isfolder(spmPath)
        error('loadSpmEnvironment:PathNotFound', ...
            'The resolved SPM directory does not exist on disk: %s', spmPath);
    end

    if ~isfile(fullfile(spmPath, SPM_SIGNATURE))
        error('loadSpmEnvironment:InvalidSpmSuite', ...
            'Signature file %s missing in %s. Not a valid SPM directory.', ...
            SPM_SIGNATURE, spmPath);
    end

    % Absolute path for string comparisons
    spmPath = char(fullfile(spmPath));

    % Find all existing registered spm.m files in the current environment
    existingSpmInstances = which(SPM_SIGNATURE, '-all');
    
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

function configPath = findConfigFile(startDir, fileName, stopDirName)
    % FINDCONFIGFILE Local helper to search for a file upwards.
    
    currentScanDir = startDir;
    configPath = '';
    
    while true
        potentialPath = fullfile(currentScanDir, fileName);
        if isfile(potentialPath)
            configPath = potentialPath;
            break;
        end
        
        [parentDir, currentName, ext] = fileparts(currentScanDir);
        fullCurrentName = [currentName, ext];
        
        % Stop conditions: Project root reached OR system root reached
        if strcmp(fullCurrentName, stopDirName) || strcmp(currentScanDir, parentDir) 
            break;
        end
        
        currentScanDir = parentDir;
    end
end