function RunStandalone(varargin)
% RUNSTANDALONE Launches Preliminaries from MATLAB with validated inputs.
%    RUNSTANDALONE() auto-detects the AD/CTRL folders and TIV CSV based on the
%    project layout (two levels above this file) and forwards them to
%    Preliminaries.m, sharing a Logger instance.
%
%    RUNSTANDALONE('dirAD', value, 'dirCTRL', value, 'tivPath', value)
%    allows overriding any of the detected paths. Each argument must be a
%    valid folder/file on disk. The default TIV file is:
%       AD_CTRL/covariateADCTRLsexAgeTIV.csv
%
%    Examples:
%        RunStandalone;
%        RunStandalone('dirAD','D:\data\AD','tivPath','D:\data\tiv.csv');

    %% --- Refresh MATLAB path cache ---
    rehash;

    %% --- Logger initialization ---
    logger = Logger('RunStandalone');
    logger.addConsoleHandler('use_colors', true);

    %% --- Resolve project folders and parse inputs ---
    thisFile    = mfilename('fullpath');
    thisDir     = fileparts(thisFile);
    projectRoot = fileparts(thisDir);

    defaults = detectDefaultPaths(projectRoot);
    args = parseInputs(defaults, varargin{:});

    logger.info('projectRoot: %s', projectRoot);
    logger.info('dirAD: %s', args.dirAD);
    logger.info('dirCTRL: %s', args.dirCTRL);
    logger.info('tivPath: %s', args.tivPath);

    %% --- Launch Preliminaries (share the same logger) ---
    logger.info('Launching Preliminaries...');
    Preliminaries(args.dirAD, args.dirCTRL, args.tivPath, 'loggerHandle', logger);
end

function defaults = detectDefaultPaths(projectRoot)
    dirAD   = fullfile(projectRoot, 'AD_CTRL', 'AD_s3');
    dirCTRL = fullfile(projectRoot, 'AD_CTRL', 'CTRL_s3');
    tivPath = fullfile(projectRoot, 'AD_CTRL', 'covariateADCTRLsexAgeTIV.csv');

    defaults = struct('dirAD', dirAD, 'dirCTRL', dirCTRL, 'tivPath', tivPath);
end

function args = parseInputs(defaults, varargin)
    p = inputParser;
    addParameter(p, 'dirAD', defaults.dirAD, @(x) validateDir(x, 'dirAD'));
    addParameter(p, 'dirCTRL', defaults.dirCTRL, @(x) validateDir(x, 'dirCTRL'));
    addParameter(p, 'tivPath', defaults.tivPath, @(x) validateFile(x, 'tivPath'));
    parse(p, varargin{:});

    args = struct( ...
        'dirAD', char(p.Results.dirAD), ...
        'dirCTRL', char(p.Results.dirCTRL), ...
        'tivPath', char(p.Results.tivPath));
end

function tf = validateDir(pathValue, label)
    if ~(ischar(pathValue) || isstring(pathValue))
        error('RunStandalone:%sNotChar', label, ...
              '%s must be a character vector or string.', label);
    end
    if ~exist(pathValue, 'dir')
        error('RunStandalone:%sMissing', label, ...
              '%s does not exist or is not a directory: %s', label, pathValue);
    end
    tf = true;
end

function tf = validateFile(pathValue, label)
    if ~(ischar(pathValue) || isstring(pathValue))
        error('RunStandalone:%sNotChar', label, ...
              '%s must be a character vector or string.', label);
    end
    if isempty(pathValue) || ~exist(pathValue, 'file')
        error('RunStandalone:%sMissing', label, ...
              '%s does not exist or is not a file: %s', label, pathValue);
    end
    tf = true;
end
