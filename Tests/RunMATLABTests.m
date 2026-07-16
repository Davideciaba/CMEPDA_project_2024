function RunMATLABTests()
%% RUNMATLABTESTS
%   Automated Test Suite Runner for the MATLAB Framework
%
% PURPOSE:
%   Configures the MATLAB path environment dependencies 
%   (e.g., adding the Logger utility) and executes all 
%   Unit Tests in the \Tests directory.

    % Define script and project paths
    scriptPath = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(scriptPath);

    % Path to the Logger class
    utilsPath = fullfile(projectRoot, 'MATLAB', 'utils');

    fprintf('Configuring Test Environment...\n');
    % Add utils path for utility functions
    if ~isfolder(utilsPath)
        error('Directory not found: %s', utilsPath);
    end
    addpath(utilsPath);

    % Run the Unit Tests
    fprintf('Starting Unit Test Suite...\n');

    % This command automatically finds and runs any test file in the \Tests
    % folder
    results = runtests(fullfile(projectRoot, 'Tests'));
    disp(results);

    % Remove path after tests
    rmpath(utilsPath);
    fprintf('Environment Restored\n');
end