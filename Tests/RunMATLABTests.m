%% RUNMATLABTESTS
%   Automated Test Suite Runner for the MATLAB Framework
%
% PURPOSE:
%   Configures the MATLAB path environment dependencies 
%   (e.g., adding the Logger utility) and executes all 
%   Unit Tests in the \Tests directory.

clear; close all;

% Define script and project paths
scriptPath = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptPath);

% Path to the Logger class
utilsPath = fullfile(projectRoot, 'MATLAB', 'utils');

fprintf('Configuring Test Environment...\n');
try
    addpath(utilsPath);
    fprintf('Utility paths injected\n');
catch ME
    error('MATLAB/utils directory not found. Tests will crash. Error: %s', ME.identifier);
end

% Run the Unit Tests
fprintf('Starting Unit Test Suite...\n');

% This command automatically finds and runs any test file in the \Tests
% folder
results = runtests(fullfile(projectRoot, 'Tests'));
disp(results);

% Remove path after tests
rmpath(utilsPath);
fprintf('Environment Restored\n');