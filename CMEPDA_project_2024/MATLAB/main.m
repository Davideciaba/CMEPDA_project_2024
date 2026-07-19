function main(options)
% MAIN Entry point for the MATLAB pipeline.
% Orchestrates the RunVBMPipeline and RunMaskComparison scripts.
%
% EXAMPLES OF USAGE (from MATLAB Command Window):
%   main('runVBM', true)
%   main('runMaskComp', true, 'enableFileLogging', true)
%   main('runVBM', true, 'outputDir', 'C:\Custom\Path\Results')

    arguments
        options.runVBM (1,1) logical = false
        options.runMaskComp (1,1) logical = false
        options.enableFileLogging (1,1) logical = false
        options.outputDir (1,:) char = ''
        options.inputDir (1,:) char = ''
        options.csvName (1,:) char = 'covariateADCTRLsexAgeTIV.csv'
    end

    % Resolve the project root
    MATLABPath = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(fileparts(MATLABPath));

    % Resolve default directories
    if isempty(options.outputDir)
        options.outputDir = fullfile(projectRoot, 'MATLAB_Results');
    end
    if isempty(options.inputDir)
        options.inputDir = fullfile(projectRoot, 'AD_CTRL');
    end

    % Add internal modules to MATLAB's search path
    utilsPath = fullfile(MATLABPath, "utils");

    if ~isfolder(utilsPath) || ~isfolder(MATLABPath)
        error('main:MissingDirectories', ...
            'One or more required directories (MATLAB/utils, MATLAB) were not found.');
    end

    addpath(utilsPath, MATLABPath);

    % Clean up utility paths on exit
    cleanerPath = onCleanup(@() rmpath(utilsPath, MATLABPath));

    % Initialize the logger to track the orchestrator's state
    logger = Logger('Orchestrator');
    logger.addConsoleHandler('level', 'DEBUG', 'useColors', true);
    
    % Environment validation
    try
        validateMatlabEnv();
    catch ME
        logger.error('Environment validation failed: %s', ME.message);
        rethrow(ME);
    end

    if ~options.runVBM && ~options.runMaskComp
        logger.warning('No execution flags provided. Pass ''runVBM'', true or ''runMaskComp'', true to begin.');
        return;
    end

    try
        logger.info('Global File Logging Enabled: %s', mat2str(options.enableFileLogging));
        logger.info('Global Output Directory mapped to: %s', options.outputDir);
        logger.info('Global Input Directory mapped to: %s', options.inputDir);
        logger.info('Target Clinical CSV file: %s', options.csvName);

        % --- Execute VBM Pipeline ---
        if options.runVBM
            logger.info('--- Handing execution over to VBM Orchestrator ---');
            RunVBMPipeline(options.enableFileLogging, options.outputDir, options.inputDir, options.csvName);
            logger.success('VBM Execution returned successfully.');
        end

        % --- Execute Mask Comparison ---
        if options.runMaskComp
            logger.info('--- Handing execution over to Mask Comparison Orchestrator ---');
            RunMaskComparison(options.enableFileLogging, options.outputDir, options.inputDir, options.csvName);
            logger.success('Mask Comparison Execution returned successfully.');
        end

        logger.success('All requested MATLAB pipelines completed.');

    catch ME
        % Error handling
        handleError(logger, 'Unhandled orchestrator error.', ME);
    end
end