function RunVBMPipeline(enableFileLogging, outputDir, inputDir, csvName)
%% RUNVBMPIPELINE
%   An object-oriented orchestrator script that executes a complete Voxel-Based 
%   Morphometry (VBM) pipeline, from raw data ingestion to final statistical rendering.
%
%  PURPOSE:
%   To automate the extraction of clinical covariates and NIfTI volumes, generate 
%   an explicit TPM Mask, estimate a General Linear Model (GLM) via SPM25 
%   (Two-Sample T-Test: CTRL > AD), apply Family-Wise Error (FWE) 
%   correction, and project the surviving significant clusters onto an 
%   anatomical background.
%
%  ARGUMENTS:
%   - enableFileLogging (logical): If true, write logs to disk. Default: false (console-only)
%   - outputDir (char): Directory for all ouputs. If empty (''), defaults to the script's directory
%   - inputDir (char): Directory containing NIfTI and CSV files. If empty (''), defaults to 
%       'AD_CTRL' in the project root
%   - csvName (char): Name of the CSV file in the inputDir. Defaults to 'covariateADCTRLsexAgeTIV.csv' 
%
%  DESCRIPTION:
%   This script acts as the VBM analysis orchestrator. It begins by establishing
%   a dual-destination (Command Window and file) logging environment to ensure reproducibility 
%   and tracking. It utilizes the CohortData class to recursively scan for clinical
%   CSVs and structural 3D NIfTI images, loading the entire dataset into memory.
%   A BrainMask object is then instantiated to derive and export an explicit binary
%   mask based on a standard Tissue Probability Map (TPM). The statistical engine,
%   VBMAnalysis, configures and runs the SPM25 GLM batch without manual intervention,
%   ultimately extracting the FWE-corrected continuous T-map (alpha = 0.05). Finally, 
%   BrainRenderer handles the graphical projection of the significant atrophy clusters
%   overlaid on a specific subject background volume (CTRL-117).

    arguments
        enableFileLogging (1,1) logical = false
        outputDir (1,:) char = ''
        inputDir (1,:) char = ''
        csvName (1,:) char = 'covariateADCTRLsexAgeTIV.csv'
    end

    %% Environment Initialization and Logging

    % Define the base paths and file names
    MATLABPath = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(fileparts(MATLABPath));
    utilsPath = fullfile(MATLABPath, 'utils');

    % Determine cohort dir path
    if isempty(inputDir)
        cohortPath = fullfile(projectRoot, 'AD_CTRL');
    else
        cohortPath = inputDir;
    end

    % Determine output root
    if isempty(outputDir)
        outputRoot = MATLABPath;
    else
        outputRoot = outputDir;
    end
    vbmBase = fullfile(outputRoot, 'VBM_Pipeline_Results');

    plotsDir = fullfile(vbmBase, 'Plots');
    resultsDir = fullfile(vbmBase, 'Results');
    logDir = fullfile(vbmBase, 'Log_Files');
    logPath = fullfile(logDir, 'VBMPipeline.log');

    % Add utils path for utility functions
    if ~isfolder(utilsPath)
        error('Directory not found: %s', utilsPath);
    end
    addpath(utilsPath);

    % Purge existing output directories
    resetDirectory(plotsDir);
    resetDirectory(resultsDir);

    % Initialize the logger to track
    logger = Logger('VBMPipeline');
    logger.addConsoleHandler('level', 'DEBUG', 'useColors', true);
    logger.success('Console logging successfully initialized.');

    % Verify toolboxes
    try
        validateMatlabEnv();
    catch ME
        logger.error('Environment validation failed: %s', ME.message);
        rethrow(ME);
    end

    if enableFileLogging
        % Make the file logging directory
        resetDirectory(logDir);

        % Initialize file logger
        try
            logger.addFileHandler(char(logPath), 'level', 'DEBUG', 'rotation', 10000);
            logger.success('File logging successfully initialized at: %s', logPath);
        catch ME
            % Since later modules require write access, we must abort
            % to prevent delayed crashes
            logger.critical('I/O ERROR: Cannot write to %s', logPath);
            error('RunVBMPipeline:PermissionDenied', ...
                'Write permission denied for directory: %s. \nError: %s', ...
                logDir, ME.message);
        end
    else
        % Dummy write test for console-only mode
        if ~exist(vbmBase, 'dir')
            mkdir(vbmBase);
        end
        dummyFile = fullfile(vbmBase, '.dummy_write_test');
        fid = fopen(dummyFile, 'w');
        if fid == -1
            logger.critical('I/O ERROR: Cannot write to the defined output space %s.', vbmBase);
            logger.critical('Pipeline aborted. Ensure you have write permissions on this filesystem.');
            error('RunVBMPipeline:PermissionDenied', 'Write permission denied for output directory.');
        end
        fclose(fid);
        delete(dummyFile);
        logger.info('Dummy write test passed. Filesystem allows writing. Operating in console-only mode.');
    end
    
    % Kill the logger when the function exits
    cleaner = onCleanup(@() delete(logger));

    try
        spmDir = loadSpmEnvironment();
        logger.success('SPM environment loaded successfully mapped at: %s', spmDir);
    catch ME
        handleError(logger, 'FATAL: Could not resolve SPM dependency.', ME);
    end
    
    tpmPath = fullfile(spmDir, 'tpm', 'TPM.nii');

    if ~isfile(tpmPath)
        logger.critical('The TPM.nii file is missing from the SPM installation: %s', tpmPath);
        error('RunVBMPipeline:TpmMissing', 'TPM file not found: %s', tpmPath);
    end


    %% Data loading and grouping (CohortData)
    logger.info('--- Phase 1: data loading and grouping ---');

    try
        % Initialize CohortData passing the root and the exact CSV name
        myCohort = CohortData(cohortPath, csvName, logger);

        % Use recursive search (**) to find that CSV and the NIfTI files
        % in any subfolder
        myCohort.scanDirectory(); 

        % Load just scanned data into RAM
        myCohort.loadData();

        % Extract spatial information (Niftiinfo, affine matrix and dimensions)
        refInfo = myCohort.getReferenceInfo();
    catch ME
        handleError(logger, 'FATAL: Data loading and grouping failed (CohortData).', ME)
    end

    %% TPM mask generation and export (BrainMask)
    logger.info('--- Phase 2: TPM mask generation ---');

    try
        % Initialize the class and compute TPM Mask
        tpmMask = BrainMask(refInfo, logger);
        absThreshold = 0.01;
        tpmMask.computeTpmMask(tpmPath, absThreshold);

        % Show TPM Mask Stats
        tpmMask.showMaskStats();

        % Export TPM Mask as NIfTI file
        tpmMaskPath = fullfile(resultsDir, 'explicit_tpm_mask.nii');
        tpmMask.exportToNifti(tpmMaskPath);
    catch ME
        handleError(logger, 'FATAL: TPM mask generation or export failed (BrainMask)', ME);
    end    

    %% VBM analysis on TPM mask (VBMAnalysis)

    logger.info('--- Phase 3: VBM Analysis on TPM Mask ---');

    try
        % Initialize VBM analysis class
        vbmDir = fullfile(resultsDir, "VBM_Results");
        vbmModel = VBMAnalysis(logger);
        contrastName = 'Atrophy: CTRL > AD';
        correctionMode = 'FWE';
        alpha = 0.05;

        % Start two sample t-test on TPM Mask
        vbmModel.twoSampleTTest(vbmDir, myCohort, tpmMaskPath, 'AD', 'CTRL');

        % Extract and export the corrected map based on TPM mask (Family-Wise Error at alpha = 0.05)
        tpmFweMapPath = fullfile(resultsDir, 'TPM_Mask_FWE_corrected_map.nii');
        [tpmFweMap, tpmThresh] = vbmModel.getCorrectedMap(vbmDir, contrastName, alpha, correctionMode, tpmFweMapPath);
    catch ME
        handleError(logger, 'VBM Analysis on TPM Mask failed (VBMAnalysis)', ME);
    end

    %% Plot statistical results on a background volume (BrainRenderer)

    logger.info('--- Phase 4: Plot statistical results on a background volume ---');

    try
        % Use the subject CTRL-117 as background volume
        CTRL117Volume = myCohort.getSubjVolume('CTRL-117');

        % Set the variables for the plot
        affineMat = refInfo.NumericMatrix;
        sliceConfig = 3.0; % Auto-mode: 3mm step in MNI space

        % Plot the corrected map on CTRL-117
        figTpmFwe = fullfile(plotsDir, 'VBM_TPM_Mask_FWE_005.png');
        renderer = BrainRenderer(logger);
        renderer.plotStatisticalOverlay(tpmFweMap, tpmThresh, CTRL117Volume, affineMat,...
                sliceConfig, contrastName, correctionMode, alpha,'TPM FWE Map', figTpmFwe);
    catch ME
        handleError(logger, 'Plot VBM analysis on a background volume failed (BrainRenderer)', ME);
    end

    logger.success('VBM Pipeline successfully completed!');
end