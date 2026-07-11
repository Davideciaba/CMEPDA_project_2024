function RunVBMPipeline()
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
%  DESCRIPTION:
%   This script acts as the main entry point for the VBM analysis, 
%   integrating the custom OOP framework (Logger, CohortData, BrainMask, 
%   VBMAnalysis, and BrainRenderer). It begins by establishing a dual-destination 
%   (Command Window and file) logging environment to ensure reproducibility 
%   and tracking. It utilizes the CohortData class to recursively scan for clinical
%   CSVs and structural 3D NIfTI images, loading the entire dataset into memory.
%   A BrainMask object is then instantiated to derive and export an explicit binary
%   mask based on a standard Tissue Probability Map (TPM). The statistical engine,
%   VBMAnalysis, configures and runs the SPM25 GLM batch without manual intervention,
%   ultimately extracting the FWE-corrected continuous T-map (alpha = 0.05). Finally, 
%   BrainRenderer handles the graphical projection of the significant atrophy clusters
%   overlaid on a specific subject background volume (CTRL-117).

    %% 1. Environment Initialization and Logging

    % Define the base paths and file names
    scriptPath = fileparts(mfilename('fullpath'));
    MATLABPath = fileparts(scriptPath);
    projectRoot = fileparts(MATLABPath);
    utilsPath = fullfile(MATLABPath, 'utils');
    cohortPath = fullfile(projectRoot, 'AD_CTRL');
    plotsDir = fullfile(scriptPath, 'Plots');
    resultsDir = fullfile(scriptPath, 'Results');
    logDir = fullfile(scriptPath, 'Log Files');
    logPath = fullfile(logDir, 'VBMPipeline.log');
    csvFileName = 'covariateADCTRLsexAgeTIV.csv';
    SPM_DIR = 'C:/Users/utente/Desktop/spm';
    try
        addpath(SPM_DIR); 
    catch ME
        error('%s directory not found. Error: %s', SPM_DIR, ME.message);
    end
    tpmPath = fullfile(fileparts(which('spm')), 'tpm', 'TPM.nii');

    % Add utils path for utility functions
    try
        addpath(utilsPath); 
    catch ME
        error('%s directory not found. Error: %s', utilsPath, ME.message);
    end

    % Initialize the logger to track the comparison
    logger = Logger('VBMPipeline');
    try
        logger.addConsoleHandler('level', 'DEBUG', 'useColors', true);
        logger.success('Console logging successfully initialized.')
    catch ME
        logger.error('Failed to setup console logging.');
        rethrow(ME);
    end

    % Make the file logging directory
    try
        mkdir(logDir);
    catch ME
        % Ignore if it already exists; otherwise, throw an exception
        if ~contains(ME.identifier, 'Exists')
            rethrow(ME);
        end
    end

    % Initialize file logger
    try
        logger.addFileHandler(char(logPath), 'level', 'DEBUG', 'rotation', 10000);
        logger.success('File logging successfully initialized at: %s', logPath);
    catch ME
        logger.error('Failed to setup file logging at: %s.\n Falling back to console only.', logPath);
        rethrow(ME);
    end

    % Kill the logger when the function exits
    cleaner = onCleanup(@() delete(logger));

    %% 2. Data Loading and Grouping (CohortData)
    logger.info('--- Phase 1: Data Loading and Grouping ---');

    % Initialize CohortData passing the root and the exact CSV name
    myCohort = CohortData(cohortPath, csvFileName, logger);

    % Use recursive search (**) to find that CSV and the NIfTI files
    % in any subfolder
    myCohort.scanDirectory(); 

    % Load just scanned data into RAM
    myCohort.loadData();

    % Extract spatial information (Niftiinfo, affine matrix and dimensions)
    refInfo = myCohort.getReferenceInfo();

    %% 3. TPM Mask Generation and Export (BrainMask)
    logger.info('--- Phase 2: TPM Mask Generation ---');

    % Initialize the class and compute TPM Mask
    tpmMask = BrainMask(refInfo, logger);
    absThreshold = 0.01;
    tpmMask.computeTpmMask(tpmPath, absThreshold);

    % Show TPM Mask Stats
    tpmMask.showMaskStats();

    % Export TPM Mask as NIfTI file
    maskDir = fullfile(resultsDir, 'Explicit Mask');
    tpmMaskPath = fullfile(maskDir, 'explicit_tpm_mask.nii');
    tpmMask.exportToNifti(tpmMaskPath);

    %% 4. VBM Analysis on TPM Mask (VBMAnalysis)

    logger.info('--- Phase 3: VBM Analysis on TPM Mask ---');

    % Initialize VBM Analysis class
    vbmDir = fullfile(resultsDir, "VBM Results");
    vbmModel = VBMAnalysis(logger);
    contrastName = 'Atrophy: CTRL > AD';
    correctionMode = 'FWE';
    alpha = 0.05;

    % Start two sample t-test on TPM Mask
    vbmModel.twoSampleTTest(vbmDir, myCohort, tpmMaskPath, 'AD', 'CTRL');

    % Extract and export the corrected map based on TPM Mask (Family-Wise Error at alpha = 0.05)
    tpmFweMapPath = fullfile(resultsDir, 'Thresholded Maps', 'TPM_Mask_FWE_corrected_map.nii');
    [tpmFweMap, tpmThresh] = vbmModel.getCorrectedMap(vbmDir, contrastName, alpha, correctionMode, tpmFweMapPath);

    %% 5. Plot statistical results on a background volume (BrainRenderer)

    logger.info('--- Phase 4: Plot statistical results on a background volume ---');

    % Use the subject CTRL-117 as background volume
    CTRL117Volume = myCohort.getSubjVolume('CTRL-117');

    % Set the variables for the plot
    affineMat = refInfo.NumericMatrix;
    sliceConfig = 3.0; % Auto-mode: 3mm step in MNI space

    % Plot the corrected map on CTRL-117
    figTpmFwe = fullfile(plotsDir, 'VBM_TPM_Mask_FWE_p005.png');
    renderer = BrainRenderer(logger);
    renderer.plotStatisticalOverlay(tpmFweMap, tpmThresh, CTRL117Volume, affineMat,...
            sliceConfig, contrastName, correctionMode, alpha,'TPM FWE Map', figTpmFwe);

    logger.success('VBM Pipeline successfully completed!');
end