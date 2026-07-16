function RunMaskComparison(enableFileLogging, outputDir, inputDir, csvName)
%% RUNMASKCOMPARISON
%   A benchmarking orchestrator that evaluates and compares the effects
%   of different brain masking strategies.
% 
%  PURPOSE:
%   Perform voxel leakage, distributional and topological comparisons among
%   different mask strategies. To systematically quantify and visualize
%   effects between a data-driven Mean Mask and a template-driven TPM Mask,
%   estimate indipendent GLM desings using SPM25, apply topological correction
%   (FWE) and evaluate the structural similarity of the resulting atrophy maps.
%
%  ARGUMENTS:
%   - enableFileLogging (logical): If true, write logs to disk. Default: false (console-only)
%   - outputDir (char): Directory for all ouputs. If empty (''), defaults to the script's directory
%   - inputDir (char): Directory containing NIfTI and CSV files. If empty (''), defaults to 
%       'AD_CTRL' in the project root
%   - csvName (char): Name of the CSV file in the inputDir. Defaults to 'covariateADCTRLsexAgeTIV.csv' 
%
%  DESCRIPTION:
%   This script initializes a dual-destination (Command Window and file) logging
%   state and leverages CohortData to scan NIfTI and clinical covariates files and cache 4D single
%   tensors and CSV file in memory. It instantiates different BrainMask objects to compute 
%   data-driven Mean and Consenus Masks and a template-driven TPM Mask. Ridgway
%   optimized threshold for Mean and TPM Masks is computed.
%   These five Mask strategies are compared with different metrics:
%       - Number of active voxels (benchmark: 6.5x10^5)
%       - Voxel leakage across clinical groups (AD vs. CTRL) against an
%       absolute threshold of 0.01
%       - Mask Overlay on the middle slice with 2 (AD-4 and CTRL-117) similar subjects and
%       the Mean Volume
%       - Voxel distribution histograms with 2 (AD-4 and CTRL-117) similar subjects and
%       the Mean Volume
%   Since the Mean and TPM masks exhibit similar spatial statistics, a visual overlay
%   projection of the masks is performed across multiple slices of a specific subject background 
%   volume (CTRL-117). The plots are generated via BrainRenderer class.
%   To also assess the effects of Mean and TPM Masks on the atrophy, a VBM Analysis 
%   is performed. In the statistical execution phase, the script passes the cohort 
%   state to VBMAnalysis, running two independent General Linear Models from scratch. 
%   Family-Wise Error topology corrections (alpha = 0.05) are applied on the estimated 
%   atrophy contrast (CTRL > AD) to generate filtered continuous maps. The script 
%   then mathematically evaluates spatial similarity (Dice, Jaccard, and Inclusion 
%   ratios) alongside Pearson intensity correlations within the intersecting voxels. 
%   Slice projections of the corrected maps are plotted over the CTRL-117 volume
%   to evaluate cluster localization.
%   Finally, a visual overlay projection of the corrected maps is performed across
%   multiple slices of the CTRL-117 volume.

        arguments
                enableFileLogging (1,1) logical = false
                outputDir (1,:) char = ''
                inputDir (1,:) char = ''
                csvName (1,:) char = 'covariateADCTRLsexAgeTIV.csv'
        end

        %% Environment initialization and logging

        % Define the base paths and file names
        scriptPath = fileparts(mfilename('fullpath'));
        MATLABPath = fileparts(scriptPath);
        projectRoot = fileparts(MATLABPath);
        utilsPath = fullfile(MATLABPath, 'utils');

        % Determine cohort dir path
        if isempty(inputDir)
                cohortPath = fullfile(projectRoot, 'AD_CTRL');
        else
                cohortPath = inputDir;
        end

        % Determine output root
        if isempty(outputDir)
                outputRoot = scriptPath;
        else
                outputRoot = outputDir;
        end

        maskBase = fullfile(outputRoot, 'Mask_Comparison_Results');

        plotsDir = fullfile(maskBase, 'Plots');
        resultsDir = fullfile(maskBase, 'Results');
        logDir = fullfile(maskBase, 'Log_Files');
        logPath = fullfile(logDir, 'maskComparison.log');

        if ~isfolder(utilsPath)
                error('Directory not found: %s', utilsPath);
        end
        addpath(utilsPath);

        % Verify toolboxes
        validateMatlabEnv();

        % Wipe existing output directories
        resetDirectory(plotsDir);
        resetDirectory(resultsDir);

        logger = Logger('MaskComparison');
        logger.addConsoleHandler('level', 'DEBUG', 'useColors', true);

        if enableFileLogging
                resetDirectory(logDir);

                % Initialize file logger
                try
                        logger.addFileHandler(char(logPath), 'level', 'DEBUG', 'rotation', 10000);
                        logger.success('File logging successfully initialized at: %s', logPath);
                catch ME
                        % Since later modules require write access, we must abort
                        % to prevent delayed crashes
                        logger.critical('I/O ERROR: Cannot write to %s', logPath);
                        error('RunMaskComparison:PermissionDenied', ...
                                'Write permission denied for directory: %s. \nError: %s', ...
                                logDir, ME.message);
                end
        else
                % Dummy write test for console-only mode
                if ~exist(maskBase, 'dir')
                        mkdir(maskBase);
                end
                        dummyFile = fullfile(maskBase, '.dummy_write_test');
                        fid = fopen(dummyFile, 'w');
                if fid == -1
                        logger.critical('I/O ERROR: Cannot write to the defined output space %s.', vbmBase);
                        logger.critical('Pipeline aborted. Ensure you have write permissions on this filesystem.');
                        error('RunMaskComparison:PermissionDenied', 'Write permission denied for output directory.');
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
                error('RunMaskComparison:TpmMissing', 'TPM file not found: %s', tpmPath);
        end

        %% Data loading and grouping (CohortData)
        logger.info('--- Phase 1: Data loading and grouping ---');

        try
                % Initialize CohortData passing the root and the exact CSV name
                myCohort = CohortData(cohortPath, csvName, logger);

                % Use recursive search (**) to find that CSV and the NIfTI files
                % in any subfolder
                myCohort.scanDirectory(); 

                % Load just scanned data into RAM
                myCohort.loadData();

                % Extract spatial information (niftiinfo, affine matrix and dimensions)
                refInfo = myCohort.getReferenceInfo();
        catch ME
                handleError(logger, 'FATAL: Data loading and grouping failed (CohortData).', ME)
        end

        %% Mask generation and first evaluation (BrainMask)
        logger.info('--- Phase 2: Mask generation and first evaluation ---');

        try
                absThreshold = 0.01;
                consensusRatio = 0.70;
                stepSize = 0.005;

                % --- Mean Mask ---
                meanMask = BrainMask(refInfo, logger);
                meanMask.computeMeanMask(myCohort, absThreshold);
                meanMask.showMaskStats;
                meanLeakageAD = meanMask.evaluateLeakage(myCohort, 'AD');
                meanLeakageCTRL = meanMask.evaluateLeakage(myCohort, 'CTRL');

                % --- Consensus Mask ---
                consensusMask = BrainMask(refInfo, logger);
                consensusMask.computeConsensusMask(myCohort, absThreshold, consensusRatio);
                consensusMask.showMaskStats;
                consensusLeakageAD = consensusMask.evaluateLeakage(myCohort, 'AD');
                consensusLeakageCTRL = consensusMask.evaluateLeakage(myCohort, 'CTRL');

                % --- TPM Mask ---
                tpmMask = BrainMask(refInfo, logger);
                tpmMask.computeTpmMask(tpmPath, absThreshold);
                tpmMask.showMaskStats;
                tpmLeakageAD = tpmMask.evaluateLeakage(myCohort, 'AD');
                tpmLeakageCTRL = tpmMask.evaluateLeakage(myCohort, 'CTRL');

                % --- Ridgway Optimized Mean Mask ---
                ridgMeanMask = BrainMask(refInfo, logger);
                ridgMeanMask.computeRidgwayMeanMask(myCohort, stepSize);
                ridgMeanMask.showMaskStats;
                ridgMeanLeakageAD = ridgMeanMask.evaluateLeakage(myCohort, 'AD');
                ridgMeanLeakageCTRL = ridgMeanMask.evaluateLeakage(myCohort, 'CTRL');

                % --- Ridgway Optimized TPM Mask ---
                ridgTpmMask = BrainMask(refInfo, logger);
                ridgTpmMask.computeRidgwayTpmMask(tpmPath, stepSize);
                ridgTpmMask.showMaskStats;
                ridgTpmLeakageAD = ridgTpmMask.evaluateLeakage(myCohort, 'AD');
                ridgTpmLeakageCTRL = ridgTpmMask.evaluateLeakage(myCohort, 'CTRL');
        catch ME
                handleError(logger, 'FATAL: Mask generation or export failed (BrainMask)', ME);
        end

        %% Plotting and visual comparison (BrainRenderer)
        logger.info('--- Phase 3: Plotting and visual comparison ---');

        try
                renderer = BrainRenderer(logger);

                % --- Leakage Histograms ---
                leakMatAD = [meanLeakageAD, consensusLeakageAD, tpmLeakageAD,...
                        ridgMeanLeakageAD, ridgTpmLeakageAD];
                leakMatCTRL = [meanLeakageCTRL, consensusLeakageCTRL, tpmLeakageCTRL,...
                        ridgMeanLeakageCTRL, ridgTpmLeakageCTRL];
                methodLabels = {meanMask.MaskType, consensusMask.MaskType, tpmMask.MaskType,...
                        ridgMeanMask.MaskType, ridgTpmMask.MaskType};
                groupColors  = {'r', 'b'}; % Red for AD, Blue for CTRL

                figLeakHist = fullfile(plotsDir, 'Leakage_Comparison.png');
                renderer.plotLeakageHistograms(leakMatAD, leakMatCTRL, methodLabels,...
                        groupColors, figLeakHist);

                % Extract mean Volume
                meanVolume = myCohort.getMeanVolume('ALL'); 

                % Extract AD-4 and CTRL-117 because they have similar TIV and age and same
                % sex
                subjectLabelAD = 'AD-4.';
                AD4Volume = myCohort.getSubjVolume(subjectLabelAD);
                subjectLabelCTRL = 'CTRL-117.';
                CTRL117Volume = myCohort.getSubjVolume(subjectLabelCTRL);

                % Extract middle slice index
                idxMidSlice = round(size(meanVolume, 3) * 0.5);

                % Take necessaries for the mask overlay and mask histograms plots
                baseVolumes = cat(4, meanVolume, AD4Volume, CTRL117Volume);
                baseSlices = squeeze(baseVolumes(:,:,idxMidSlice,:));
                baseTitles = {'Mean Vol', [subjectLabelAD, ' Vol'],...
                        [subjectLabelCTRL, ' Vol']};
                maskVolumes = cat(4, meanMask.Matrix, consensusMask.Matrix, tpmMask.Matrix, ...
                                ridgMeanMask.Matrix, ridgTpmMask.Matrix);
                maskSlices = squeeze(maskVolumes(:,:,idxMidSlice,:));
                maskColors  = {'r', 'g', 'b', 'y', 'm'};
                meanLabel = sprintf('%s Thr = %.3f', meanMask.MaskType, meanMask.IntensityThreshold);
                consensusLabel = sprintf('Cons Mask Thr = %.3f, Ratio = %.2f', ...
                        consensusMask.IntensityThreshold, consensusMask.ConsensusRatio);
                tpmLabel = sprintf('%s Thr = %.3f', tpmMask.MaskType, tpmMask.IntensityThreshold);
                ridgMeanLabel = sprintf('Ridg Mean Mask Thr = %.3f', ridgMeanMask.IntensityThreshold);
                ridgTpmLabel = sprintf('Ridg TPM Mask Thr = %.3f', ridgTpmMask.IntensityThreshold);
                maskLabels = {meanLabel, consensusLabel, tpmLabel, ridgMeanLabel, ridgTpmLabel};
                figOverlay = fullfile(plotsDir, 'Overlay_Comparison.png');
                renderer.plotMaskOverlays(baseSlices, baseTitles, maskSlices, maskColors,...
                        maskLabels, figOverlay);

                % --- Mask Histograms ---
                figMaskHist = fullfile(plotsDir, 'Mask_Hist_Comparison.png');
                renderer.plotMaskHistograms(baseVolumes, baseTitles, maskVolumes, maskColors,...
                        maskLabels, stepSize, absThreshold, figMaskHist);

                % --- Composite Overlap ---
                % Use CTRL-117 as background volume
                affineMat = refInfo.NumericMatrix;
                figOverlap = fullfile(plotsDir, 'Mask_Overlap_Comparison.png');

                % Compare only meanMask and tpmMask since they have similar spatial stats
                % (active voxels, leakage, voxel distributions, overlaid regions) 
                sliceConfig = 3; % Auto-mode: 3mm step in MNI space
                renderer.plotCompositeOverlap(meanMask.Matrix, tpmMask.Matrix, CTRL117Volume, affineMat, 3,...
                        'Mean Mask', 'TPM Mask', figOverlap);
        catch ME
                handleError(logger, 'FATAL: Plotting and visual comparison failed (BrainRenderer)', ME);
        end

        %% VBM Analysis on Mean Mask and TPM Mask (VBMAnalysis)

        % Since Mean Mask and TPM Mask have similar spatial stats, we will now
        % export them as NIfTI files and use them as explicit masks in the VBM
        % Analysis to verify whether they also lead to the same statistical results

        logger.info('--- Phase 4: VBM Analysis on Mean Mask and TPM Mask');

        try
                % Export Mean Mask and TPM Mask as NIfTI files
                maskDir = fullfile(resultsDir, "Explicit_Masks");
                if ~exist(maskDir, 'dir')
                        mkdir(maskDir);
                end
                meanMaskPath = fullfile(maskDir, 'explicit_mean_mask.nii');
                meanMask.exportToNifti(meanMaskPath);
                tpmMaskPath = fullfile(maskDir, 'explicit_tpm_mask.nii');
                tpmMask.exportToNifti(tpmMaskPath);

                % Initialize VBM Analysis class
                vbmModel = VBMAnalysis(logger);
                contrastName = 'Atrophy: CTRL > AD';
                correctionMode = 'FWE';
                alpha = 0.05;

                % Start two sample t-test on Mean Mask
                meanResultsDir = fullfile(resultsDir, 'Mean_Mask_VBM_Results');
                if ~exist(meanResultsDir, 'dir')
                        mkdir(meanResultsDir);
                end
                vbmModel.twoSampleTTest(meanResultsDir, myCohort, meanMaskPath, 'AD', 'CTRL');

                % Extract the corrected map based on Mean Mask (Family-Wise Error at alpha = 0.05)
                [meanFweMap, meanThresh] = vbmModel.getCorrectedMap(meanResultsDir, contrastName, alpha, correctionMode);

                % Start two sample t-test on TPM Mask
                tpmResultsDir = fullfile(resultsDir, 'TPM_Mask_VBM_Results');
                if ~exist(tpmResultsDir, 'dir')
                        mkdir(tpmResultsDir);
                end
                vbmModel.twoSampleTTest(tpmResultsDir, myCohort, tpmMaskPath, 'AD', 'CTRL');

                % Extract the corrected map based on TPM Mask (Family-Wise Error at alpha = 0.05)
                [tpmFweMap, tpmThresh] = vbmModel.getCorrectedMap(tpmResultsDir, contrastName, alpha, correctionMode);

                % Calculate similarity metrics on the corrected maps
                meanFweMapName = 'Mean FWE Map';
                tpmFweMapName = 'TPM FWE Map';
                vbmModel.evaluateMapSimilarity(meanFweMap, tpmFweMap, meanFweMapName, tpmFweMapName);

                % Plot the corrected maps on CTRL-117 volume as background
                figMeanFwe = fullfile(plotsDir, 'VBM_Mean_Mask_FWE_005.png');
                renderer.plotStatisticalOverlay(meanFweMap, meanThresh, CTRL117Volume, affineMat,...
                        sliceConfig, contrastName, correctionMode, alpha, meanFweMapName, figMeanFwe);
                figTpmFwe = fullfile(plotsDir, 'VBM_TPM_Mask_FWE_005_0.png');
                renderer.plotStatisticalOverlay(tpmFweMap, tpmThresh, CTRL117Volume, affineMat,...
                        sliceConfig, contrastName, correctionMode, alpha, tpmFweMapName, figTpmFwe);

                % Compare the corrected maps
                figOverlapFwe = fullfile(plotsDir, 'Overlap_Mean_TPM_FWE_005.png');
                renderer.plotCompositeOverlap(meanFweMap, tpmFweMap, CTRL117Volume, affineMat,...
                        sliceConfig, meanFweMapName, tpmFweMapName, figOverlapFwe);
        catch ME
                handleError(logger, 'FATAL: VBM Analysis on Mean Mask and TPM Mask failed (VBMAnalysis)', ME);
        end

        logger.success('Mask Comparison successfully completed!');
end