classdef BrainMask < handle
    % BRAINMASK Represents a 3D boolean mask for VBM analysis.
    %
    % PURPOSE: Encapsulates the 3D boolean tensor of a brain mask and provides 
    %   mathematical methods to compute Mean, Consensus, TPM, RidgwayMean
    %   and RidgwayTPM masks directly from a CohortData object.
    
    properties (Access = public)
        MaskType char = 'Unallocated'   % Automatically updated by compute methods
        IntensityThreshold double       % Scalar threshold (default: 0.01) used for binarization (or T* found by Ridgway)
        ConsensusRatio double           % Used only for Consensus masks (default: 0.70)
        Matrix logical                  % 3D boolean tensor representing the mask
        RefInfo struct                  % Spatial metadata from CohortData object
    end
    
    properties (Access = private)
        PrivateLogger Logger    % Custom Logger
    end
    
    methods (Access = public)
        
        function obj = BrainMask(refInfo, loggerObj)
            % CONSTRUCTOR: Initializes the mask object.
            arguments
                refInfo (1,1) struct {mustBeNonempty}
                loggerObj (1,1) Logger = Logger('NullLogger') % Fallback instantiation
            end
          
            obj.RefInfo = refInfo;
            obj.PrivateLogger = loggerObj;
        end
        
        function computeMeanMask(obj, cohortDataObj, threshold)
            % METHOD: computeMeanMask
            % PURPOSE: Averages the entire 4D cohort tensor and binarizes it
            arguments
                obj BrainMask
                cohortDataObj CohortData
                threshold (1,1) double {mustBePositive} = 0.01
            end
            
            obj.PrivateLogger.info('Computing Mean Mask (Threshold: %.3f)...', threshold);
            obj.MaskType = 'Mean Mask';
            obj.IntensityThreshold = threshold;
            
            % Delegate averaging to CohortData
            meanVol3D = cohortDataObj.getMeanVolume('ALL');
            
            % Binarization
            obj.Matrix = meanVol3D >= obj.IntensityThreshold;
            
            obj.PrivateLogger.success('%s computed successfully.', obj.MaskType);
        end
        
        function computeConsensusMask(obj, cohortDataObj, threshold, consensusRatio)
            % METHOD: computeConsensusMask
            % PURPOSE: Creates a boolean mask for each subject, then keeps 
            %   voxels present in at least 'consensusRatio' of the subjects.
            arguments
                obj BrainMask
                cohortDataObj CohortData
                threshold (1,1) double {mustBePositive} = 0.01
                consensusRatio (1,1) double {mustBeGreaterThanOrEqual(consensusRatio, 0.0),...
                            mustBeLessThanOrEqual(consensusRatio, 1.0)} = 0.70
            end

            obj.PrivateLogger.info('Computing Consensus Mask (Threshold: %.3f and Ratio: %.2f)...', threshold, consensusRatio);
            obj.MaskType = 'Consensus Mask';
            obj.IntensityThreshold = threshold;
            obj.ConsensusRatio = consensusRatio;
            
            % Delegate loading to CohortData
            allVols4D = cohortDataObj.getVolumes('ALL');
            numSubjects = size(allVols4D, 4);
            
            % Creates a 4D logical tensor
            boolean4D = allVols4D >= obj.IntensityThreshold;
            
            % Sum along 4th dimension gives the vote map
            voteMap3D = sum(boolean4D, 4);
            
            % Apply consensus threshold
            requiredVotes = ceil(numSubjects * obj.ConsensusRatio);
            obj.Matrix = voteMap3D >= requiredVotes;
            
            obj.PrivateLogger.success('%s computed successfully.', obj.MaskType);
        end
        
        function computeTpmMask(obj, tpmPath, threshold)
            % METHOD: computeTpmMask
            % PURPOSE: Loads the SPM Tissue Probability Map (TPM) and binarizes it.
            arguments
                obj BrainMask
                tpmPath (1,:) char {mustBeNonempty}
                threshold (1,1) double {mustBePositive} = 0.01
            end
            
            obj.MaskType = 'TPM Mask';
            obj.IntensityThreshold = threshold;

            % Delegate to private helper for TPM loading
            tpmData = obj.loadTpmVolume(tpmPath);
            
            obj.Matrix = tpmData >= obj.IntensityThreshold;
            obj.PrivateLogger.success('%s computed successfully (Threshold: %.3f)', obj.MaskType, obj.IntensityThreshold);
        end
        
        function computeRidgwayMeanMask(obj, cohortDataObj, stepSize)
            % METHOD: computeRidgwayMeanMask
            % PURPOSE: Extracts cohort mean and delegates to Ridgway
            %   optimizer the threshold research.
            arguments
                obj BrainMask
                cohortDataObj CohortData
                stepSize (1,1) double {mustBePositive} = 0.01
            end
            
            obj.PrivateLogger.info('Starting Ridgway Optimization on Mean Volume...');
            
            % Delegate averaging to CohortData
            meanVol3D = cohortDataObj.getMeanVolume('ALL');
            
            % Ridgway optimization
            [optThresh, maxCorr] = obj.optimizeRidgwayThreshold(meanVol3D, stepSize);
            
            obj.MaskType = 'Ridgway Mean Mask';
            obj.IntensityThreshold = optThresh;
            obj.Matrix = meanVol3D >= obj.IntensityThreshold;
            
            obj.PrivateLogger.success('%s computed successfully. Optimal T* = %.3f (Pearson r = %.3f)', ...
                obj.MaskType, obj.IntensityThreshold, maxCorr);
        end
        
        function computeRidgwayTpmMask(obj, tpmPath, stepSize)
            % METHOD: computeRidgwayTpmMask
            % PURPOSE: Loads TPM via SPM and delegates to Ridgway optimizer 
            %   the threshold research.
            arguments
                obj BrainMask
                tpmPath (1,:) char {mustBeNonempty}
                stepSize (1,1) double {mustBePositive} = 0.01
            end
            
            obj.PrivateLogger.info('Starting Ridgway Optimization on TPM Volume...');
            
            % Delegate to private helper for TPM loading
            tpmData = obj.loadTpmVolume(tpmPath);
            
            % Ridgway optimization
            [optThresh, maxCorr] = obj.optimizeRidgwayThreshold(tpmData, stepSize);
            
            obj.MaskType = 'Ridgway TPM Mask';
            obj.IntensityThreshold = optThresh;
            obj.Matrix = tpmData >= obj.IntensityThreshold;
            
            obj.PrivateLogger.success('%s computed successfully. Optimal T* = %.3f (Pearson r = %.3f)', ...
                obj.MaskType, obj.IntensityThreshold, maxCorr);
        end
        
        function stats = getMaskStats(obj)
            % METHOD: getMaskStats
            % PURPOSE: Returns volumetric metrics of the instantiated mask.
            arguments
                obj BrainMask
            end
            
            stats = struct();
            % Since Matrix is logical, summing it yields the active voxel count
            stats.ActiveVoxels = sum(obj.Matrix(:)); 
            stats.TotalVoxels = numel(obj.Matrix);
            stats.ActivePercentage = (stats.ActiveVoxels / stats.TotalVoxels) * 100;
            
            obj.PrivateLogger.info('%s has %d active voxels (%.2f%% of total volume).', ...
                obj.MaskType, stats.ActiveVoxels, stats.ActivePercentage);
        end

        function leakageArray = evaluateLeakage(obj, cohortDataObj, groupSelector)
            % METHOD: evaluateLeakage
            % PURPOSE: Quantifies how many voxels within the mask are structurally 
            %   zero in the cohort/group subjects.
            arguments
                obj BrainMask
                cohortDataObj CohortData
                groupSelector (1,:) char {mustBeMember(groupSelector, {'AD', 'CTRL', 'ALL'})} = 'ALL'
            end
            
            % Delegate loading to CohortData
            vols4D = cohortDataObj.getVolumes(groupSelector);

            maskSize = sum(obj.Matrix(:)); 
            
            % Compute the background voxels of each cohort/group subject that
            % are in the mask. Uses MATLAB Implicit Expansion which allows 
            % operations between a 3D tensor and a 4D tensor by copying the
            % former along the 4th dimension.
            leakedVoxels4D = (obj.Matrix == true) & (vols4D == 0);
            
            % Sum across spatial dimensions (X, Y and Z) and squeeze it
            leakedPerSubject = squeeze(sum(leakedVoxels4D, [1, 2, 3]));
            
            % Compute the percentage of leaked voxels
            leakageArray = (leakedPerSubject / maskSize) * 100;
        end

        function exportToNifti(obj, outPath)
            % METHOD: exportToNifti
            % PURPOSE: Exports the current binarized mask to disk as a NIfTI file.
            %    If folder is missing, creates the directory structure on the fly.
            arguments
                obj BrainMask
                outPath (1,:) char {mustBeNonempty}
            end
            
            obj.PrivateLogger.info('Attempting to export %s to: %s', obj.MaskType, outPath);
            
            % Must cast logical to single/double as NIfTI standards require numeric types
            exportData = single(obj.Matrix);
            
            try
                niftiwrite(exportData, outPath, obj.RefInfo.RawNiftiInfo);
                obj.PrivateLogger.success('NIfTI file exported successfully.');
            catch ME
                % Filtering for missing paths/folders and
                % cannotOpenHeaderWrite niftirwite error
                identifier = lower(ME.identifier);
                if contains(identifier, 'file') || ...
                   contains(identifier, 'path') || ...
                   contains(identifier, 'folder') || ...
                   contains(identifier, 'cannotopen')

                    [folderPath, ~, ~] = fileparts(outPath);
                    try
                        % Fallback: Create parent directory and retry
                        mkdir(folderPath);
                        niftiwrite(exportData, outPath, obj.RefInfo.RawNiftiInfo);
                        obj.PrivateLogger.success('Missing folder automatically created. NIfTI exported successfully.');
                    catch fatalME
                        obj.PrivateLogger.error('Fatal I/O error during NIfTI export fallback: %s', fatalME.identifier);
                        rethrow(fatalME);
                    end
                else
                    % Rethrow for unrelated bugs
                    rethrow(ME);
                end
            end
        end
    end

    methods (Access = private)

        function tpmData = loadTpmVolume(obj, tpmPath)
            % HELPER: loadTpmVolume
            % PURPOSE: Centralizes SPM TPM loading and spatial alignment verifications.
            
            obj.PrivateLogger.info('Attempting to load SPM Tissue Probability Map (TPM) from: %s', tpmPath);

            try
                loadedHeader = spm_vol(tpmPath);
                if numel(loadedHeader) > 1
                    obj.PrivateLogger.info('Detected a 4D NIfTI file with %d volumes. Forcing extraction of Volume 1 (Gray Matter).', numel(loadedHeader));
                    loadedHeader = loadedHeader(1);
                end
                tpmData = single(spm_read_vols(loadedHeader));
            catch ME
                obj.PrivateLogger.error('Failed to read external TPM file via SPM API.');
                rethrow(ME);
            end

            % Dimension and Affine Matrix Check
            obj.verifySpatialAlignment(loadedHeader, tpmData);

            % Clean up for NaNs
            tpmData(isnan(tpmData)) = 0;
        end

        function [optimalThreshold, maxCorrelation] = optimizeRidgwayThreshold(obj, volume3D, stepSize)
            % HELPER: optimizeRidgwayThreshold
            % PURPOSE: The core method for Ridgway optimization.
            
            flatVolume = volume3D(:); 
            
            % Variance check: zero variance -> NaN Pearson corr
            if var(flatVolume) == 0
                obj.PrivateLogger.error('The reference volume lacks variance.');
                error('BrainMask:ConstantVolume', 'Input yields a constant volume.');
            end
            
            % Normalize the search space relative to the maximum intensity
            % of the volume
            maxIntensity = max(flatVolume);
            thresholdRange = stepSize : stepSize : (maxIntensity - stepSize);
            
            % Initialize tracking variables
            maxCorrelation = -1.0;
            optimalThreshold = stepSize;
            
            obj.PrivateLogger.info('Computing Ridgway Threshold. Searching %d thresholds...', numel(thresholdRange));
            step = max(1, floor(numel(thresholdRange)/4));
            
            for i = 1:numel(thresholdRange)
                if mod(i, step) == 0
                    obj.PrivateLogger.debug('Ridgway Scan: %d / %d (%.0f%%)', i, numel(thresholdRange), (i/numel(thresholdRange))*100);
                end
                
                % Generate candidate binary vector
                currentThreshold = thresholdRange(i);
                candidateBinary = single(flatVolume >= currentThreshold);
                
                % Ensure the binary vector is not completely uniform (all 0s or all 1s)
                if var(candidateBinary) > 0
                    % Compute Pearson correlation matrix. Index (1,2) is the cross-correlation
                    corrMat = corrcoef(flatVolume, candidateBinary);
                    currentCorr = corrMat(1, 2);
                    
                    % Update optimal values if a new maximum is found
                    if currentCorr > maxCorrelation
                        maxCorrelation = currentCorr;
                        optimalThreshold = currentThreshold;
                    end
                end
            end
        end

        function verifySpatialAlignment(obj, loadedHeader, loadedData)
            % HELPER: verifySpatialAlignment
            % PURPOSE: Ensure loaded external maps (e.g. TPM) match the cohort's spatial space
            
            % Dimension Check
            if ~isequal(size(loadedData), obj.RefInfo.ImageSize)
                obj.PrivateLogger.error('Dimensions mismatch: Loaded %s vs Cohort %s', ...
                    mat2str(size(loadedData)), mat2str(obj.RefInfo.ImageSize));
                error('BrainMask:DimensionMismatch', 'Spatial dimensions are incompatible.');
            end
            
            % Affine Matrix Check (floating point tolerance)
            tolerance = 1e-4;
            refMat = obj.RefInfo.NumericMatrix; 
            maxDeviation = max(abs(loadedHeader.mat(:) - refMat(:)));
            if maxDeviation > tolerance
                obj.PrivateLogger.error('Affine matrices differ significantly. Max deviation: %f', maxDeviation);
                error('BrainMask:AffineMismatch', 'The external mask and the cohort volumes are not aligned. Please coregister them first or check if you are comparing SPM and MATLAB parserers.');
            end
        end
    end
end