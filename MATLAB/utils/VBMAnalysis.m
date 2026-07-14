classdef VBMAnalysis < handle
    % VBMANALYSIS Encapsulates SPM25 statistical design and thresholding.
    %
    % PURPOSE: Wraps the GLM model estimation and the derivation of
    %   corrected statistical thresholds (FDR/FWE). Receives CohortData 
    %   directly to build the GLM design matrix automatically.
    
    properties (Access = private)
        PrivateLogger Logger     % Custom Logger
    end
    
    methods (Access = public)
        function obj = VBMAnalysis(loggerObj)
             % CONSTRUCTOR: Initializes the VBMAnalysis object
            arguments
                loggerObj (1,1) Logger = Logger('NullLogger')
            end
            obj.PrivateLogger = loggerObj;
        end

        function twoSampleTTest(obj, outputDir, cohortObj, maskPath, group1Name, group2Name)
            % METHOD: fitTwoSampleTTest
            % PURPOSE: Dynamically constructs and estimates the SPM Factorial Design 
            %   (Two-Sample T-Test) by extracting data directly from the CohortData object.
            arguments
                obj VBMAnalysis
                outputDir (1,:) char {mustBeNonempty}
                cohortObj CohortData
                maskPath (1,:) char {mustBeNonempty}
                group1Name (1,:) char = 'AD'
                group2Name (1,:) char = 'CTRL'
            end
            
            obj.PrivateLogger.info('Initializing SPM VBM Analysis...');

            % Check if SPM is in path
            if isempty(which('spm'))
                obj.PrivateLogger.error('SPM is not installed or not in the MATLAB path.');
                error('VBMAnalysis:SpmMissing', 'SPM must be added to the path before running VBM Analysis.');
            end
            
            % Directory management
            if ~exist(outputDir, 'dir')
                mkdir(outputDir);
            end
            
            % Initialize SPM
            spm('defaults', 'PET');
            spm_jobman('initcfg');
            
            % Extract the paths and covariates directly
            outDirCell = cellstr(outputDir);
            maskPathCell = cellstr(maskPath);
            scansG1 = cellstr(cohortObj.getFilePaths(group1Name));
            scansG2 = cellstr(cohortObj.getFilePaths(group2Name));
            covTable = cohortObj.getCovariatesTable();
            
            % --- Module 1: Factorial Design Specification ---
            % Initialize Factorial Design
            matlabbatch{1}.spm.stats.factorial_design.dir = outDirCell(:);
            matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = scansG1(:);
            matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = scansG2(:);
            matlabbatch{1}.spm.stats.factorial_design.des.t2.dept = 0;          % Independent groups
            matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1;      % Unequal variance 
            matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca = 0;         % No Grand Mean Scaling 
            matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova = 0;        % No ANCOVA

            % Dynamic Covariates Injection (Age, Sex, TIV)
            matlabbatch{1}.spm.stats.factorial_design.cov(1).c = covTable.Age;
            matlabbatch{1}.spm.stats.factorial_design.cov(1).cname = 'Age';
            matlabbatch{1}.spm.stats.factorial_design.cov(1).iCFI = 1;          % Interaction among covariates: None
            matlabbatch{1}.spm.stats.factorial_design.cov(1).iCC = 1;           % Centering: Overall mean
            matlabbatch{1}.spm.stats.factorial_design.cov(2).c = covTable.Sex;
            matlabbatch{1}.spm.stats.factorial_design.cov(2).cname = 'Sex';
            matlabbatch{1}.spm.stats.factorial_design.cov(2).iCFI = 1;
            matlabbatch{1}.spm.stats.factorial_design.cov(2).iCC = 5;           % No Centering
            matlabbatch{1}.spm.stats.factorial_design.cov(3).c = covTable.TIV;
            matlabbatch{1}.spm.stats.factorial_design.cov(3).cname = 'TIV';
            matlabbatch{1}.spm.stats.factorial_design.cov(3).iCFI = 1;
            matlabbatch{1}.spm.stats.factorial_design.cov(3).iCC = 1;

            % Masking using explicit mask (from BainMask object) (zero-var voxels will be excluded anyway)
            matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;           % No absolute thresholding
            matlabbatch{1}.spm.stats.factorial_design.masking.im = 0;                   % Implicit masking off
            matlabbatch{1}.spm.stats.factorial_design.masking.em = maskPathCell(:);     % Explicit mask 3D volume
            
            % No global calculation and normalization
            matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
            matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
            matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

            % --- Module 2: Model Estimation ---
            % Take SPM.mat output from matlabbatch{1}
            matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('Factorial design specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
            matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;  % No write images of residuals to disk
            matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;  % Restricted Maximum Likelihood estimation
            
            % --- Module 3: Contrast Manager ---
            % Take SPM.mat output from matlabbatch{2}
            matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
            
            % Atrophy Contrast (e.g., CTRL > AD)
            % Weights: [AD CTRL Age Sex TIV] -> [-1 1 0 0 0]
            contrastTitle = sprintf('Atrophy: %s > %s', group2Name, group1Name);
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = contrastTitle;
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [-1 1 0 0 0];
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
            matlabbatch{3}.spm.stats.con.delete = 1;    % Delete existing contrasts to have only the last contast
            
            obj.PrivateLogger.info('SPM Batch successfully configured. Executing GLM...');
            
            % Run the matlabbatch
            try
                spm_jobman('run', matlabbatch);
                obj.PrivateLogger.success('SPM GLM estimation completed successfully. Results saved in: %s', outputDir);
            catch ME
                obj.PrivateLogger.error('SPM GLM estimation crashed: %s', ME.identifier);
                rethrow(ME);
            end
        end
        
        function [thresholdedMap, statThreshold] = getCorrectedMap(obj, outputDir, contrastName, alpha, correctionMode, exportPath)
            % METHOD: getCorrectedMap
            % PURPOSE: Extracts the corrected continuous T-map and calculates the 
            %       threshold based on FWE/FDR corrections.
            arguments
                obj VBMAnalysis
                outputDir (1,:) char {mustBeNonempty}
                contrastName (1, :) char {mustBeNonempty}
                alpha (1, 1) double {mustBePositive, mustBeLessThan(alpha, 1)}
                correctionMode (1, :) char {mustBeMember(correctionMode, {'FDR', 'FWE', 'none'})}
                exportPath (1,:) char = ''
            end
            
            obj.PrivateLogger.info('Computing %s correction (alpha = %.3f) for contrast: %s', correctionMode, alpha, contrastName);
            
            % Check if SPM is in path
            if isempty(which('spm'))
                obj.PrivateLogger.error('SPM is not installed or not in the MATLAB path.');
                error('VBMAnalysis:SpmMissing', 'SPM must be added to the path before running VBM Analysis.');
            end
            
            % Load SPM.mat
            spmMatPath = fullfile(outputDir, 'SPM.mat');
            try
                spmData = load(spmMatPath);
            catch ME
                obj.PrivateLogger.error('Cannot load SPM.mat at %s', spmMatPath);
                rethrow(ME);
            end
            SPM = spmData.SPM;
            
            % Match contrast name
            availableContrasts = {SPM.xCon.name};
            matchLogical = strcmpi(availableContrasts, contrastName);
            resolvedIdx = find(matchLogical);
            if isempty(resolvedIdx)
                obj.PrivateLogger.error('Contrast "%s" not found in SPM.mat.', contrastName);
                error('VBMAnalysis:ContrastNotFound', 'Requested contrast is missing.');
            end
            targetContrast = SPM.xCon(resolvedIdx);
            
            % Extract spmT_*.nii path
            tMapPath = fullfile(outputDir, targetContrast.Vspm.fname);

            % Extract d.f. = [Effective Interest d.f., Effective Residual d.f.]
            degreesOfFreedom = [targetContrast.eidf, SPM.xX.erdf];

            % Extraxt 'T' Stat
            statDistributionType = targetContrast.STAT;
            
            % --- Compute statistical threshold ---
            % False Discovery Rate (Benjamini-Hochberg) Correction
            if strcmpi(correctionMode, 'FDR')
                try
                    % Extract spmT_*.nii header
                    volHeader = spm_vol(tMapPath);
                catch ME
                    obj.PrivateLogger.error('Cannot read t-map header for FDR: %s', tMapPath);
                    rethrow(ME);
                end
                % Vm = 0 because no implicit mask needed
                % n = 1 because only one null hypothesis to test
                statThreshold = spm_uc_FDR(alpha, degreesOfFreedom, statDistributionType, 1, volHeader, 0);
            
            % Family-Wise Error Correction
            elseif strcmpi(correctionMode, 'FWE')
                % SPM.xVol.R = resolution element counts
                % SPM.xVol.S = Mask voxel count
                statThreshold = spm_uc(alpha, degreesOfFreedom, statDistributionType, SPM.xVol.R, 1, SPM.xVol.S);
            
            % Uncorrected
            else 
                statThreshold = spm_u(alpha, degreesOfFreedom, statDistributionType);
            end
            
            % Load t-map volume 
            try
                tHeader = spm_vol(tMapPath); 
                tVolume = spm_read_vols(tHeader);
            catch ME
                obj.PrivateLogger.error('Cannot read t-map volume: %s', tMapPath);
                rethrow(ME);
            end
            
            % Sanification
            tVolume(isnan(tVolume)) = 0;
            
            % Apply threshold
            thresholdedMap = single(tVolume);
            thresholdedMap(thresholdedMap < statThreshold) = 0;
            
            % Warning if the threshold killed all significance
            if max(thresholdedMap(:)) == 0
                obj.PrivateLogger.warning('Zero voxels survived the %s threshold (alpha=%.3f). Map is completely empty.', correctionMode, alpha);
            end

            obj.PrivateLogger.success('Statistical threshold computed successfully: T > %.3f', statThreshold);
        
            % --- Export to NIfTI if requested ---
            if ~isempty(exportPath)
                obj.PrivateLogger.info('Exporting thresholded map to: %s', exportPath);
                try
                    % Safely create parent directories
                    [exportDir, ~, ~] = fileparts(exportPath);
                    if ~isempty(exportDir) && ~exist(exportDir, 'dir')
                        mkdir(exportDir);
                    end
                    
                    % Clone and modify native SPM header
                    outHeader = tHeader;
                    outHeader.fname = exportPath;
                    outHeader.descrip = sprintf('SPM VBM %s corrected map (alpha=%.3f)', correctionMode, alpha);
                    outHeader.dt = [16, 0]; % Single precision

                    spm_write_vol(outHeader, thresholdedMap);
                    obj.PrivateLogger.success('NIfTI file exported successfully.');
                catch writeME
                    obj.PrivateLogger.error('Failed to export map to %s', exportPath);
                    rethrow(writeME);
                end
            end
        end

        function evaluateMapSimilarity(obj, mapA, mapB, nameA, nameB)
            % METHOD: evaluateMapSimilarity
            % PURPOSE: Calculates Dice, Jaccard, and Inclusion ratios for two thresholded 3D 
            %   statistical maps. Also computes the Pearson correlation of the t-values 
            %   within the intersecting volume.
            arguments
                obj VBMAnalysis
                mapA (:,:,:) double {mustBeNonempty}
                mapB (:,:,:) double {mustBeNonempty}
                nameA (1,:) char = 'Map A'
                nameB (1,:) char = 'Map B'
            end
            
            if ~isequal(size(mapA), size(mapB))
                obj.PrivateLogger.error('Similarity arrays dimension mismatch.');
                error('VBMAnalysis:DimensionMismatch', 'Maps must have identical dimensions. %s: %s, %s: %s', ...
                    nameA, mat2str(size(mapA)), nameB, mat2str(size(mapB)));
            end
            
            % Binarize Maps
            maskA = mapA > 0; 
            maskB = mapB > 0;
            
            volA = sum(maskA(:)); 
            volB = sum(maskB(:));
            
            intersection = sum(maskA(:) & maskB(:)); 
            unionVol = sum(maskA(:) | maskB(:));
            
            % Prevent NaN if both maps are completely empty
            if unionVol == 0
                obj.PrivateLogger.warning('Both maps are entirely empty. Similarity metrics are undefined.');
                return;
            end
            
            % Calculate spatial metrics
            dice = (2 * intersection) / (volA + volB);
            jaccard = intersection / unionVol;
            if volA > 0, inclusionA_in_B = intersection / volA; else, inclusionA_in_B = 0; end
            if volB > 0, inclusionB_in_A = intersection / volB; else, inclusionB_in_A = 0; end
            
            % Calculate Pearson correlation
            if intersection > 3  % to avoid non-significant results
                % Extract continuous t-values where both maps are active
                tValsA = mapA(maskA & maskB); 
                tValsB = mapB(maskA & maskB);
                % Do the areas of atrophy scale in the same way? 
                intensityCorr = corr(tValsA, tValsB, 'Type', 'Pearson');
            else
                intensityCorr = NaN; % Not enough points for valid correlation
            end
            
            obj.PrivateLogger.info('Similarity [%s] vs [%s]', nameA, nameB);
            obj.PrivateLogger.info('Active Voxels: %s = %d | %s = %d | Intersection: %d voxels',...
                    nameA, volA, nameB, volB, intersection);
            obj.PrivateLogger.info('Dice: %.4f | Jaccard: %.4f',  dice, jaccard);
            obj.PrivateLogger.info('Inclusion: %.2f%% of %s is inside %s | %.2f%% of %s is inside %s',...
                    inclusionA_in_B * 100, nameA, nameB, inclusionB_in_A * 100, nameB, nameA)
            obj.PrivateLogger.info('Pearson Corr in intersection: %.4f', intensityCorr);
        end
    end
end