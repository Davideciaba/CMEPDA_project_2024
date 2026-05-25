%% Tests/TestNiftiinfo.m
% -------------------------------------------------------------------------
% This script processes NIfTI files for Alzheimer's Disease (AD) and 
% Control (CTRL) subjects. It calculates ........ It creates binary masks based on an 
% intensity threshold and generates visualizations for quality control 
% (slice overlays and histograms).
% -------------------------------------------------------------------------

% Setup Paths and Directories
scriptPath = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptPath);

% Define directories for Alzheimer's Disease (AD) and Control (CTRL) groups
dirAD_CTRL   = fullfile(projectRoot, 'AD_CTRL', 'AD_CTRL_nii');

% -------------------------------------------------------------------------

% Define file paths for the AD e CTRL files
files_AD = dir(fullfile(dirAD_CTRL, 'smwc1AD-*.nii'));
files_CTRL = dir(fullfile(dirAD_CTRL, 'smwc1CTRL-*.nii'));

% Combine file lists into a single structure
all_files_struct = [files_AD; files_CTRL];
GM_files = fullfile({all_files_struct.folder}', {all_files_struct.name}');
N = numel(GM_files);


% Check if files exist
if N == 0
    error('No NIfTI files found in %s or %s', dirAD, dirCTRL);
end
fprintf('Found %d subjects.\n', N);

% -------------------------------------------------------------------------

% Read the first NIfTI file and extract image dimensions
info = niftiinfo(GM_files{1});

% Initialize accumulators for raw and normalized volumes
sum_vol = zeros(info.ImageSize);

fprintf('Computing Mean Mask...\n');
fprintf('Progress: |');
step = floor(N/20); % Visual progress bar step

for i = 1:N
    if mod(i, step) == 0, fprintf('='); end
    
    % Read volume data
    vol = niftiread(GM_files{i});

    % Replace NaNs with 0
    vol(isnan(vol)) = 0;
    
    % Accumulate volumes
    sum_vol = sum_vol + vol;
end
fprintf('| \n');

% Calculate the mean
mean_GM = sum_vol / N;


% -------------------------------------------------------------------------


% --- Statistics ---
[M_mean, total_mean, pct_mean] = analyzeVoxelVolume(mean_GM, 'mean_GM');

AD_1 = niftiread(GM_files{79});
[M_AD, total_AD, pct_AD] = analyzeVoxelVolume(AD_1, all_files_struct(79).name);

CTRL_1 = niftiread(GM_files{164});
[M_CTRL, total_CTRL, pct_CTRL] = analyzeVoxelVolume(CTRL_1, all_files_struct(164).name);


thr = 0.01; % Intensity threshold
% Create binary masks
mask = mean_GM > thr;
[M_mask, total_mask, pct_mask] = analyzeVoxelVolume(mask, 'Mask');
[percentages_mean, worst_idx_mean] = evaluateDataLeakage(GM_files, mask, thr);
CONSENSUS_RATIO = 0.7;
[consensus_mask, vote_map] = buildConsensusMask(GM_files, thr, CONSENSUS_RATIO);
[M_mask_cons, total_mask_cons, pct_mask_cons] = analyzeVoxelVolume(consensus_mask, 'Consensus_Mask');
[percentages_cons, worst_idx_cons] = evaluateDataLeakage(GM_files, consensus_mask, thr);

% --- NUOVA IMPLEMENTAZIONE RIDGWAY ---
% Compute the objective mask using a dynamic optimal threshold
step_size = 0.01;
[mask_rig, opt_thr, max_corr] = computeObjectiveMask(mean_GM, step_size);

% Visualizzazione dei risultati tramite la tua funzione custom
[M_mask_rig, total_mask_rig, pct_mask_rig] = analyzeVoxelVolume(mask_rig, 'ObjectiveMask_Ridgway');
[percentages_rig, worst_idx_rig] = evaluateDataLeakage(GM_files, mask_rig, opt_thr);

% Define TPM Path dynamically from SPM installation
spmInstallDir = fileparts(which('spm'));
if isempty(spmInstallDir)
    error('FeatureExtraction:SpmNotDetected', 'SPM is not in the MATLAB path.');
end
tpmFilePath = fullfile(spmInstallDir, 'tpm', 'TPM.nii');

[tpmMask, ~] = buildTpmMask(tpmFilePath, GM_files{1}, thr);
[M_mask_TPM, total_mask_TPM, pct_mask_TPM] = analyzeVoxelVolume(tpmMask, 'TPM_Mask');
[percentages_TPM, worst_idx_TPM] = evaluateDataLeakage(GM_files, tpmMask, thr);


if ~contains(tpmFilePath, ',')
      tpmPathGM = [tpmFilePath, ',1'];
else
      tpmPathGM = tpmFilePath;
end
headerTpm = spm_vol(tpmPathGM);
tpmVol = spm_read_vols(headerTpm);
tpmVol(isnan(tpmVol)) = 0;
[mask_rig_TPM, opt_thr_TPM, max_corr_TPM] = computeObjectiveMask(tpmVol, step_size);
[M_mask_rigTPM, total_mask_rigTPM, pct_mask_rigTPM] = analyzeVoxelVolume(mask_rig_TPM, 'TPM_Mask');
[percentages_rigTPM, worst_idx_rigTPM] = evaluateDataLeakage(GM_files, mask_rig_TPM, opt_thr_TPM);

% --- Data Splitting for Unified Group Visualization ---
numAD = numel(files_AD); % Number of Alzheimer's subjects

% Split the linear output vectors into AD and CTRL subsets
leakageAD_mean  = percentages_mean(1:numAD);
leakageCTRL_mean = percentages_mean(numAD+1:end);

leakageAD_cons  = percentages_cons(1:numAD);
leakageCTRL_cons = percentages_cons(numAD+1:end);

leakageAD_rig   = percentages_rig(1:numAD);
leakageCTRL_rig  = percentages_rig(numAD+1:end);

leakageAD_TPM   = percentages_TPM(1:numAD);
leakageCTRL_TPM  = percentages_TPM(numAD+1:end);

leakageAD_rigTPM   = percentages_rigTPM(1:numAD);
leakageCTRL_rigTPM  = percentages_rigTPM(numAD+1:end);

% Group arrays for the plotter function
dataADCell   = {leakageAD_mean, leakageAD_cons, leakageAD_rig, leakageAD_TPM, leakageAD_rigTPM};
dataCTRLCell = {leakageCTRL_mean, leakageCTRL_cons, leakageCTRL_rig, leakageCTRL_TPM, leakageCTRL_rigTPM};
methodLabels = {'Mean Mask', 'Consensus Mask', 'Ridgway Mask', 'TPM Mask', 'TPM Rig Mask'};
groupColors  = {'r', 'b'}; % Red for AD, Blue for CTRL

% Generate the dashboard
figLeakage = plotUnifiedGroupLeakageHistograms(dataADCell, dataCTRLCell, methodLabels, groupColors);
% -------------------------------------------------------------------------

% --- Raw Data Visualization ---
mid_slice_idx = round(size(mean_GM, 3) * 0.33);
slice_mean = mean_GM(:, :, mid_slice_idx);
slice_mask = mask(:, :, mid_slice_idx);
slice_AD = AD_1(:, :, mid_slice_idx);
slice_CTRL = CTRL_1(:, :, mid_slice_idx);
slice_consensus = consensus_mask(:, :, mid_slice_idx); 
slice_rig = mask_rig(:, :, mid_slice_idx);
slice_TPM = tpmMask(:, :, mid_slice_idx);
slice_rigTPM = mask_rig_TPM(:, :, mid_slice_idx); 
color_limits = [0, max([slice_mean(:); slice_AD(:); slice_CTRL(:); slice_consensus(:); slice_rig(:); slice_TPM(:); slice_rigTPM(:)])];

% Array Base
baseImages = {slice_mean, slice_AD, slice_CTRL};
baseTitles = {'Mean Map', 'AD_1 Map', 'CTRL_1 Map'};

% Array Maschere
masksToTest = {slice_mask, slice_consensus, slice_rig, slice_TPM, slice_rigTPM};
maskColors  = {'r', 'g', 'b', 'y' 'm'};

% Creazione etichette usando i tuoi threshold specifici
labelMask = sprintf('Mask Threshold = %.3f', thr);
labelCons = sprintf('Consensus Threshold = %.3f', thr);
labelRig = sprintf('Ridgway Threshold = %.3f', opt_thr);
labelTPM = sprintf('TPM Threshold = %.3f', thr);
label_rigTPM = sprintf('RigTPM Threshold = %.3f', opt_thr_TPM);
maskLabels  = {labelMask, labelCons, labelRig, labelTPM, label_rigTPM};

fig1 = plotMaskOverlays(baseImages, baseTitles, masksToTest, maskColors, maskLabels);


% -------------------------------------------------------------------------

% --- Raw Histogram ---
bins = 0 : 0.01 : max([mean_GM(:); AD_1(:); CTRL_1(:)]);

colorsList = {'r', 'g', 'b', 'y', 'm'};
labelsList = {'Masked', 'Consensus_Masked', 'Ridgway_Masked', 'TPM_Masked', 'TPM_rig_Masked'};

figHistograms = plotMaskedHistograms(baseImages, baseTitles, ...
                                     masksToTest, colorsList, labelsList, ...
                                     bins, thr);



function [lostPercentages, worstSubjectsIdx] = evaluateDataLeakage(filePaths, referenceMask, intensityThreshold)
%EVALUATEDATALEAKAGE Calculates the percentage of active Gray Matter voxels 
% excluded by a given reference group mask for each subject.
%
% PURPOSE:
%   Identifies severe anatomical mismatches (e.g., in advanced AD cases) 
%   where a group-level mask drops patient-specific viable tissue.
%
% PARAMS:
%   filePaths          - (cell array of strings) Full paths to NIfTI files.
%   referenceMask      - (3D logical array) The group mask to validate against.
%   intensityThreshold - (double) Minimum intensity to consider a voxel as GM.
%
% RETURNS:
%   lostPercentages    - (Nx1 double) Percentage of excluded GM per subject.
%   worstSubjectsIdx   - (Nx1 double) Indices of subjects sorted by highest loss.
%
% RAISES:
%   MATLAB:validators:mustBeNonempty - If input arrays are empty.

    arguments
        filePaths (:, 1) cell {mustBeNonempty}
        referenceMask (:, :, :) logical {mustBeNonempty}
        intensityThreshold (1, 1) double 
    end
    
    numSubjects = length(filePaths);
    lostVoxelsCount = zeros(numSubjects, 1);
    lostPercentages = zeros(numSubjects, 1);
    
    fprintf('Evaluating Data Leakage across %d subjects...\n', numSubjects);
    progressStep = max(1, floor(numSubjects / 20));
    fprintf('Progress: |');
    
    for i = 1:numSubjects
        if mod(i, progressStep) == 0, fprintf('='); end
        
        try
            % Read volume and clean NaNs
            currentVol = niftiread(filePaths{i});
            currentVol(isnan(currentVol)) = 0;
            
            % Define individual valid tissue mask
            individualMask = currentVol > intensityThreshold;
            
            % Identify voxels present in individual but MISSING in reference
            excludedVoxelsMap = individualMask & (~referenceMask);
            
            % Calculate metrics
            totalIndividualVoxels = sum(individualMask(:));
            if totalIndividualVoxels > 0
                lostVoxelsCount(i) = sum(excludedVoxelsMap(:));
                lostPercentages(i) = (lostVoxelsCount(i) / totalIndividualVoxels) * 100;
            else
                lostPercentages(i) = 0; % Edge case: empty volume
            end
            
        catch ME
            fprintf('\n[ERROR] Failed to read or process file: %s\n', filePaths{i});
            rethrow(ME);
        end
    end
    fprintf('|\n');
    
    % Sort results to easily identify the most problematic scans
    [~, worstSubjectsIdx] = sort(lostPercentages, 'descend');
    
end



function figHandle = plotUnifiedGroupLeakageHistograms(leakageADCell, leakageCTRLCell, methodLabels, groupColors)
%PLOTUNIFIEDGROUPLEAKAGEHISTOGRAMS Plots AD vs CTRL data leakage distributions.
%
% PURPOSE:
%   Generates a grid of overlapping histograms. Each subplot represents a 
%   masking method, displaying two semi-transparent histograms (AD vs CTRL) 
%   sharing the same X-axis (Leakage %) and Y-axis (Occurrences).
%
% PARAMS:
%   leakageADCell   - (1D cell array) Numeric arrays of leakage % for AD subjects.
%   leakageCTRLCell - (1D cell array) Numeric arrays of leakage % for CTRL subjects.
%   methodLabels    - (1D cell array) Strings representing the titles of each method.
%   groupColors     - (cell array) 2 Strings/Chars for colors (e.g., {'r', 'b'}).
%
% RETURNS:
%   figHandle       - (Figure) Handle to the generated figure object.
%
% RAISES:
%   plotUnifiedGroup:DimensionMismatch - If cell arrays do not match in length.

    arguments
        leakageADCell (1, :) cell {mustBeNonempty}
        leakageCTRLCell (1, :) cell {mustBeNonempty}
        methodLabels (1, :) cell {mustBeNonempty}
        groupColors (1, 2) cell = {'r', 'b'} % Default: Red (AD), Blue (CTRL)
    end
    
    numMethods = numel(methodLabels);
    
    % 1. Input Validation
    if (numel(leakageADCell) ~= numMethods) || (numel(leakageCTRLCell) ~= numMethods)
        error('plotUnifiedGroup:DimensionMismatch', ...
              'Data cells and methodLabels must have the exact same number of elements.');
    end
    
    % 2. Calculate Global Limits for robust binning and visual comparison
    globalMaxLeakage = 0;
    globalMaxOccurrences = 0;
    
    for i = 1:numMethods
        maxAD = max(leakageADCell{i});
        maxCTRL = max(leakageCTRLCell{i});
        globalMaxLeakage = max([globalMaxLeakage, maxAD, maxCTRL]);
    end
    
    % Define uniform bin edges (e.g., steps of 0.5% or dynamic based on max)
    % Ensuring at least 1% max to prevent linspace errors on 0-variance sets
    globalMaxLeakage = max(1.0, globalMaxLeakage * 1.05); 
    numBins = 30; 
    binEdges = linspace(0, globalMaxLeakage, numBins);
    
    % Pre-compute maximum occurrences (Y-axis) using histcounts with fixed bins
    for i = 1:numMethods
        countsAD = histcounts(leakageADCell{i}, binEdges);
        countsCTRL = histcounts(leakageCTRLCell{i}, binEdges);
        globalMaxOccurrences = max([globalMaxOccurrences, max(countsAD), max(countsCTRL)]);
    end
    
    unifiedYLimit = [0, globalMaxOccurrences * 1.1]; % Add 10% vertical padding
    unifiedXLimit = [0, globalMaxLeakage];
    
    % 3. Determine Dynamic Grid Layout
    numCols = ceil(sqrt(numMethods));
    numRows = ceil(numMethods / numCols);
    
    % 4. Figure Initialization
    figHandle = figure('Name', 'Unified_AD_vs_CTRL_Leakage', ...
                       'Color', 'w', 'Position', [100, 100, 1200, 800]);
    
    % 5. Subplot Rendering Loop
    for idx = 1:numMethods
        subplot(numRows, numCols, idx);
        hold on;
        
        dataAD = leakageADCell{idx};
        dataCTRL = leakageCTRLCell{idx};
        
        % Plot AD Histogram (Alpha blended)
        histogram(dataAD, binEdges, 'FaceColor', groupColors{1}, ...
                  'FaceAlpha', 0.5, 'EdgeColor', 'none', ...
                  'DisplayName', 'AD Subjects');
              
        % Plot CTRL Histogram (Alpha blended)
        histogram(dataCTRL, binEdges, 'FaceColor', groupColors{2}, ...
                  'FaceAlpha', 0.5, 'EdgeColor', 'none', ...
                  'DisplayName', 'CTRL Subjects');
              
        % Overlay outline for clarity on dense bins
        histogram(dataAD, binEdges, 'DisplayStyle', 'stairs', ...
                  'EdgeColor', groupColors{1}, 'LineWidth', 1.5, 'HandleVisibility', 'off');
        histogram(dataCTRL, binEdges, 'DisplayStyle', 'stairs', ...
                  'EdgeColor', groupColors{2}, 'LineWidth', 1.5, 'HandleVisibility', 'off');

        % Subplot Formatting
        title(sprintf('%s', methodLabels{idx}), 'Interpreter', 'none', ...
              'FontSize', 12, 'FontWeight', 'bold');
        xlabel('Data Leakage (%)', 'FontSize', 10);
        ylabel('Occurrences (Subjects)', 'FontSize', 10);
        
        % Lock Axes to global scale
        xlim(unifiedXLimit);
        ylim(unifiedYLimit);
        
        grid on;
        
        % Add legend only to the first subplot to save space, or all if preferred
        legend('Location', 'northeast');
        
        hold off;
    end
    
    % Super Title
    sgtitle('Data Leakage Distribution: AD vs CTRL Comparison', 'FontSize', 14, 'FontWeight', 'bold');
end



function [consensusMask, voteMap] = buildConsensusMask(filePaths, intensityThreshold, consensusRatio)
%BUILDCONSENSUSMASK Generates a robust group mask requiring a minimum percentage 
% of subjects to exhibit active Gray Matter at each voxel.
%
% PURPOSE:
%   Replaces simple mean-thresholding to preserve voxels subject to high 
%   anatomical variance, ensuring minority features (e.g., shifted cortex 
%   due to atrophy) are retained in the VBM GLM pipeline.
%
% PARAMS:
%   filePaths          - (cell array of strings) Full paths to NIfTI files.
%   intensityThreshold - (double) Minimum intensity to consider a voxel as GM.
%   consensusRatio     - (double) Fraction of subjects [0, 1] that must have 
%                        active GM at a specific voxel to include it.
%
% RETURNS:
%   consensusMask      - (3D logical array) The final binary mask.
%   voteMap            - (3D double array) The raw spatial voting distribution.

    arguments
        filePaths (:, 1) cell {mustBeNonempty}
        intensityThreshold (1, 1) double
        consensusRatio (1, 1) double {mustBeGreaterThanOrEqual(consensusRatio, 0), ...
                                      mustBeLessThanOrEqual(consensusRatio, 1)} = 0.20
    end

    numSubjects = length(filePaths);
    minSubjectsRequired = ceil(consensusRatio * numSubjects);
    
    % Peek at the first file to extract dimensions dynamically
    info = niftiinfo(filePaths{1});
    voteMap = zeros(info.ImageSize);
    
    fprintf('Building Consensus Mask (Ratio: %.2f, Min Subjects: %d)...\n', ...
        consensusRatio, minSubjectsRequired);
    progressStep = max(1, floor(numSubjects / 20));
    fprintf('Progress: |');

    for i = 1:numSubjects
        if mod(i, progressStep) == 0, fprintf('='); end
        
        try
            currentVol = niftiread(filePaths{i});
            currentVol(isnan(currentVol)) = 0;
            
            % Binary accumulation: +1 vote if voxel is above threshold
            voteMap = voteMap + double(currentVol > intensityThreshold);
            
        catch ME
            fprintf('\n[ERROR] Failed during consensus loop on file: %s\n', filePaths{i});
            rethrow(ME);
        end
    end
    fprintf('|\n');
    
    % Finalize mask based on consensus rule
    consensusMask = voteMap >= minSubjectsRequired;
    
end



function [activeVoxels, totalVoxels, activePercentage] = analyzeVoxelVolume(volumeData, volumeName)
    % ANALYZEVOXELVOLUME Calculates and prints active voxel statistics.
    %
    % PURPOSE:
    % Analyzes an N-dimensional array (e.g., 3D neuroimaging NIfTI data) to 
    % determine the number and percentage of active voxels (values > 0) 
    % and prints the formatted result to the console.
    %
    % PARAMS:
    %   volumeData (numeric or logical array): The image data to analyze.
    %   volumeName (char or string): The identifier for the volume, used in console output.
    %
    % RETURN:
    %   activeVoxels (double): Total count of voxels strictly greater than 0.
    %   totalVoxels (double): Total number of elements in the array.
    %   activePercentage (double): Percentage of active voxels relative to total volume.
    %
    % RAISES:
    %   analyzeVoxelVolume:EmptyData: If the input volume data is empty.
    %   analyzeVoxelVolume:InvalidName: If the volumeName is not a character array or string.

    % 1. Input Validation
    if isempty(volumeData)
        error('analyzeVoxelVolume:EmptyData', 'The input volume data cannot be empty.');
    end
    
    if ~(ischar(volumeName) || isstring(volumeName))
        error('analyzeVoxelVolume:InvalidName', 'The volumeName must be a character array or string.');
    end

    % Ensure string is converted to char array for robust fprintf compatibility
    volumeNameStr = char(volumeName);

    % 2. Core Logic
    % Note: volumeData(:) flattens the N-D array to 1D for vectorization.
    % Comparing volumeData(:) > 0 covers both binary masks and intensity images safely.
    activeVoxels = sum(volumeData(:) > 0);
    totalVoxels = numel(volumeData);
    
    % Prevent division by zero mathematically, though isempty() already catches 0-element arrays
    activePercentage = (activeVoxels / totalVoxels) * 100;

    % 3. Console Output
    fprintf('%s has %d active voxels (%.2f%% of total volume).\n', volumeNameStr, activeVoxels, activePercentage);
end


function figHandle = plotMaskOverlays(baseImages, baseTitles, maskImages, maskColors, maskLabels)
    % PLOTMASKOVERLAYS Generates a dynamic grid of medical image overlays.
    %
    % PURPOSE:
    % Creates a dynamically sized figure with M columns (base images) and 
    % N+1 rows. The first row displays the base images unmodified. Each 
    % subsequent row displays the base images overlaid with a specific 
    % mask contour.
    %
    % PARAMS:
    %   baseImages  (cell array): 1D cell array of 2D numeric matrices (e.g. {mean, AD, CTRL}).
    %   baseTitles  (cell array): 1D cell array of strings/chars for column titles.
    %   maskImages  (cell array): 1D cell array of 2D logical/numeric matrices representing masks.
    %   maskColors  (cell array): 1D cell array of strings/chars for contour colors (e.g., {'r', 'g'}).
    %   maskLabels  (cell array): 1D cell array of strings/chars describing the masks for the titles.
    %
    % RETURN:
    %   figHandle   (Figure): Handle to the generated MATLAB figure object.
    %
    % RAISES:
    %   plotMaskOverlays:DimensionMismatch: If the number of images and titles do not match.
    %   plotMaskOverlays:MaskParameterMismatch: If masks, colors, and labels counts differ.
    
    % 1. Input Validation & Configuration
    numCols = numel(baseImages);
    numMasks = numel(maskImages);
    
    if numel(baseTitles) ~= numCols
        error('plotMaskOverlays:DimensionMismatch', 'Number of baseImages and baseTitles must match.');
    end
    
    if (numel(maskColors) ~= numMasks) || (numel(maskLabels) ~= numMasks)
        error('plotMaskOverlays:MaskParameterMismatch', 'Number of maskImages, maskColors, and maskLabels must match.');
    end
    
    % Constants to avoid magic numbers
    CONTOUR_LEVEL = [1 1];
    LINE_WIDTH = 1.5;
    COLOR_MAP_STYLE = 'gray';
    BACKGROUND_COLOR = 'w';
    
    numRows = numMasks + 1; % 1 row for base images + 1 row per mask
    
    % 2. Calculate Global Color Limits
    % Preallocate array to store the maximum values of each base image to uniform visual contrast
    maxVals = zeros(1, numCols);
    for i = 1:numCols
        currentImg = baseImages{i};
        maxVals(i) = max(currentImg(:));
    end
    globalColorLimits = [0, max(maxVals)];

    % 3. Figure Initialization
    % Dynamically scale figure height based on number of rows (e.g., ~150px per row)
    figHeight = max(300, 150 * numRows); 
    figHandle = figure('Name', 'Multi_Mask_Analysis', 'Color', BACKGROUND_COLOR, ...
                       'Position', [100, 100, 1200, figHeight]);
    
    % 4. Render Row 1: Base Images
    for colIdx = 1:numCols
        subplot(numRows, numCols, colIdx);
        imagesc(baseImages{colIdx});
        clim(globalColorLimits);
        axis image off;
        colormap(gca, COLOR_MAP_STYLE);
        title(baseTitles{colIdx}, 'Interpreter', 'none');
    end
    
    % 5. Render Rows 2 to N: Mask Overlays
    for maskIdx = 1:numMasks
        
        currentMask = maskImages{maskIdx};
        currentColor = maskColors{maskIdx};
        currentLabel = maskLabels{maskIdx};
        
        for colIdx = 1:numCols
            % Calculate correct 1D index for subplot layout
            plotIdx = (maskIdx * numCols) + colIdx;
            
            subplot(numRows, numCols, plotIdx);
            imagesc(baseImages{colIdx});
            clim(globalColorLimits);
            axis image off;
            colormap(gca, COLOR_MAP_STYLE);
            
            % Overlay Contour
            hold on;
            contour(currentMask, CONTOUR_LEVEL, currentColor, 'LineWidth', LINE_WIDTH);
            hold off;
            
            % Construct dynamic title resolving the original copy-paste error
            fullTitle = sprintf('%s (%s Contour: %s)', baseTitles{colIdx}, upper(currentColor), currentLabel);
            title(fullTitle, 'Interpreter', 'none');
        end
    end
end



function figHandle = plotMaskedHistograms(baseImages, baseTitles, maskImages, maskColors, maskLabels, bins, thresholdValue)
    % PLOTMASKEDHISTOGRAMSTRIANGULAR Plots 3 normalized intensity histograms in a triangular layout.
    %
    % PURPOSE:
    % Creates a single figure with a pyramid layout:
    %   - Top Left (AD expected)     -> Subplot mapped to baseImages{2}
    %   - Top Right (CTRL expected)  -> Subplot mapped to baseImages{3}
    %   - Bottom Center (Mean exp.)  -> Subplot mapped to baseImages{1}
    % All subplots maintain identical width and height to prevent visual bias.
    %
    % PARAMS:
    %   baseImages     (cell array): 1D cell array of EXACTLY 3 N-D numeric arrays.
    %                                Order must be: {BottomCenterImg, TopLeftImg, TopRightImg}.
    %   baseTitles     (cell array): 1D cell array of exactly 3 chars/strings for titles.
    %   maskImages     (cell array): 1D cell array of N-D logical arrays representing masks.
    %   maskColors     (cell array): 1D cell array of chars/strings specifying edge colors.
    %   maskLabels     (cell array): 1D cell array of chars/strings for the legend.
    %   bins           (numeric array): Edges or number of bins for the histogram.
    %   thresholdValue (double): The threshold value to display as a vertical reference line.
    %
    % RETURN:
    %   figHandle      (Figure): Handle to the generated figure.
    %
    % RAISES:
    %   plotMaskedHistogramsTriangular:DimensionMismatch: If baseImages does not contain exactly 3 elements.

    % 1. Input Validation
    if numel(baseImages) ~= 3 || numel(baseTitles) ~= 3
        error('plotMaskedHistogramsTriangular:DimensionMismatch', ...
              'This triangular layout requires exactly 3 base images and 3 titles.');
    end
    
    numMasks = numel(maskImages);
    if (numel(maskColors) ~= numMasks) || (numel(maskLabels) ~= numMasks)
        error('plotMaskedHistogramsTriangular:MaskMismatch', ...
              'Number of masks, colors, and labels must match.');
    end

    % Define constants
    BACKGROUND_COLOR = 'w';
    LINE_WIDTH = 2;
    PLOT_STYLE = 'stairs';
    NORM_TYPE = 'pdf';
    
    % Subplot mapping for a 2x4 grid to guarantee equal widths and perfect centering
    % index 1 (Mean) -> Bottom Center uses cells 6, 7
    % index 2 (AD)   -> Top Left uses cells 1, 2
    % index 3 (CTRL) -> Top Right uses cells 3, 4
    subplotMap = {[6, 7], [1, 2], [3, 4]};

    % 2. Figure Initialization
    figHandle = figure('Name', 'Triangular_Intensity_Histograms', ...
                       'Color', BACKGROUND_COLOR, 'Position', [100, 100, 1200, 800]);

    % 3. Subplot Generation Loop
    for imgIdx = 1:3
        % Apply the calculated fractional grid positioning
        subplot(2, 4, subplotMap{imgIdx});
        hold on;

        currentImg = baseImages{imgIdx};
        currentTitle = baseTitles{imgIdx};

        % Plot 1: Base Image Distribution (Global)
        histogram(currentImg, bins, 'Normalization', NORM_TYPE, ...
                  'DisplayName', sprintf('%s Global Dist.', currentTitle));

        % Plot 2 to N+1: Masked Distributions
        for maskIdx = 1:numMasks
            currentMask = maskImages{maskIdx};
            currentColor = maskColors{maskIdx};
            currentLabel = maskLabels{maskIdx};

            % Extract masked voxels
            maskedData = currentImg(currentMask);

            histogram(maskedData, bins, 'Normalization', NORM_TYPE, ...
                      'DisplayStyle', PLOT_STYLE, 'EdgeColor', currentColor, ...
                      'LineWidth', LINE_WIDTH, ...
                      'DisplayName', sprintf('%s Dist.', currentLabel));
        end

        % Plot Threshold Reference Line
        xline(thresholdValue, 'r--', 'LineWidth', LINE_WIDTH, ...
              'DisplayName', sprintf('Threshold = %.3f', thresholdValue));

        % Subplot Formatting
        xlabel('Voxel Intensity');
        ylabel('Probability Density');
        title(sprintf('%s Intensity', currentTitle), 'Interpreter', 'none');
        legend('Location', 'best', 'Interpreter', 'none');
        grid on;
        yscale('log');
        
        hold off;
    end
end

function [optimalMask, optimalThreshold, maxCorrelation] = computeObjectiveMask(meanVolume, stepSize)
    % COMPUTEOBJECTIVEMASK Determines the optimal binary mask based on Ridgway (2009).
    %
    % PURPOSE:
    % Iteratively calculates the Pearson correlation coefficient between the 
    % continuous mean image and candidate binary masks thresholded at various levels.
    % Returns the mask that maximizes this correlation (Objective Masking strategy).
    %
    % PARAMS:
    %   meanVolume (numeric array): The N-dimensional continuous mean image.
    %   stepSize (double): The step size for the threshold iteration (e.g., 0.01).
    %                      Defaults to 0.01 if not provided.
    %
    % RETURN:
    %   optimalMask (logical array): The binary mask generated using the optimal threshold.
    %   optimalThreshold (double): The threshold value (T*) that maximized correlation.
    %   maxCorrelation (double): The maximum Pearson correlation coefficient achieved.
    %
    % RAISES:
    %   computeObjectiveMask:EmptyVolume: If the input volume is empty.
    %   computeObjectiveMask:ConstantVolume: If the input volume has zero variance.

    % 1. Input Validation & Defaults
    if nargin < 2 || isempty(stepSize)
        stepSize = 0.01;
    end
    
    if isempty(meanVolume)
        error('computeObjectiveMask:EmptyVolume', 'The input meanVolume cannot be empty.');
    end
    
    % 2. Data Preparation
    % Flatten the multi-dimensional array into a 1D column vector for corrcoef
    flatVolume = meanVolume(:);
    
    if isempty(flatVolume) || var(flatVolume) == 0
        error('computeObjectiveMask:ConstantVolume', 'The input volume lacks variance or active voxels.');
    end

    % 3. Determine Dynamic Threshold Range
    % Normalize the search space relative to the actual maximum intensity of the mean image
    maxIntensity = max(flatVolume);
    thresholdRange = stepSize : stepSize : (maxIntensity - stepSize);
    
    % Initialize tracking variables
    maxCorrelation = -1.0;
    optimalThreshold = stepSize;
    
    % 4. Iterative Optimization (Core Logic)
    fprintf('Computing Objective Mask (Ridgway criterion)...\n');
    fprintf('Searching %d thresholds: |', numel(thresholdRange));
    
    for i = 1:numel(thresholdRange)
        if mod(i, max(1, floor(numel(thresholdRange)/20))) == 0
            fprintf('='); % Progress indicator
        end
        
        currentThreshold = thresholdRange(i);
        
        % Generate candidate binary vector
        candidateBinary = double(flatVolume > currentThreshold);
        
        % Ensure the binary vector is not completely uniform (all 0s or all 1s)
        if var(candidateBinary) > 0
            % Compute Pearson correlation matrix; index (1,2) is the cross-correlation
            corrMat = corrcoef(flatVolume, candidateBinary);
            currentCorr = corrMat(1, 2);
            
            % Update optimal values if a new maximum is found
            if currentCorr > maxCorrelation
                maxCorrelation = currentCorr;
                optimalThreshold = currentThreshold;
            end
        end
    end
    fprintf(']\n');
    
    % 5. Final Output Generation
    % Apply the optimal threshold to the original N-dimensional volume
    optimalMask = meanVolume > optimalThreshold;
    
    fprintf('Optimal Threshold (T*) found: %.4f (Pearson r = %.4f)\n', optimalThreshold, maxCorrelation);
end

function [tpmMask, tpmVol] = buildTpmMask(tpmPath, referenceVolPath, intensityThreshold)
%BUILDTPMMASK Generates a Universal Binary Mask by reading the SPM Tissue Probability Map (TPM).
%
% PURPOSE:
%   Directly loads the SPM TPM (Gray Matter channel) and applies a threshold.
%   Optimized for pipelines where the DARTEL spatial grid perfectly matches 
%   the TPM grid (e.g., 121x145x121 at 1.5mm isotropic), skipping interpolation.
%
% PARAMS:
%   tpmPath            - (string/char) Full path to the TPM.nii file.
%   referenceVolPath   - (string/char) Full path to a single reference DARTEL NIfTI file.
%   intensityThreshold - (double) Minimum probability [0, 1] to consider a voxel as GM.
%
% RETURNS:
%   tpmMask            - (3D logical array) The final binary mask.
%   tpmVol             - (3D double array) The raw read TPM probabilities.
%
% RAISES:
%   buildTpmMask:FileNotFound - If either the TPM or reference file is missing.
%   buildTpmMask:SpatialMismatch - If the dimensions or affines do not match.

    arguments
        tpmPath (1, :) char {mustBeNonempty}
        referenceVolPath (1, :) char {mustBeNonempty}
        intensityThreshold (1, 1) double {mustBeGreaterThanOrEqual(intensityThreshold, 0), ...
                                          mustBeLessThanOrEqual(intensityThreshold, 1)} = 0.01
    end
    
    % 1. Validation & System Checks
    if exist(referenceVolPath, 'file') ~= 2
        error('buildTpmMask:FileNotFound', 'Reference volume not found: %s', referenceVolPath);
    end
    
    % Ensure we explicitly point to Volume 1 (Gray Matter) of the 4D TPM
    if ~contains(tpmPath, ',')
        tpmPathGM = [tpmPath, ',1'];
    else
        tpmPathGM = tpmPath;
    end

    % 2. Header Extraction
    fprintf('Loading headers to verify spatial grid identity...\n');
    try
        headerRef = spm_vol(referenceVolPath);
        headerTpm = spm_vol(tpmPathGM);
    catch ME
        fprintf('[ERROR] Failed to read NIfTI headers using spm_vol.\n');
        rethrow(ME);
    end

    % 3. Safety Check: Verify identical spatial footprints
    if ~isequal(headerTpm.dim, headerRef.dim)
        error('buildTpmMask:DimensionMismatch', ...
            'Grid size mismatch: TPM is [%d %d %d], Reference is [%d %d %d].', ...
            headerTpm.dim, headerRef.dim);
    end
    
    % Allow a tiny tolerance for floating point inaccuracies in affine matrices
    tolerance = 1e-4;
    if max(abs(headerTpm.mat(:) - headerRef.mat(:))) > tolerance
        warning('buildTpmMask:AffineMismatch', ...
            'Affine matrices differ slightly. Make sure the origins are perfectly aligned!');
    end

    % 4. Direct Volume Read (No Interpolation needed)
    fprintf('Direct spatial match confirmed. Reading TPM volume...\n');
    try
        tpmVol = spm_read_vols(headerTpm);
    catch ME
        fprintf('[ERROR] Failed during spm_read_vols.\n');
        rethrow(ME);
    end

    % 5. Matrix Cleanup and Thresholding
    tpmVol(isnan(tpmVol)) = 0;
    
    % Generate the binary mask
    tpmMask = tpmVol > intensityThreshold;
    
    fprintf('TPM Mask successfully generated. Active threshold: > %.3f\n', intensityThreshold);
end