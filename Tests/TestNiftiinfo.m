%% Tests/TestNiftiinfo.m
% -------------------------------------------------------------------------
% This script processes NIfTI files for Alzheimer's Disease (AD) and 
% Control (CTRL) subjects. It calculates the mean Gray Matter (GM) 
% volume across all subjects, both with and without Total Intracranial 
% Volume (TIV) normalization. It creates binary masks based on an 
% intensity threshold and generates visualizations for quality control 
% (slice overlays and histograms).
% -------------------------------------------------------------------------

% Setup Paths and Directories
scriptPath = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptPath);

% Define directories for Alzheimer's Disease (AD) and Control (CTRL) groups
dirAD   = fullfile(projectRoot, 'AD_CTRL', 'AD_s3');
dirCTRL = fullfile(projectRoot, 'AD_CTRL', 'CTRL_s3');
TIVpath = fullfile(projectRoot, 'AD_CTRL', 'covariateADCTRLsexAgeTIV.csv');

% -------------------------------------------------------------------------

% Load TIV Data
tiv_table = readtable(TIVpath);
TIV = tiv_table.TIV;
TIV_max = max(TIV);

% Normalize TIV relative to the maximum TIV in the dataset
TIV_norm = TIV ./ TIV_max;

% -------------------------------------------------------------------------

% Define file paths for the AD e CTRL files
files_AD = dir(fullfile(dirAD, 'smwc1AD-*.nii'));
files_CTRL = dir(fullfile(dirCTRL, 'smwc1CTRL-*.nii'));

% Combine file lists into a single structure
all_files_struct = [files_AD; files_CTRL];
GM_files = arrayfun(@(s) fullfile(s.folder, s.name), all_files_struct, 'UniformOutput', false);
N = numel(GM_files);


% Check if files exist
if N == 0
    error('No NIfTI files found in %s or %s', dirAD, dirCTRL);
end
fprintf('Found %d subjects. Starting mean calculation...\n', N);

% -------------------------------------------------------------------------

% Read the first NIfTI file and extract image dimensions
info = niftiinfo(GM_files{1});

% Initialize accumulators for raw and normalized volumes
sum_vol = zeros(info.ImageSize);
sum_vol_norm = zeros(info.ImageSize);


fprintf('Progress: |');
step = floor(N/20); % Visual progress bar step

for i = 1:N
    if mod(i, step) == 0, fprintf('='); end
    
    % Read volume data
    vol = niftiread(GM_files{i});

    % Replace NaNs with 0
    vol(isnan(vol)) = 0;
    
    % Apply TIV normalization for the specific subject
    vol_norm = vol ./ TIV_norm(i);
    
    % Accumulate volumes
    sum_vol_norm = sum_vol_norm + vol_norm;
    sum_vol = sum_vol + vol;
end
fprintf('| \n');

% Calculate the mean
mean_GM = sum_vol / N;
mean_GM_norm = sum_vol_norm / N;

% -------------------------------------------------------------------------

thr = 0.01; % Intensity threshold

% Create binary masks
mask = mean_GM > thr;
mask_norm = mean_GM_norm > thr;

% --- Statistics for Raw Mean ---
M_mean = sum(mean_GM(:)>0);
total_voxels_mean = numel(mean_GM);
active_percentage_mean = (M_mean / total_voxels_mean) * 100;
fprintf('mean_GM has %d active voxels (%.2f%% of total volume).\n', M_mean, active_percentage_mean);

M_mask = sum(mask(:));
total_voxels_mask = numel(mask);
active_percentage_mask = (M_mask / total_voxels_mask) * 100;
fprintf('mask has %d active voxels (%.2f%% of total volume).\n', M_mask, active_percentage_mask);

% --- Statistics for Normalized Mean ---
M_mean_norm = sum(mean_GM_norm(:)>0);
total_voxels_mean_norm = numel(mean_GM_norm);
active_percentage_mean_norm = (M_mean_norm / total_voxels_mean_norm) * 100;
fprintf('mean_GM_norm has %d active voxels (%.2f%% of total volume).\n', M_mean_norm, active_percentage_mean_norm);

M_mask_norm = sum(mask_norm(:));
total_voxels_mask_norm = numel(mask_norm);
active_percentage_mask_norm = (M_mask_norm / total_voxels_mask_norm) * 100;
fprintf('mask_norm has %d active voxels (%.2f%% of total volume).\n', M_mask_norm, active_percentage_mask_norm);

% -------------------------------------------------------------------------

% --- Raw Data Visualization ---
mid_slice_idx = round(size(mean_GM, 3) / 2);
slice_mean = mean_GM(:, :, mid_slice_idx);
slice_mask = mask(:, :, mid_slice_idx);

fig1 = figure('Name', 'Mean_Mask_Analisys', 'Color', 'w', 'Position', [100, 100, 1200, 500]);
subplot(1, 3, 1);
imagesc(slice_mean);
axis image off;
colormap(gca, 'gray');
title('Mean GM Map');

subplot(1, 3, 2);
imagesc(slice_mask);
axis image off;
colormap(gca, 'gray');
title(['Mask (Threshold = ' num2str(thr, '%.3f') ')']);

subplot(1, 3, 3);
imagesc(slice_mean); 
axis image off; 
colormap(gca, 'gray');
hold on;
contour(slice_mask, [1 1], 'r', 'LineWidth', 1.5);
title('Overlay (Red = Mask)');
hold off;


% --- Normalized Data Visualization ---
mid_slice_idx_norm = round(size(mean_GM_norm, 3) / 2);
slice_mean_norm = mean_GM_norm(:, :, mid_slice_idx_norm);
slice_mask_norm = mask_norm(:, :, mid_slice_idx_norm);

fig2 = figure('Name', 'Mean_Mask_Analisys_Normalized', 'Color', 'w', 'Position', [100, 100, 1200, 500]);
subplot(1, 3, 1);
imagesc(slice_mean_norm);
axis image off;
colormap(gca, 'gray');
title('Mean GM Map (Normalized)');

subplot(1, 3, 2);
imagesc(slice_mask_norm);
axis image off;
colormap(gca, 'gray');
title(['Mask (Normalized) (Threshold = ' num2str(thr, '%.3f') ')']);

subplot(1, 3, 3);
imagesc(slice_mean_norm); 
axis image off; 
colormap(gca, 'gray');
hold on;
contour(slice_mask_norm, [1 1], 'r', 'LineWidth', 1.5);
title('Overlay (Red = Mask)');
hold off;

% -------------------------------------------------------------------------

% --- Raw Histogram ---
bins = 0 : 0.01 : max(mean_GM(:));

fig3 = figure('Name', 'Mean_Intensity_Histogram', 'Color', 'w');
hold on;
    histogram(mean_GM, bins, 'Normalization', 'pdf', 'DisplayName', 'Mean Voxel Distribution');
    histogram(mean_GM(mask), bins, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'EdgeColor', 'b', 'LineWidth', 2, 'DisplayName', 'Masked Voxel Distribution');
    
    xline(thr, 'r--', 'LineWidth', 2, 'DisplayName', ['Threshold = ' num2str(thr, '%.3f')]);
    
    xlabel('Mean Intensity');
    ylabel('Probability Density');
    title('Mean GM Intensity Distribution');
    legend;
    grid on;
    yscale('log');
hold off;

% --- Normalized Histogram ---
bins_norm = 0 : 0.01 : max(mean_GM_norm(:));

fig4 = figure('Name', 'Mean_Intensity_Histogram_Normalized', 'Color', 'w');
hold on;
    histogram(mean_GM_norm, bins_norm, 'Normalization', 'pdf', 'DisplayName', 'Mean Voxel Distribution');
    histogram(mean_GM_norm(mask_norm), bins_norm, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'EdgeColor', 'b', 'LineWidth', 2, 'DisplayName', 'Masked Voxel Distribution');
    
    xline(thr, 'r--', 'LineWidth', 2, 'DisplayName', ['Threshold = ' num2str(thr, '%.3f')]);
    
    xlabel('Mean Intensity');
    ylabel('Probability Density');
    title('Mean GM Intensity Distribution (Normalized)');
    legend;
    grid on;
    yscale('log');
hold off;


