function Preliminaries(dirAD, dirCTRL, TIVpath, varargin)
% PRELIMINARIES Preprocessing for NIfTI data (AD vs CTRL) with SPM or IPT.
%   Loads NIfTI images for AD and CTRL groups, builds a 3D brain mask,
%   vectorizes the data, applies TIV normalization, and saves results to a
%   .mat file. Works with SPM (SPM25 tested) or, as a fallback,
%   with the Image Processing Toolbox (IPT). Can be run in standalone MATLAB
%   or called from Python via the MATLAB Engine.
%
%   Required Inputs:
%       dirAD      - path to the AD group NIfTI folder
%       dirCTRL    - path to the CTRL group NIfTI folder
%       TIVpath    - path to the .csv file with TIV values
%
%   Optional Name-Value Pair Arguments:
%
%       'outputPath' : Full path for the output .mat file.
%                      Default: '[this_function_folder]/preliminaries_output.mat'
%
%
%       'loggerHandle'    : pre-initialized MATLAB Logger instance
%                           (typically created in RunStandalone.m).
%                           If present and 'pythonLogHandle' is not used,
%                           Preliminaries reuses this logger.
%
%   Logging Priority:
%
%       1. If 'loggerHandle' is provided -> reuse caller's MATLAB logger
%       2. Else -> create a local MATLAB Logger
%
%   When called from Python via MATLAB Engine, console output is captured by
%   the Engine and forwarded to the Python CustomLogger (near realtime).
%
%   Syntax and Usage Examples:
%
%   1. Running in Standalone Mode (MATLAB only):
%       Preliminaries('path/AD', 'path/CTRL', 'path/tiv.csv', 'outputPath', 'processed_data.mat');
%
%   2. Running in Standalone Mode with a shared logger:
%       logger = Logger('Preliminaries'); logger.addConsoleHandler('use_colors', true);
%       Preliminaries('path/AD', 'path/CTRL', 'path/tiv.csv', ...
%                     'loggerHandle', logger);
%

    %% --- Initialize inputParser ---
    p = inputParser;

    % Define validation functions for paths
    validateDir = @(x) isstring(x) || (ischar(x) && exist(x, 'dir'));
    validateFile = @(x) isstring(x) || (ischar(x) && exist(x, 'file'));
   
    % Define the required arguments
    addRequired(p, 'dirAD', validateDir);
    addRequired(p, 'dirCTRL', validateDir);
    addRequired(p, 'TIVpath', validateFile);

    % Determine the default output path based on this function's location
    functionPath = fileparts(mfilename('fullpath'));
    defaultOutputPath = fullfile(functionPath, 'preliminaries_output.mat');

    % Define the optional name-value parameters
    addParameter(p, 'outputPath', defaultOutputPath, @ischar);
    addParameter(p, 'loggerHandle', [], @(x) ~isempty(x));


    % --- Parse the provided inputs ---
    try
        parse(p, dirAD, dirCTRL, TIVpath, varargin{:});
    catch ME
        error('Input validation failed! Reason: [%s] %s', ME.identifier, ME.message);
    end
    
    %% --- Logger Initialization ---
    if ~isempty(p.Results.loggerHandle)
        % Standalone MATLAB mode with logger provided by the caller
        logger = p.Results.loggerHandle;
        logger.info('MATLAB standalone mode detected. Using loggerHandle passed from caller.');
    else
        % Standalone MATLAB mode
        wantColors = usejava('desktop') && ~isempty(which('cprintf'));
        logger = Logger('Preliminaries');
        logger.addConsoleHandler('use_colors', wantColors);
        if wantColors
            logger.info('MATLAB console logger with colors enabled (Desktop).');
        else
            logger.info('MATLAB console logger (plain, Engine-compatible).');
        end
        logger.info('MATLAB standalone mode detected. Native logger enabled.');
    end

    %% --- Attempt to start a parallel pool ---
    logger.info('Checking for parallel computing availability...');
    [pool_handle, toolbox_available] = safe_parpool(logger);

    % Parallel Computing Toolbox must be installed AND a pool must be active
    use_parfor = toolbox_available && ~isempty(pool_handle);

    logger.info('===== PIPELINE SECTION 1: START =====');
    %% --- 1.1: Reading NIfTI files and dimension check ---
    logger.info('STEP 1.1: Reading NIfTI files and dimension check...');

    % Use the paths passed as arguments
    dirname_AD = p.Results.dirAD;
    dirname_CTRL = p.Results.dirCTRL;


    %% --- Automatically detect SPM/IPT and decide which functions to use ---
    use_spm = ~isempty(which('spm'));
    IPT_funcs = ~isempty(which('niftiread')) && ~isempty(which('niftiinfo'));
    
    if use_spm
        logger.info('SPM (%s) detected on path. Will use SPM functions for NIfTI operations.', spm('Ver'));
    else
        try
            % License name for Image Processing Toolbox
            IPT_license = license('test', 'Image_Toolbox');
        catch
            IPT_license = false;
        end
        if ~(IPT_license && IPT_funcs)
            logger.critical('Image Processing Toolbox required when SPM is not available. Aborting.');
            error('Required dependency missing: SPM and Image Processing Toolbox are not installed or not in the MATLAB path.');
        else
            logger.info('SPM not found. Image Processing Toolbox available: proceeding with niftiread/niftiinfo.');
        end
    end


    %% --- Load GM files based on SPM availability ---
    if use_spm
        logger.info('Using SPM (%s) to select NIfTI files...', spm('Ver'));
        try
            % spm_select('FPList', ...) returns the full path list of files
            GM_files_AD = spm_select('FPList', dirname_AD, '^smwc1AD-.*\.nii$');
            GM_files_CTRL = spm_select('FPList', dirname_CTRL, '^smwc1CTRL-.*\.nii$');

            % Vertically concatenates the two lists of paths as arrays
            GM_files = [cellstr(GM_files_AD); cellstr(GM_files_CTRL)];

            % Calculate the number of subjects for each group and in total
            N = size(GM_files, 1);
            N_AD = size(GM_files_AD, 1);
            N_CTRL = size(GM_files_CTRL, 1);

        catch ME
            logger.critical('Failed during file selection with SPM (%s)! Reason: [%s] %s',  spm('Ver'), ME.identifier, ME.message);
            error('SPM file selection failed. Check SPM configuration and file paths.');
        end

    else
        logger.info('Using standard MATLAB functions (Image Processing Toolbox) to find NIfTI files...');

    
        % The 'dir' function searches for files matching the pattern
        % and returns an array of structures
        files_AD_struct = dir(fullfile(dirname_AD, 'smwc1AD-*.nii'));
        files_CTRL_struct = dir(fullfile(dirname_CTRL, 'smwc1CTRL-*.nii'));

        % Concatenate the two struct arrays
        all_files_struct = [files_AD_struct; files_CTRL_struct];
    
        % Every struct array has separate 'name' and 'folder' fields 
        % We use 'arrayfun' to recombine them into a list of full paths
        GM_files = arrayfun(@(s) fullfile(s.folder, s.name), all_files_struct, 'UniformOutput', false);
    
        % Calculate the number of subjects from the number of files found
        N_AD = numel(files_AD_struct);
        N_CTRL = numel(files_CTRL_struct);
        N = numel(GM_files);
    end

    % Check if any files were found
    if N == 0
        logger.error('No NIfTI files found in the specified directories.');
        error('Execution aborted: No NIfTI files found in the specified directories.');
    end
    logger.success('%d GM map files found (%d AD, %d CTRL)!', N, N_AD, N_CTRL);

    % Load the first image only to get the volume dimensions
    logger.info('Loading the first image to get volume dimensions...');
    try
        if use_spm
            GM0 = spm_read_vols(spm_vol(GM_files{1}));
        else
            GM0 = niftiread(GM_files{1});
        end
        logger.info('Volume dimensions: %s', mat2str(size(GM0)));
    catch ME
        logger.critical('Failed to read the first NIfTI file! Reason: [%s] %s', ME.identifier, ME.message);
        error('Could not read initial NIfTI file. Check file integrity.');
    end

    %% --- Dimension check ---
    logger.info('Verifying that all NIfTI volumes have the same dimensions...');
    all_dims_match = true;
    all_dims = cell(N, 1); % Pre-allocate a cell array for dimensions
    try
        if use_parfor
           
            % Use a parallel loop to read the header/info of each file
            parfor i = 1:N
                if use_spm
                    % SPM Branch: Read the volume header
                    vol_header = spm_vol(GM_files{i});
                    all_dims{i} = vol_header.dim;
                else
                    % Non-SPM Branch: Read the NIfTI info
                    info = niftiinfo(GM_files{i});
                    all_dims{i} = info.ImageSize;
                end
            end
        else
           
            % Use a serial loop (fallback)
            for i = 1:N
                if use_spm
                    % SPM Branch: Read the volume header
                    vol_header = spm_vol(GM_files{i});
                    all_dims{i} = vol_header.dim;
                else
                    % Non-SPM Branch: Read the NIfTI info
                    info = niftiinfo(GM_files{i});
                    all_dims{i} = info.ImageSize;
                end
            end
        end


        % After collecting all dimensions in parallel, check for mismatches serially
        for i = 2:N % Start from the second image
            if ~isequal(size(GM0), all_dims{i})
                logger.critical('Dimension mismatch found! File %s has dimensions [%s], expected [%s].', ...
                                GM_files{i}, mat2str(all_dims{i}), mat2str(size(GM0)));
                all_dims_match = false;
                break; % Exit the loop on the first mismatch found
            end
        end

    catch ME
        logger.critical('An error occurred during dimension check! Reason: [%s] %s', ME.identifier, ME.message);
        all_dims_match = false;
    end

    % Abort if any dimension mismatch was detected
    if ~all_dims_match
        logger.critical('Aborting execution due to dimension mismatch.');
        error('Dimension mismatch detected. All NIfTI volumes must have the same dimensions.');
    end
    logger.success('All %d volumes have consistent dimensions: %s', N, mat2str(size(GM0)));


    %% --- 1.2: Creating 3D mask and vectorizing raw data ---
    logger.info('STEP 1.2: Creating 3D mask and vectorizing raw data...');
    % --- Creating the 3D mask ---
    logger.info('Creating the 3D mask...');
    % Pre-allocate a 4D logical array to store voxel masks for each subject
    mask_temp = false([size(GM0), N]);
    % Loop through all files to build the mask
    if use_parfor
       
        parfor i = 1:N
            if use_spm
                current_image = spm_read_vols(spm_vol(GM_files{i}));
            else
                current_image = niftiread(GM_files{i});
            end
            % Create a binary mask for the current volume and store it in the 4th dimension of mask_temp
            mask_temp(:,:,:,i) = current_image > 0;
        end
    else
        
        for i = 1:N
            if use_spm
                current_image = spm_read_vols(spm_vol(GM_files{i}));
            else
                current_image = niftiread(GM_files{i});
            end
            % Create a binary mask for the current volume and store it in the 4th dimension of mask_temp
            mask_temp(:,:,:,i) = current_image > 0;
        end
    end

    % Collapse the 4th dimension
    mask = any(mask_temp, 4);
    logger.success('Created the 3D mask!')


    % Calculate the number and percentage of active voxels in the mask
    voxelIdx = find(mask);
    M = numel(voxelIdx);
    total_voxels = numel(GM0);
    active_percentage = (M / total_voxels) * 100;
    logger.success('3D mask created with %d active voxels (%.2f%% of total volume).', M, active_percentage);

    %% --- Vectorization of raw data ---
    logger.info('Vectorizing and populating X_raw...');
    % Pre-allocate the data matrix X_raw with zeros
    X_raw = zeros(N, M);
    if use_parfor
        
        % Read the volumes
        parfor i = 1:N
            if use_spm
                current_image = spm_read_vols(spm_vol(GM_files{i}));
            else
                current_image = niftiread(GM_files{i});
            end
            % Vectorize subject by subject: extract the M active voxels from the
            % current image using the mask and place them in the i-th row of X_raw
            X_raw(i, :) = current_image(mask)';
        end
    else 
        
        for i = 1:N
            if use_spm
                current_image = spm_read_vols(spm_vol(GM_files{i}));
            else
                current_image = niftiread(GM_files{i});
            end
            % Vectorize subject by subject: extract the M active voxels from the
            % current image using the mask and place them in the i-th row of X_raw
            X_raw(i, :) = current_image(mask)';
        end
    end
    logger.success('Created the X_raw matrix!');


    %% --- 1.3: TIV Normalization ---
    logger.info('STEP 1.3: Applying multiplicative TIV normalization...');
    try
        % Load Total Intracranial Volume (TIV) data
        tiv_path = p.Results.TIVpath;
        TIV = readtable(tiv_path).TIV;
        % Check if the number of TIV entries matches the number of subjects
        if numel(TIV) ~= N
            logger.critical('TIV count (%d) does not match subject count (%d).', numel(TIV), N);
            error('Mismatch between number of subjects and TIV entries.');
        end
        % Normalize TIV values by dividing each value by the maximum TIV.
        TIV_norm = TIV / max(TIV);
        % Element-wise multiplication with the data matrix.
        X_raw = X_raw ./ TIV_norm;
        logger.success('TIV normalization applied!');
    catch ME
        logger.critical('Failed during TIV application! Reason: [%s] %s', ME.identifier, ME.message);
        error('An error occurred during TIV normalization.');
    end
    
    %% --- 1.4: Creating Labels and Saving Data ---
    logger.info('STEP 1.4: Creating labels and saving processed data...')
    % ---Creating labels ---
    logger.info('Creating labels...')
    % Create labels (1 = AD, 0 = CTRL) based on file order
    y_all = [ones(N_AD, 1); zeros(N_CTRL, 1)];
    logger.success('Labels created successfully!');
    
    try
        logger.info('Saving processed data to %s...', p.Results.outputPath);
        % Save the variables to the specified path (or to the default one)
        save(p.Results.outputPath, 'X_raw', 'y_all', 'mask', 'voxelIdx', 'M', 'N', '-v7.3');
        logger.success('File saved successfully!');
    catch ME
        logger.critical('Failed during save! Reason: [%s] %s', ME.identifier, ME.message);
        return;
    end
    logger.info('===== PIPELINE SECTION 1: COMPLETE =====');

end


%% --- Helper Functions ---

function [pool, toolbox_available] = safe_parpool(logger)
%SAFE_PARPOOL Start a parallel pool if possible; otherwise run in serial.
% Returns:
%   pool               - pool handle (empty if running in serial)
%   toolbox_available  - true if PCT is installed/licensed and core funcs exist

    pool = [];                 % default: serial execution
    toolbox_available = false; % set true only if PCT is usable

    % License check 
    try
        PCT_license = license('test', 'Distrib_Computing_Toolbox');
    catch
        PCT_license = false;
    end

    % Core functions availability
    PCT_func = ~isempty(which('parpool')) || ~isempty(which('gcp'));

    if ~(PCT_license && PCT_func)
        logger.warn('Parallel Computing Toolbox not found or PCT license may be unavailable. Continuing in serial mode.');
        return;
    end

    % PCT is in principle usable
    toolbox_available = true;

    % Reuse an existing pool if present
    try
        pool = gcp('nocreate');
        if ~isempty(pool)
            logger.success('Parallel pool already active with %d workers.', pool.NumWorkers);
            return;
        end
    catch ME
        % Non-fatal: proceed and attempt to start a pool
        logger.warn('Error while checking existing parallel pool (gcp). Continuing. Reason: [%s] %s', ME.identifier, ME.message);
    end

    % Choose a cluster profile to use
    profileToUse = '';
    try
        defProf = parallel.defaultClusterProfile; 
        % If a default profile exists, use it
        if ~isempty(defProf)
            profileToUse = defProf;
        else
            % No default configured: check if "local" exists
            profs = parallel.clusterProfiles();
            if any(strcmpi(profs, 'local'))
                profileToUse = 'local';
            end
        end
    catch ME
        % If cluster profiles are not available or fail, try "local"
        logger.warn('Could not query cluster profiles. Attempting "local". Reason: [%s] %s', ME.identifier, ME.message);
        profileToUse = 'local';
    end

    if isempty(profileToUse)
        % PCT is available, but no profile can be used. Remain in serial
        logger.warn('Parallel Computing Toolbox detected, but no usable cluster profile found. Continuing in serial mode.');
        pool = [];
        return;
    end

    % Start the pool 
    try
        pool = parpool(profileToUse);
        logger.success('Parallel pool started on profile "%s" with %d workers.', profileToUse, pool.NumWorkers);
    catch ME
        logger.warn('Failed to start parallel pool on profile "%s". Continuing in serial mode. Reason: [%s] %s', ...
                    profileToUse, ME.identifier, ME.message);
        pool = [];
        % toolbox_available remains true: PCT exists, but the pool did not start
    end
end





  


