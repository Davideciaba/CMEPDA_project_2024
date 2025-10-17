function Preliminaries(dirAD, dirCTRL, TIVpath, varargin)
% PRELIMINARIES Performs preprocessing for NIfTI data analysis.
%   This function loads NIfTI images for AD and CTRL groups, creates a 3D
%   brain mask, vectorizes the data, applies TIV normalization, and saves
%   the results to a .mat file.
%   The function is capable of operating with or without SPM and the Parallel Computing Toolbox.
%
%   Required Inputs:
%       dirAD      - Path to the directory containing AD group NIfTI files.
%       dirCTRL    - Path to the directory containing CTRL group NIfTI files.
%       TIVpath    - Path to the .csv file containing TIV data.
%
%   Optional Name-Value Pair Arguments:
%       'outputPath' - Full path for the output .mat file.
%                      Default: '[function_directory]/preliminaries_output.mat'
%
%       'pythonLogHandle' - Handle to a Python logging function.
%                           If this parameter is provided, the function enters 
%                           "proxy mode": logs are forwarded to the Python logger 
%                           for centralized handling, in addition to being printed 
%                           to the MATLAB Command Window (if available).
%                           Default: [] (uses native MATLAB logger Logger.m)
%
%   Syntax and Usage Examples:
%
%   1. Running in Standalone Mode (MATLAB only):
%      Preliminaries('path/to/AD', 'path/to/CTRL', 'path/to/tiv.csv', 'outputPath', 'processed_data.mat');
%
%   2. Running from Python (syntax is for the Python engine):
%      eng.Preliminaries('path/AD', 'path/CTRL', 'path/tiv.csv', 'pythonLogHandle', py_logger.log, nargout=0)
%

    % --- Initialize inputParser ---
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
    addParameter(p, 'pythonLogHandle', [], @(x) isa(x, 'function_handle'));

    % --- Parse the provided inputs ---
    try
        parse(p, dirAD, dirCTRL, TIVpath, varargin{:});
    catch ME
        error('Input validation failed! Reason: [%s] %s', ME.identifier, ME.message);
    end
    
    % --- Logger Initialization ---
    if ~isempty(p.Results.pythonLogHandle)
        % Python-driven mode
        logger = PythonLoggerProxy(p.Results.pythonLogHandle);
        logger.info('Python-driven mode detected. Proxy logger enabled.');
    else
        % Standalone MATLAB mode
        logger = Logger('Preliminaries');
        logger.addConsoleHandler('use_colors', true); % Stampa solo sulla Command Window
        logger.info('MATLAB standalone mode detected. Native logger enabled.');
    end

    % --- Attempt to start a parallel pool ---
    logger.info('Checking for parallel computing availability...');
    safe_parpool(logger);

    logger.info('===== PIPELINE SECTION 1: START =====');
    % --- 1.1: Reading NIfTI files and dimension check ---
    logger.info('Step 1.1: Reading NIfTI files and dimension check...');

    % Use the paths passed as arguments
    dirname_AD = p.Results.dirAD;
    dirname_CTRL = p.Results.dirCTRL;


    % --- Automatically detect SPM and decide which functions to use ---
    use_spm = (exist('spm_vol', 'file') == 2);
    if use_spm
        logger.info('SPM (%s) found and will be used for NIfTI operations.', spm('Ver'));
    else
        logger.warn('SPM not found. Falling back to standard MATLAB NIfTI functions.');
    end


    % --- Load GM files based on SPM availability ---
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
        logger.info('Using standard MATLAB functions to find NIfTI files...');
    
        % Check to immediately stop the script if the Image Processing
        % Toolbox is missing
        if isempty(which('niftiread'))
            logger.critical('Function "niftiread" (Image Processing Toolbox) is required when not using SPM. Aborting execution.');
            error('Required dependency missing: Image Processing Toolbox is not installed or not in the MATLAB path.');
        end
    
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

    % --- Dimension check ---
    logger.info('Verifying that all NIfTI volumes have the same dimensions...');
    all_dims_match = true;
    all_dims = cell(N, 1); % Pre-allocate a cell array for dimensions
    try
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


    % --- 1.2: Creating 3D mask and vectorizing raw data ---
    logger.info('Step 1.2: Creating 3D mask and vectorizing raw data...');
    % --- Creating the 3D mask ---
    logger.info('Creating the 3D mask...');
    % Initialize an empty logical mask with the correct dimensions
    mask = false(size(GM0));
    % Loop through all files to build the mask
    parfor i = 1:N
        if use_spm
            current_image = spm_read_vols(spm_vol(GM_files{i}));
        else
            current_image = niftiread(GM_files{i});
        end
        % Update the mask: a voxel becomes 'true' if it has a value > 0
        % in this image OR if it was already 'true' from a previous image
        mask = mask | (current_image > 0);
    end
    logger.success('Created the 3D mask!')

    % Calculate the number and percentage of active voxels in the mask
    voxelIdx = find(mask);
    M = numel(voxelIdx);
    logger.success('3D mask created with %d active voxels.', M);
    total_voxels = numel(GM0);
    active_percentage = (M / total_voxels) * 100;
    logger.success('Active voxels constitute %.2f%% of the total volume.', active_percentage);

    % --- Vectorization of raw data ---
    logger.info('Vectorizing and populating X_raw...');
    % Pre-allocate the data matrix X_raw with zeros
    X_raw = zeros(N, M);
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
    logger.success('Created the X_raw matrix!');


    % --- 1.3: TIV Normalization ---
    logger.info('Step 1.3: Applying multiplicative TIV normalization...');
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
        X_raw = X_raw .* TIV_norm;
        logger.success('TIV normalization applied!');
    catch ME
        logger.critical('Failed during TIV application! Reason: [%s] %s', ME.identifier, ME.message);
        error('An error occurred during TIV normalization.');
    end


    % --- 1.4: Creating Labels and Saving Data ---
    logger.info('Step 1.4: Creating labels and saving processed data...')
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


% --- Helper Functions ---

function pool = safe_parpool(logger)
    % Starts a parallel pool if available, otherwise runs in serial mode
    pool = []; % Default value (serial)
    try
        % Check if the Parallel Computing Toolbox is installed 
        if isempty(ver('parallel'))
            logger.warn('Parallel Computing Toolbox not found. Continuing in serial mode.');
            return; 
        end
        
        % Check if a default cluster profile is configured
        if isempty(parallel.defaultClusterProfile)
            logger.warn('Parallel Toolbox is installed, but no default execution environment is configured. Continuing in serial mode.');
            return; 
        end

        % Check if a pool is already running
        pool = gcp('nocreate');
        if isempty(pool)
            % If no pool is running, try to start a new one
            pool = parpool;
        end
        logger.success('Parallel pool is active with %d workers.', pool.NumWorkers);
        
    catch ME       
        logger.warn('An error occurred while starting the parallel pool. Continuing in serial mode. Reason: [%s] %s', ME.identifier, ME.message);
    end
end




  


