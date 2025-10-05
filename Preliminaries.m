function Preliminaries()
% === PIPELINE SECTION 1: COMMON PRELIMINARIES ===
% This script executes the common preliminary steps for both the SVM and
% EfficientNet3D models
% The script is capable of operating with or without SPM and the Parallel Computing Toolbox.

% --- Logger Initialization ---
logger = Logger('Preliminaries');

% Logger ConsoleHandler configuration
logger.addConsoleHandler('use_colors', true);

% Ask the user if they want to create a log file
setup_file_logging(logger);

% --- Attempt to start a parallel pool ---
logger.info('Checking for parallel computing availability...');
safe_parpool(logger);

logger.info('=== PIPELINE SECTION 1: START ===');
logger.info('Objective: Data preparation for linear SVM and EfficientNet3D models');
% --- 1.1: Reading NIfTI files and dimension check ---
logger.info('Step 1.1: Reading NIfTI files and dimension check...');

% Define data paths
dirname_AD = "../CMEPDA_project_2024/AD_CTRL/AD_s3/";
dirname_CTRL = "../CMEPDA_project_2024/AD_CTRL/CTRL_s3/";

% Check if the input folders exist
if ~exist(dirname_AD, 'dir') || ~exist(dirname_CTRL, 'dir')
    logger.critical('Input directories not found. Aborting.');
    return;
end

logger.success('AD and CTRL directories found!');

% --- Check SPM availability and get user preference ---
use_spm = check_spm_availability(logger);


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
        logger.critical('Failed during file selection with SPM (%s): Reason: [%s] %s',  spm('Ver'), ME.identifier, ME.message);
        return;
    end

else
    logger.info('Using standard MATLAB functions to find NIfTI files...');
    
    % Check to immediately stop the script if the Image Processing
    % Toolbox is missing
    if isempty(which('niftiread'))
        logger.critical('Function "niftiread" (Image Processing Toolbox) is required when not using SPM. Aborting execution.');
        return;
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
    return;
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
    return;
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
    return;
end
logger.success('All %d volumes have consistent dimensions!', N);


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
    tiv_path = '../CMEPDA_project_2024/covariateADCTRLsexAgeTIV.csv';
    TIV = readtable(tiv_path).TIV;
    % Check if the number of TIV entries matches the number of subjects
    if numel(TIV) ~= N
        logger.critical('TIV count (%d) does not match subject count (%d).', numel(TIV), N);
        return;
    end
    % Normalize TIV values by dividing each value by the maximum TIV.
    TIV_norm = TIV / max(TIV);
    % Element-wise multiplication with the data matrix.
    X_raw = X_raw .* TIV_norm;
    logger.success('TIV normalization applied!');
catch ME
    logger.critical('Failed during TIV application! Reason: [%s] %s', ME.identifier, ME.message);
    return;
end


% --- 1.4: Creating Labels and Saving Data ---
logger.info('Step 1.4: Creating labels and saving processed data...')
% ---Creating labels ---
logger.info('Creating labels...')
% Create labels (1 = AD, 0 = CTRL) based on file order
y_all = [ones(N_AD, 1); zeros(N_CTRL, 1)];
try
    % Get the directory of the currently running script
    script_dir = fileparts(mfilename('fullpath'));
    % Construct the full path for the output file
    output_path = fullfile(script_dir, 'preliminari.mat');
    % Save the variables to the specified path.ù
    save(output_path, 'X_raw', 'y_all', 'mask', 'voxelIdx', 'M', 'N', '-v7.3');
    logger.success('File saved successfully to %s', output_path);
catch ME
    logger.critical('Failed during save! Reason: [%s] %s', ME.identifier, ME.message);
    return;
end


logger.info('=== END PIPELINE SECTION 1 ===');
end


% --- Helper Functions ---

function pool = safe_parpool(logger)
    % Start a parallel pool if available, otherwise run in serial mode
    pool = []; % Default value (serial)
    try
        % Check if the Parallel Computing Toolbox is installed 
        if ~isempty(ver('parallel'))
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
        logger.warn('An error occurred while starting the parallel pool. Continuing in serial mode.');
        logger.warn('Reason: [%s] %s', ME.identifier, ME.message);
    end
end


function setup_file_logging(logger)
    % Ask the user if they want to create a log file
    default_answer = 'y';
    prompt_color = '_*blue';
    prompt_text = sprintf('Do you want to create a log file? [%s]:', default_answer);
    
    % Use cprintf for a colored prompt if it's on the path
    if ~isempty(which('cprintf'))
        cprintf(prompt_color, prompt_text);
        user_input = input(' ', 's');
    else
        % Fallback to standard input prompt
        user_input = input(prompt_text, 's');
    end
    
    % If the user just presses Enter, use the default answer
    if isempty(user_input)
        user_input = default_answer;
    end
    
    % Check the user's response
    if strcmpi(user_input, 'y') || strcmpi(user_input, 'yes')
        filename = 'log_preliminaries.txt';
        % Add a file handler with default level (DEBUG) and a rotation size of 50KB
        logger.addFileHandler(filename, 'level', 'DEBUG', 'rotation', 50*1024);
        logger.info('File logging enabled. Saving logs to %s', filename);
    else
        logger.info('File logging disabled. Using console output only.');
    end
end


function use_spm = check_spm_availability(logger)
    % Check for SPM availability and prompt the user for their preference
    % Return true if SPM is available and the user agrees to use it
    spm_in_path = (exist('spm_vol', 'file') == 2);

    if spm_in_path
        logger.info('SPM (%s) found in MATLAB path.', spm('Ver'));
        default_answer = 'y';
    else
        logger.warn('SPM not found in MATLAB path.');
        default_answer = 'n';
    end
    
    % Ask the user
    prompt_color = '_*blue';
    prompt_text = sprintf('Do you want to use SPM (%s) functions? [%s]:', spm('Ver'), default_answer);
    
    % Use cprintf for a colored prompt if it's available
    if ~isempty(which('cprintf'))
        cprintf(prompt_color, prompt_text);
        user_input = input(' ', 's');
    else
        % Fallback to the standard input prompt
        user_input = input(prompt_text, 's');
    end
    
    % If the user just presses Enter, use the default answer
    if isempty(user_input)
        user_input = default_answer;
    end
    
    % Handle the case where the user wants SPM, but it's not available
    use_spm = strcmpi(user_input, 'y') || strcmpi(user_input, 'yes');
    if use_spm && ~spm_in_path
        logger.error('SPM required but not found. Add SPM to your MATLAB path.');
        use_spm = false; % Forces to false to avoid errors
        return;
    end
    
    % Log the final decision
    if use_spm
        logger.info('SPM (%s) functions will be used for NIfTI operations.', spm('Ver'));
    else
        logger.info('Standard MATLAB functions will be used for NIfTI operations.');
    end
end

  


