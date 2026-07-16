function validateMatlabEnv()
    % VALIDATEMATLABENVIRONMENT Verifies the presence and licensing of required MATLAB toolboxes.

    % Define the requirements for the pipeline
    % - 'displayName': commercial name.
    % - 'verName': The internal directory name used by the 'ver' command.
    % - 'licenseName': The internal feature name for license checkout.
    requiredToolboxes = struct(...
        'displayName', {'Image Processing Toolbox', 'Statistics and Machine Learning Toolbox'}, ...
        'verName', {'images', 'stats'}, ...
        'licenseName', {'image_toolbox', 'statistics_toolbox'} ...
    );

    fprintf('Starting MATLAB Environment Validation...\n');

    for idx = 1:length(requiredToolboxes)
        currentBox = requiredToolboxes(idx);
        fprintf('Verifying: %s... ', currentBox.displayName);

        % Version check
        installedVersion = ver(currentBox.verName);
        if isempty(installedVersion)
            error('validateEnvironment:MissingToolbox', ...
                'FATAL: %s is not installed. Please install it via MATLAB Add-On Explorer.', ...
                currentBox.displayName);
        end

        % License check
        licenseAvailable = license('checkout', currentBox.licenseName);
        if ~licenseAvailable
            error('validateEnvironment:MissingLicense', ...
                'FATAL: Valid license not found for %s.', ...
                currentBox.displayName);
        end

        fprintf('OK.\n');
    end

    fprintf('MATLAB environment validation passed successfully.\n');
end