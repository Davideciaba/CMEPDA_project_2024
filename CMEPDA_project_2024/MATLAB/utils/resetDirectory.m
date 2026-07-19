function resetDirectory(dirPath)
    % Purge an existing directory and recreates it
    if exist(dirPath, 'dir')
        [status, msg, ~] = rmdir(dirPath, 's');
        if status == 0
            error('Pipeline:DirectoryPurgeFailed', ...
                'Could not purge directory %s. A file might be open in another program. Reason: %s', ...
                dirPath, msg);
        end
    end
    mkdir(dirPath);
end