function handleError(loggerObj, personalizedMessage, ME)
    % Extracts the stack trace and logs it before killing the pipeline
    loggerObj.critical('%s', personalizedMessage);
    loggerObj.critical('Reason: %s', ME.message);
    
    if ~isempty(ME.stack)
        loggerObj.critical('--- Stack Trace ---');
        for i = 1:length(ME.stack)
            [~, fName, fExt] = fileparts(ME.stack(i).file);
            loggerObj.critical(' -> File: %s%s | Function: %s | Line: %d', ...
                fName, fExt, ME.stack(i).name, ME.stack(i).line);
        end
    end
    
    rethrow(ME);
end