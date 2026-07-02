classdef TestLogger < matlab.unittest.TestCase
    % TESTLOGGER Automated Unit Test suite for the Logger class.
    %
    % PURPOSE: Implements an automated Sandbox to test file I/O, rotation,
    %   and garbage collection.
    
    properties
        SandboxDir char     % The Sandbox Directory
    end
    
    methods (TestMethodSetup)
        function createSandbox(testCase)
            % Create temporary directory
            testCase.SandboxDir = tempname;
            mkdir(testCase.SandboxDir);
        end
    end
    
    methods (TestMethodTeardown)
        function destroySandbox(testCase)
            % Create temporary directory
            if exist(testCase.SandboxDir, 'dir')
                rmdir(testCase.SandboxDir, 's');
            end
        end
    end
    
    methods (Test)
        
        function testInstantiationAndConsole(testCase)
            % PURPOSE: Verify basic instantiation and property defaults.
            log = Logger("TestRoot");
            
            testCase.verifyEqual(log.name, "TestRoot", 'Logger name should be initialized correctly.');
            testCase.verifyTrue(isa(log.extraContext, 'containers.Map'), 'Context should be initialized as a Map.');
            
            % Add console handler
            log.addConsoleHandler(level="TRACE", useColors=true);
            log.success("Console instantiation test passed.");
        end
        
        function testContextInjection(testCase)
            % PURPOSE: Verify context injection and clearance mechanisms.
            log = Logger();
            
            log.addContext("SubjectID", "002_S_0816");
            testCase.verifyTrue(log.extraContext.isKey("SubjectID"), 'Context Map should contain the injected key.');
            testCase.verifyEqual(log.extraContext("SubjectID"), "002_S_0816", 'Context value should match.');
            
            log.clearContext();
            testCase.verifyTrue(isempty(log.extraContext.keys), 'Context Map should be empty after clearing.');
        end
        
        function testSprintfWrapper(testCase)
            % PURPOSE: Verify that sprintf formatting works and writes to disk.
            logFile = fullfile(testCase.SandboxDir, 'sprintf_test.log');
            log = Logger();
            log.addFileHandler(logFile, level="INFO");
            
            tivValue = 1450.32;
            maskThreshold = 0.2;
            log.info("TIV (%.2f) deviates. Threshold set to %.2f.", tivValue, maskThreshold);
            
            % Forces fclose(h.fh) before we read the file
            delete(log); 
            
            % Assert file content
            content = fileread(logFile);
            testCase.verifyTrue(contains(content, "TIV (1450.32) deviates. Threshold set to 0.20."), ...
                'The log file should contain the correctly formatted sprintf string.');
        end
        
        function testFileRotation(testCase)
            % PURPOSE: Verify that exceeding byte limit triggers archival rotation
            %   and produces the exact expected number of files.
            baseLogPath = fullfile(testCase.SandboxDir, 'rotation_test.log');
            log = Logger();
            
            % Set the rotation limit, considering the logger message format 
            % (timestamp | levelName | caller | message_out)
            log.addFileHandler(baseLogPath, level="DEBUG", rotation=400);
            
            % Write lines to push past 400 bytes
            log.info("Line 1: Padding to trigger rotation..........................");
            log.info("Line 2: Padding to trigger rotation..........................");
            log.info("Line 3: Padding to trigger rotation..........................");
            
            log.info("Line 4: Padding to trigger rotation..........................");
            
            % Close file handles to allow system inspection
            delete(log);
            
            % Verify the new file still exists
            testCase.verifyTrue(exist(baseLogPath, 'file') == 2, 'The new log file must exist.');
            
            % Check Sandbox for rotated files (archived with timestamps containing '_')
            rotatedFiles = dir(fullfile(testCase.SandboxDir, 'rotation_test_*.log'));
            
            % Verify exactly 1 rotation should have occurred
            testCase.verifyEqual(numel(rotatedFiles), 1, ...
                'File rotation failed: Expected exactly 1 archived log based on byte limit.');
        end
        
        function testGarbageCollection(testCase)
            % PURPOSE: Verify the destructor deletes 0-byte log files.
            emptyLogPath = fullfile(testCase.SandboxDir, 'empty_garbage.log');
            log = Logger();
            log.addFileHandler(emptyLogPath);
            
            % Do not write any logs and verify the file exists
            testCase.verifyTrue(exist(emptyLogPath, 'file') == 2, 'File should be created initially.');
            
            % Trigger the destructor
            delete(log);
            
            % Verify the file was cleaned up
            testCase.verifyFalse(exist(emptyLogPath, 'file') == 2, 'Destructor failed to delete the 0-byte log file.');
        end
        
        function testTypeValidations(testCase)
            % PURPOSE: Verify MATLAB arguments blocks reject improper types.
            log = Logger();
            
            % Check 1: Numeric masquerading as Logical (useColors=123)
            testCase.verifyError(@() log.addConsoleHandler(useColors=123), ...
                'MATLAB:validators:mustBeA', ...
                'Should reject numeric inputs for logical arguments.');
                
            % Check 2: Negative rotation values
            testCase.verifyError(@() log.addFileHandler(fullfile(testCase.SandboxDir, 'dummy.log'), rotation=-50), ...
                'MATLAB:validators:mustBePositive', ...
                'Should reject negative numbers for byte rotation limits.');
                
            % Check 3: Invalid Level Name
            testCase.verifyError(@() log.log("FAKE_LEVEL", "This should crash."), ...
                ?MException, ... % Using general MException since error string is custom
                'Should crash when an unknown log level is requested.');
        end
    end
end