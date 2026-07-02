classdef TestCohortData < matlab.unittest.TestCase
    % TESTCOHORTDATA Automated Unit Test suite for the CohortData class.
    %
    % PURPOSE: Implements an automated Sandbox. It generates dummy NIfTI 
    %   volumes and clinical CSV in a temporary hidden directory before 
    %   each test, evaluates the methods and destroys the sandbox afterwards.
    
    properties
        SandboxDir char    % The Sandbox Directory
        DummyCsvPath char  % Dummy clinical CSV Path
        TestLogger Logger  % Custom Logger
    end
    
    methods (TestMethodSetup)
        function createSandbox(testCase)
            % Create temporary directory
            testCase.SandboxDir = tempname;
            mkdir(testCase.SandboxDir);

            % Create a subfolder to simulate real nested structures
            niftiSubDir = fullfile(testCase.SandboxDir, 'NIfTI_Files');
            mkdir(niftiSubDir);
            
            % Create dummy clinical CSV matching real data structure
            testCase.DummyCsvPath = fullfile(testCase.SandboxDir, 'covariate_data.csv');
            
            % Defining table with ID, Group, Age, Sex, MMSE, TIV
            dummyTable = table({'Subj1'; 'Subj2'; 'Subj3'}, {'AD'; 'AD'; 'CTRL'}, ...
                [75; 68; 70], {'M'; 'F'; 'M'}, [22; 29; 28], [1450; 1520; 1480], ...
                'VariableNames', {'ID', 'Group', 'Age', 'Sex', 'MMSE', 'TIV'});
                
            writetable(dummyTable, testCase.DummyCsvPath);
            
            % Create Valid NIfTI files (Dummy 10x10x10 matrices)
            dummyVol = zeros(10, 10, 10, 'single');
            dummyVol(5,5,5) = 1; % Add some signal
            
            niftiwrite(dummyVol, fullfile(niftiSubDir, 'smwc1AD_01.nii'));
            niftiwrite(dummyVol, fullfile(niftiSubDir, 'smwc1AD_02.nii'));
            niftiwrite(dummyVol, fullfile(niftiSubDir, 'smwc1CTRL_01.nii'));
            
            % Create a corrupted NIfTI file
            corruptedPath = fullfile(niftiSubDir, 'smwc1CTRL_02_CORRUPT.nii');
            writelines("This is not a real NIfTI file, it will crash niftiread", corruptedPath);
            
            % Suppress console spam during tests
            testCase.TestLogger = Logger('NullLogger'); 
        end
    end
    
    methods (TestMethodTeardown)
        function destroySandbox(testCase)
            % Clean up the temporary folder entirely regardless of test result
            if exist(testCase.SandboxDir, 'dir')
                rmdir(testCase.SandboxDir, 's');
            end
        end
    end
    
    methods (Test)

        function testCsvNotFound(testCase)
            % PURPOSE: Verify scanDirectory throws an error if CSV pattern is missing.
            emptyDir = tempname;
            mkdir(emptyDir);
            
            % Pass emptyDir as the RootDirectory with explicit name
            cohort = CohortData(emptyDir, 'covariate_data.csv', testCase.TestLogger);
            
            testCase.verifyError(@() cohort.scanDirectory(), ...
                'CohortData:CsvNotFound', ...
                'scanDirectory must throw CsvNotFound if the exact CSV is missing.');
                
            rmdir(emptyDir, 's');
        end

        function testStateErrorBeforeScan(testCase)
            % PURPOSE: Verify State Error in getFilePaths before scanDirectory.
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            
            testCase.verifyError(@() cohort.getFilePaths('ALL'), ...
                'CohortData:StateError', ...
                'Should block path extraction if scanDirectory has not been called.');
        end
        
        function testStateErrorBeforeLoad(testCase)
            % PURPOSE: Verify State Error in getVolumes before loadData.
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory();
            
            testCase.verifyError(@() cohort.getVolumes('AD'), ...
                'CohortData:StateError', ...
                'Should block tensor access if loadData has not been called.');
        end
        
        function testScanDirectoryGrouping(testCase)
            % PURPOSE: Verify if file regex scanning correctly groups AD and CTRL.
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory();
            
            % Verify counts (2 AD and 2 CTRL) and CSV path
            testCase.verifyEqual(numel(cohort.getFilePaths('AD')), 2, 'Should find exactly 2 AD files.');
            testCase.verifyEqual(numel(cohort.getFilePaths('CTRL')), 2, 'Should find exactly 2 CTRL files.');
            testCase.verifyEqual(numel(cohort.getFilePaths('ALL')), 4, 'Should find exactly 4 total files.');
            testCase.verifyEqual(cohort.getCovariatesPath(), testCase.DummyCsvPath, 'Should automatically resolve the CSV path.');
        end
        
        function testAmbiguityCsv(testCase)
            % PURPOSE: Verify ambiguity on CSV file name.
            
            % Create a duplicate of the CSV in a subfolder
            duplicateSubDir = fullfile(testCase.SandboxDir, 'AnotherFolder');
            mkdir(duplicateSubDir);
            copyfile(testCase.DummyCsvPath, fullfile(duplicateSubDir, 'covariate_data.csv'));
            
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            
            % Must fail because it finds 2 files with the exact same name
            testCase.verifyError(@() cohort.scanDirectory(), ...
                'CohortData:AmbiguousCsv', ...
                'Should crash if multiple CSV files with the same name exist.');
        end

        function testEmptyDirectory(testCase)
            % PURPOSE: Verify that a directory with a CSV but no nifti throws the error.
            emptyNiftiDir = tempname;
            mkdir(emptyNiftiDir);
            
            % Inject CSV so it bypasses the first check
            copyfile(testCase.DummyCsvPath, fullfile(emptyNiftiDir, 'covariate_data.csv'));
            
            cohort = CohortData(emptyNiftiDir, 'covariate_data.csv', testCase.TestLogger);
            testCase.verifyError(@() cohort.scanDirectory(), 'CohortData:EmptyDirectory');
            
            rmdir(emptyNiftiDir, 's'); 
        end
        
        function testEmptyClinicalGroup(testCase)
            % PURPOSE: Verify error on unbalanced/empty clinical groups.
            
            % Remove all CTRL files to simulate an unbalanced directory
            delete(fullfile(testCase.SandboxDir, 'NIfTI_Files', '*CTRL*.nii'));
            
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            
            % Scan must fail because the CTRL group is 0
            testCase.verifyError(@() cohort.scanDirectory(), ...
                'CohortData:EmptyClinicalGroup', ...
                'Must fail if a specific clinical group has zero files.');
        end

        function testCorruptedVolume(testCase)
            % PURPOSE: Verify that corrupted files halt the pipeline
            
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory();
            
            testCase.verifyError(@() cohort.loadData(), ?MException, ...
                'Pipeline must crash when encountering corrupted NIfTI file.');
        end

        function testCorruptedCsv(testCase)
            % PURPOSE: Verify the readtable logic.
            
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory(); % Finds the CSV successfully
            
            % Delete the CSV file before loading to trigger the readtable error
            delete(testCase.DummyCsvPath);
            
            % The loadData method must crash because the CovariatesPath is now unreadable
            testCase.verifyError(@() cohort.loadData(), ?MException, ...
                'Pipeline must crash if CSV becomes unreadable during data load.');
        end

        function testMissingCsvColumns(testCase)
            % PURPOSE: Verify validation of required CSV columns.
            
            % Overwrite the dummy CSV with missing the 'Age' column
            badTable = table({'Subj1'}, {'AD'}, {'M'}, [22], [1450], ...
                'VariableNames', {'ID', 'Group', 'Sex', 'MMSE', 'TIV'});
            writetable(badTable, testCase.DummyCsvPath);
            
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory();
            
            testCase.verifyError(@() cohort.loadData(), 'CohortData:MissingCsvColumns', ...
                'Should crash if required columns are missing from the CSV.');
        end

        function testInvalidNumericColumns(testCase)
            % PURPOSE: Verify non-numeric continuous variables.
            
            % Overwrite with a CSV containing string data in the 'Age' column
            badTable = table({'Subj1'}, {'AD'}, {'Seventy'}, {'M'}, [22], [1450], ...
                'VariableNames', {'ID', 'Group', 'Age', 'Sex', 'MMSE', 'TIV'});
            writetable(badTable, testCase.DummyCsvPath);
            
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory();
            
            testCase.verifyError(@() cohort.loadData(), 'CohortData:InvalidDataType', ...
                'Should crash if continuous columns (Age, MMSE, TIV) contain non-numeric data.');
        end

        function testInvalidSexFormat(testCase)
            % PURPOSE: Verify wrong Sex formatting.
            
            % Overwrite with a CSV containing datetime data in the Sex column
            badTable = table({'Subj1'}, {'AD'}, 70, datetime('now'), 22, 1450, ...
                'VariableNames', {'ID', 'Group', 'Age', 'Sex', 'MMSE', 'TIV'});
            writetable(badTable, testCase.DummyCsvPath);
            
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory();
            
            testCase.verifyError(@() cohort.loadData(), 'CohortData:InvalidSexFormat', ...
                'Should crash if the Sex column contains unsupported data types.');
        end

        function testSexBinaryValidation(testCase)
            % PURPOSE: Verify that Sex is numeric and contains 0s and 1s.
            
            % Overwrite with a CSV containing invalid numeric values
            badTable = table({'Subj1'; 'Subj2'}, {'AD'; 'CTRL'}, [70; 65], [1; 2], [22; 28], [1450; 1500], ...
                'VariableNames', {'ID', 'Group', 'Age', 'Sex', 'MMSE', 'TIV'});
            writetable(badTable, testCase.DummyCsvPath);
            
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory();
            
            testCase.verifyError(@() cohort.loadData(), 'CohortData:InvalidSexValues', ...
                'Should crash if a numeric Sex column contains values other than 0s and 1s.');
        end

        function testSexBinarization(testCase)
            % PURPOSE: Verify Sex column is correctly transformed into 0s and 1s.
            
            % Remove the corrupted NIfTI to allow loadData to succeed
            delete(fullfile(testCase.SandboxDir, 'NIfTI_Files', '*CORRUPT*.nii'));
            
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory();
            cohort.loadData();
            
            covTable = cohort.getCovariatesTable();
            
            expectedSex = [1; 0; 1];
            testCase.verifyEqual(covTable.Sex, expectedSex, ...
                'The Sex covariate should be binarized to M=1 and F=0.');
        end

        function testGetReferenceInfo(testCase)
            % PURPOSE: Verify spatial metadata extraction from a valid NIfTI file.
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory();
            
            refInfo = cohort.getReferenceInfo();
            
            % Verify expected fields exist
            expectedFields = {'ImageSize' 'AffineTransform', 'RawNiftiInfo', 'NumericMatrix'};
            if ~isempty(which('spm'))
                expectedFields{end+1} = 'SpmMatrix'; % Also expect SpmMatrix if SPM is present
            end
            for i = 1:numel(expectedFields)
                testCase.verifyTrue(isfield(refInfo, expectedFields{i}), ...
                    sprintf('infoStruct must contain %s', expectedFields{i}));
            end
            
            % Verify correct ImageSize
            testCase.verifyEqual(refInfo.ImageSize, [10 10 10], 'ImageSize should match the 10x10x10 dummy volume.');
        end
        
        function testGetReferenceInfoEmptyCohort(testCase)
            % PURPOSE: Verify that calling getReferenceInfo before scanDirectory throws the error.
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            
            testCase.verifyError(@() cohort.getReferenceInfo(), 'CohortData:StateError');
        end
        
        function testGetMeanVolume(testCase)
            % PURPOSE: Verify that the getMeanVolume calculates the correct 3D mean tensor.
            % Remove the corrupted file
            delete(fullfile(testCase.SandboxDir, 'NIfTI_Files', '*CORRUPT*.nii'));
            
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory();
            cohort.loadData();
            
            % Extract the mean
            meanAll = cohort.getMeanVolume('ALL');
            
            % Test Dimensionality
            testCase.verifyEqual(size(meanAll), [10 10 10], ...
                'Mean volume must be 3D (10x10x10).');
                
            % Since all valid dummy volumes have a value of 1 at (5,5,5) and 0 elsewhere,
            % the mathematical mean must exactly preserve this.
            testCase.verifyEqual(meanAll(5,5,5), single(1), ...
                'Mean computation failed at the active voxel. Expected 1.');
            testCase.verifyEqual(meanAll(1,1,1), single(0), ...
                'Mean computation failed at the background voxel. Expected 0.');
        end

        function testGetSubjVolume(testCase)
            % PURPOSE: Verify correct extraction of a single subject's volume.
            
            % Remove the corrupted NIfTI to allow loadData to succeed
            delete(fullfile(testCase.SandboxDir, 'NIfTI_Files', '*CORRUPT*.nii'));
            
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory();
            cohort.loadData();
            
            % We injected 'smwc1AD_01.nii' in the TestMethodSetup
            volAD1 = cohort.getSubjVolume('AD_01');
            
            % Test Dimensionality (must be exactly 3D)
            testCase.verifyEqual(size(volAD1), [10 10 10], ...
                'Extracted subject volume must be 3D (10x10x10).');
                
            % Test Data Integrity
            testCase.verifyEqual(volAD1(5,5,5), single(1), ...
                'Extracted volume did not match the expected active voxel.');
        end
        
        function testGetSubjVolumeFailure(testCase)
            % PURPOSE: Verify the fail fast logic for missing or ambiguous subject IDs.
            
            % Remove the corrupted NIfTI to allow loadData to succeed
            delete(fullfile(testCase.SandboxDir, 'NIfTI_Files', '*CORRUPT*.nii'));
            
            cohort = CohortData(testCase.SandboxDir, 'covariate_data.csv', testCase.TestLogger);
            cohort.scanDirectory();
            cohort.loadData();
            
            % Missing subject check
            testCase.verifyError(@() cohort.getSubjVolume('GHOST_99'), ...
                'CohortData:SubjectNotFound', ...
                'Must fail if the subject ID does not exist in the loaded NIfTI array.');
                
            % Ambiguous subject sheck 
            % 'AD' matches both 'smwc1AD_01.nii' and 'smwc1AD_02.nii'
            testCase.verifyError(@() cohort.getSubjVolume('AD'), ...
                'CohortData:AmbiguousSubjectID', ...
                'Must fail and refuse to guess if the ID string is ambiguous.');
        end
    end
end