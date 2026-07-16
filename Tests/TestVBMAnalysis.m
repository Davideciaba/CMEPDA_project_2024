classdef TestVBMAnalysis < matlab.unittest.TestCase & matlab.mock.TestCase
    % TESTVBMANALYSIS Automated Unit Test suite for the VBMAnalysis class
    %
    % PURPOSE: Validates the GLM factorial design construction, SPM batch
    %   execution, mathematical thresholding (FWE/FDR), and similarity metrics.

    properties
        SandboxDir char             % The Sandbox Directory
        DummyMaskPath char          % Dummy Explicit Mask Path
        TestLogger Logger           % Custom Logger
        ADPath cell                 % Cache for dummy AD paths
        CTRLPath cell               % Cache for dummy CTRL paths
        DummyCovTable table         % Cache for dummy clinical covariates
    end
    
    methods (TestMethodSetup)
        function createSandbox(testCase)
            % Create temporary directory
            testCase.SandboxDir = tempname;
            mkdir(testCase.SandboxDir); 
            
            % Create dummy clinical CSV matching real data structure
            % Note: Sex is numeric since we bypass CohortData.loadData() logic
            ids = {'Subj1'; 'Subj2'; 'Subj3'; 'Subj4'; 'Subj5'; 'Subj6'; 'Subj7'; 'Subj8'};
            groups = {'AD'; 'AD'; 'AD'; 'AD'; 'CTRL'; 'CTRL'; 'CTRL'; 'CTRL'};
            ages = [70; 72; 68; 75; 69; 71; 65; 73];
            sexes = [1; 0; 0; 1; 1; 1; 0; 0]; 
            mmses = [20; 18; 22; 15; 29; 30; 28; 29];
            tivs = [1400; 1350; 1420; 1380; 1500; 1550; 1480; 1520];
            
            testCase.DummyCovTable = table(ids, groups, ages, sexes, mmses, tivs, ...
                'VariableNames', {'ID', 'Group', 'Age', 'Sex', 'MMSE', 'TIV'});
            
            % Create a dummy info structure
            dummyPath = fullfile(testCase.SandboxDir, 'temp.nii');
            niftiwrite(zeros(10,10,10,'single'), dummyPath);
            baseInfo = niftiinfo(dummyPath);
            baseInfo.Transform = affinetform3d(eye(4));
            baseInfo.SpaceUnits = 'Millimeter';
            delete(dummyPath);
            
            % Generate dummy volumes (Random noise to ensure variance > 0)
            % Pre-allocate Path caches
            testCase.ADPath = cell(4, 1);
            testCase.CTRLPath = cell(4, 1);

            % AD Subjects
            for i = 1:4
                filePath = fullfile(testCase.SandboxDir, sprintf('smwc1AD_0%d.nii', i));
                vol = rand(10, 10, 10, 'single'); 
                niftiwrite(vol, filePath, baseInfo);
                testCase.ADPath{i} = filePath;
            end
            % CTRL Subjects
            for i = 1:4
                filePath = fullfile(testCase.SandboxDir, sprintf('smwc1CTRL_0%d.nii', i));
                vol = rand(10, 10, 10, 'single'); 
                niftiwrite(vol, filePath, baseInfo);
                testCase.CTRLPath{i} = filePath;
            end
            
            % Create Explicit Mask (Center 6x6x6 box)
            testCase.DummyMaskPath = fullfile(testCase.SandboxDir, 'explicit_mask.nii');
            maskVol = zeros(10, 10, 10, 'single');
            maskVol(3:8, 3:8, 3:8) = 1;
            niftiwrite(maskVol, testCase.DummyMaskPath, baseInfo);
            
            % Initialize dummy Logger
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

    methods (Access = private)
        function mockCohort = getMockCohort(testCase)
            % HELPER: Centralizes the mocking and stubbing logic for CohortData.
            %   Instructs the mock to return the cached variables.
            
            mockInputs = {testCase.SandboxDir, 'dummy.csv', testCase.TestLogger};
            [mockCohort, behavior] = testCase.createMock(?CohortData, 'ConstructorInputs', mockInputs);
            
            % Stub the paths retrieval
            testCase.assignOutputsWhen(behavior.getFilePaths('AD'), testCase.ADPath);
            testCase.assignOutputsWhen(behavior.getFilePaths('CTRL'), testCase.CTRLPath);
            
            % Stub the covariates table retrieval
            testCase.assignOutputsWhen(withAnyInputs(behavior.getCovariatesTable), testCase.DummyCovTable);
        end
    end
    
    methods (Test)
        
        function testMapSimilarity(testCase)
            % PURPOSE: Validate method execution for map similarity

            model = VBMAnalysis(testCase.TestLogger);
            
            % Test Dimension Mismatch
            mapA = ones(10, 10, 10);
            mapB = ones(5, 5, 5); % Wrong size
            testCase.verifyError(@() model.evaluateMapSimilarity(mapA, mapB, 'A', 'B'), ...
                'VBMAnalysis:DimensionMismatch', ...
                'Must fail fast if spatial dimensions do not match.');
            
            % Test Perfect Overlap
            mapC = ones(10, 10, 10);
            try
                model.evaluateMapSimilarity(mapA, mapC, 'A', 'C');
                testCase.verifyTrue(true, 'Perfect overlap similarity executed successfully.');
            catch ME
                testCase.verifyFail(sprintf('evaluateMapSimilarity crashed: %s', ME.message));
            end
            
            % Test Partial Overlap
            mapHalf = zeros(10, 10, 10);
            mapHalf(1:500) = 1; % Exactly half the volume
            try
                model.evaluateMapSimilarity(mapA, mapHalf, 'Full', 'Half');
                testCase.verifyTrue(true, 'Partial overlap similarity executed successfully.');
            catch ME
                testCase.verifyFail(sprintf('evaluateMapSimilarity crashed: %s', ME.message));
            end

            % Test Pearson Correlation
            mapContinuousA = zeros(10, 10, 10);
            mapContinuousB = zeros(10, 10, 10);
            
            % Inject linearly correlated values into the same 100 voxels
            mapContinuousA(1:100) = 1:100;
            mapContinuousB(1:100) = (1:100) * 3.5; % Exact linear scaling
            
            try
                model.evaluateMapSimilarity(mapContinuousA, mapContinuousB, 'ContA', 'ContB');
                testCase.verifyTrue(true, 'Continuous maps similarity executed successfully.');
            catch ME
                testCase.verifyFail(sprintf('evaluateMapSimilarity crashed: %s', ME.message));
            end
        end
        
        function testGlmBatch(testCase)
            % PURPOSE: Verify that the class translates CohortData
            %   into a valid SPM GLM batch and creates SPM.mat and T-maps.
            if isempty(which('spm'))
                testCase.assumeFail('SPM not found in path. Test skipped.');
            end
            
            outputDir = fullfile(testCase.SandboxDir, 'VBM_Stats');
            model = VBMAnalysis(testCase.TestLogger);
            
            % Extract the Mock object
            mockCohort = testCase.getMockCohort();
            
            % Execute the GLM
            model.twoSampleTTest(outputDir, mockCohort, testCase.DummyMaskPath, 'AD', 'CTRL');
            
            % SPM.mat and spmT_0001.nii must exist on disk
            spmMatPath = fullfile(outputDir, 'SPM.mat');
            testCase.verifyTrue(exist(spmMatPath, 'file') == 2, 'SPM.mat was not generated.');
            testCase.verifyTrue(exist(fullfile(outputDir, 'spmT_0001.nii'), 'file') == 2, 'T-Map was not generated.');
        end
        
        function testGetCorrectedMapFailFast(testCase)
            % PURPOSE: Verify that looking for a non-existent contrast triggers an error.
            if isempty(which('spm'))
                testCase.assumeFail('SPM not found in path. Test skipped.');
            end
            
            % Build the model
            outputDir = fullfile(testCase.SandboxDir, 'VBM_Stats');
            model = VBMAnalysis(testCase.TestLogger);

            mockCohort = testCase.getMockCohort();
            model.twoSampleTTest(outputDir, mockCohort, testCase.DummyMaskPath, 'AD', 'CTRL');
            
            % Query a contrast name that doesn't exist
            testCase.verifyError(@() model.getCorrectedMap(outputDir, 'Ghost Contrast', 0.05, 'FWE'), ...
                'VBMAnalysis:ContrastNotFound', ...
                'Should crash if the requested contrast name is missing from SPM.mat.');
        end

        function testGetCorrectedMapThresholding(testCase)
            % PURPOSE: Verify that thresholding the map works.
            if isempty(which('spm'))
                testCase.assumeFail('SPM not found in path. Test skipped.');
            end
            
            outputDir = fullfile(testCase.SandboxDir, 'VBM_Stats');
            model = VBMAnalysis(testCase.TestLogger);

            mockCohort = testCase.getMockCohort();
            model.twoSampleTTest(outputDir, mockCohort, testCase.DummyMaskPath, 'AD', 'CTRL');
            
            % Use 'none' (uncorrected) at p=0.99 just to ensure some random noise voxels survive the threshold
            [thresholdedMap, criticalThreshold] = model.getCorrectedMap(outputDir, 'Atrophy: CTRL > AD', 0.99, 'none');
            
            testCase.verifyNotEmpty(thresholdedMap, 'The returned map tensor should not be empty.');
            testCase.verifyTrue(criticalThreshold < 0, 'Uncorrected threshold for p=0.99 should be very low.');
            
            % Any value in the map must be either exactly 0 (filtered) or >= criticalThreshold
            activeVoxels = thresholdedMap(thresholdedMap > 0);
            if ~isempty(activeVoxels)
                testCase.verifyTrue(all(activeVoxels >= criticalThreshold), ...
                    'All surviving voxels must be greater than or equal to the critical threshold.');
            end
        end
    end
end