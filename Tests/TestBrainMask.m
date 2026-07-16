classdef TestBrainMask < matlab.unittest.TestCase & matlab.mock.TestCase
    % TESTBRAINMASK Automated Unit Test suite for the BrainMask class.
    %
    % PURPOSE: Implements a Unit Test using Mock Objects
    %   to validate mathematical and logical behavior, isolating the tests 
    %   from Disk I/O. A partial micro-sandbox is retained for 
    %   boundary tests involving SPM TPM loading and NIfTI exports.
    
    properties
        SandboxDir char             % The Sandbox Directory
        TpmPath char                % SPM TPM Path
        TpmCorruptedPath char       % SPM TPM path to TPM with incorrect dimensions
        TestLogger Logger           % Custom Logger
        RefInfo struct              % Dummy patial metadata
    end
    
    methods (TestMethodSetup)
        function createSandbox(testCase)
            % Initialize dummy Logger
            testCase.TestLogger = Logger('NullLogger'); 

            % Setup standard dummy metadata
            testCase.RefInfo = struct();
            testCase.RefInfo.ImageSize = [10 10 10];
            testCase.RefInfo.RawNiftiInfo = struct('ImageSize', [10 10 10], 'Datatype', 'single');

            if ~isempty(which('spm'))
                testCase.RefInfo.SpmMatrix = eye(4);
            end

            % Micro sandbox for I/O tests
            testCase.SandboxDir = tempname;
            mkdir(testCase.SandboxDir);
             
            % Create a dummy info structure for SPM interaction 
            dummyPath = fullfile(testCase.SandboxDir, 'temp.nii');
            niftiwrite(zeros(10,10,10,'single'), dummyPath);
            baseInfo = niftiinfo(dummyPath);
            baseInfo.Transform = affinetform3d(eye(4));
            baseInfo.SpaceUnits = 'Millimeter';
            testCase.RefInfo.RawNiftiInfo = baseInfo;
            delete(dummyPath);
            
            % Generate dummy TPM volume (with variance to test Ridgway)
            testCase.TpmPath = fullfile(testCase.SandboxDir, 'dummy_TPM.nii');
            tpmVol = rand(10, 10, 10, 'single'); % Rand ensures var(tpm) > 0
            niftiwrite(tpmVol, testCase.TpmPath, baseInfo);
            
            % Generate TPM with incorrect dimensions to test spatial
            % alignment
            testCase.TpmCorruptedPath = fullfile(testCase.SandboxDir, 'bad_dim_TPM.nii');
            niftiwrite(zeros(5, 5, 5, 'single'), testCase.TpmCorruptedPath);
            
            % Force RefInfo to mirror how SPM reads the NIfTI file.
            % MATLAB (niftiinfo) and SPM (spm_vol) use different engines
            % to parse spatial metadata
            if ~isempty(which('spm'))
                spmHeader = spm_vol(testCase.TpmPath);
                testCase.RefInfo.NumericMatrix = single(spmHeader.mat);
                testCase.RefInfo.SpmMatrix = single(spmHeader.mat);
            else
                % Fallback if SPM is missing (tests will be skipped anyway)
                testCase.RefInfo.NumericMatrix = eye(4);
            end
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
        
        function testMaskState(testCase)
            % PURPOSE: Verify that the MaskType updates automatically and correctly
            %   based on the compute method called.
            mockInputs = {testCase.SandboxDir, 'dummy.csv', testCase.TestLogger};
            [mockCohort, behavior] = testCase.createMock(?CohortData, 'ConstructorInputs', mockInputs);
            
            % Create a fake mean volume because computeMeanMask invokes the
            % CohortData's method getMeanVolume('ALL') 
            fakeMean = zeros(10, 10, 10, 'single');
            testCase.assignOutputsWhen(behavior.getMeanVolume('ALL'), fakeMean);

            % Instantiation leaves the state 'Unallocated'
            mask = BrainMask(testCase.RefInfo, testCase.TestLogger);
            testCase.verifyEqual(mask.MaskType, 'Unallocated', 'Initial state should be Unallocated.');
            
            % Compute method automatically injects the correct identity
            mask.computeMeanMask(mockCohort, 0.6);
            testCase.verifyEqual(mask.MaskType, 'Mean Mask', 'State should automatically update to Mean Mask after computation.');
        end
        
        function testComputeMeanMask(testCase)
            % PURPOSE: Verify the mathematical computation of the mean mask.
            mockInputs = {testCase.SandboxDir, 'dummy.csv', testCase.TestLogger};
            [mockCohort, behavior] = testCase.createMock(?CohortData, 'ConstructorInputs', mockInputs);

            % Create a fakeMean tensor and setup the mock behavior
            fakeMean = zeros(10, 10, 10, 'single');
            fakeMean(5,5,5) = 1.0; 
            fakeMean(6,6,6) = 0.5;
            testCase.assignOutputsWhen(behavior.getMeanVolume('ALL'), fakeMean);

            mask = BrainMask(testCase.RefInfo, testCase.TestLogger);

            mask.computeMeanMask(mockCohort, 0.6);
            
            testCase.verifyEqual(mask.MaskType, 'Mean Mask', 'Mask identity should be Mean Mask.');
            testCase.verifyTrue(mask.Matrix(5,5,5), 'Voxel with mean 1.0 should pass the 0.6 threshold.');
            testCase.verifyFalse(mask.Matrix(6,6,6), 'Voxel with mean 0.5 should NOT pass the 0.6 threshold.');
            
            % Verify BrainMask requested the 'ALL' group
            testCase.verifyCalled(behavior.getMeanVolume('ALL'));
        end
        
        function testComputeConsensusMask(testCase)
            % PURPOSE: Verify the voting math for the Consensus Mask.
            mockInputs = {testCase.SandboxDir, 'dummy.csv', testCase.TestLogger};
            [mockCohort, behavior] = testCase.createMock(?CohortData, 'ConstructorInputs', mockInputs);

            % Generate a 4D tensor simulating N=2 subjects
            fakeVols = zeros(10, 10, 10, 2, 'single');
            fakeVols(5,5,5,1) = 1; fakeVols(5,5,5,2) = 1; % Voxel present in 100% of cohort
            fakeVols(6,6,6,1) = 1;                        % Voxel present in 50% of cohort
            testCase.assignOutputsWhen(behavior.getVolumes('ALL'), fakeVols);

            mask = BrainMask(testCase.RefInfo, testCase.TestLogger);
            
            % Require the voxel to be present in 100% of patients (ratio 1.0)
            mask.computeConsensusMask(mockCohort, 0.5, 1.0);
            
            testCase.verifyEqual(mask.MaskType, 'Consensus Mask', 'Mask identity should be Consensus Mask.');
            testCase.verifyTrue(mask.Matrix(5,5,5), '(5,5,5) is shared by both, must be included.');
            testCase.verifyFalse(mask.Matrix(6,6,6), '(6,6,6) is only in AD, ratio 1.0 must exclude it.');
            
            % Lower ratio to 50%
            mask.computeConsensusMask(mockCohort, 0.5, 0.5);
            testCase.verifyTrue(mask.Matrix(6,6,6), 'With 50% ratio, (6,6,6) must now be included.');

            testCase.verifyCalled(behavior.getVolumes('ALL'));
        end
        
        function testComputeTpmMask(testCase)
            % PURPOSE: Verify TPM loading via SPM works with correct dimensions.

            % Check if SPM is installed before running the test
            if isempty(which('spm'))
                testCase.assumeFail('SPM not found in path. TPM loading test skipped.');
            end

            mask = BrainMask(testCase.RefInfo, testCase.TestLogger);
            mask.computeTpmMask(testCase.TpmPath, 0.5);
            
            testCase.verifyNotEmpty(mask.Matrix, 'The TPM matrix should not be empty.');
            testCase.verifyEqual(mask.MaskType, 'TPM Mask', 'Mask identity should be TPM Mask.');
        end
        
        function testComputeTpmMaskDimensionMismatch(testCase)
            % PURPOSE: Verify passing a wrong-dimension NIfTI throws spatial alignment error.
            
            if isempty(which('spm'))
                testCase.assumeFail('SPM not found in path. Spatial alignment test skipped.');
            end
            
            mask = BrainMask(testCase.RefInfo, testCase.TestLogger);
            testCase.verifyError(@() mask.computeTpmMask(testCase.TpmCorruptedPath, 0.5), ...
                'BrainMask:DimensionMismatch', ...
                'Should throw a spatial mismatch error.');
        end

        function testComputeTpmMaskAffineMismatch(testCase)
            % PURPOSE: Verify LBYL logic crashes on > 1e-4 affine deviation
            
            if isempty(which('spm'))
                testCase.assumeFail('SPM not found in path. Spatial alignment test skipped.');
            end
            
            % Intentionally corrupt the RefInfo in memory
            corruptedRef = testCase.RefInfo;
            corruptedMat = eye(4);
            corruptedMat(1,1) = corruptedMat(1,1) + 10; % Shift
            corruptedRef.NumericMatrix = corruptedMat;

            % Initialize mask with the corrupted memory
            mask = BrainMask(corruptedRef, testCase.TestLogger);

            % Loading the TPM file must now crash, because the 
            % mask will compare it to its corruptedRef
            testCase.verifyError(@() mask.computeTpmMask(testCase.TpmPath, 0.5), ...
                'BrainMask:AffineMismatch', ...
                'Should strictly crash if the Affine matrices are not aligned.');
        end

        function testComputeTpmMaskAffineTolerance(testCase)
            % PURPOSE: Verify sub-tolerance (<1e-4) deviations are ignored
            
            if isempty(which('spm'))
                testCase.assumeFail('SPM not found in path. Spatial alignment test skipped.');
            end
            
            % Intentionally corrupt the RefInfo in memory
            slightRef = testCase.RefInfo;
            slightMat = eye(4);
            slightMat(1,1) = slightMat(1,1) + 1e-5; % Safe sub-tolerance shift
            slightRef.NumeriMatrix = slightMat;
            
            mask = BrainMask(slightRef, testCase.TestLogger);

            try
                mask.computeTpmMask(testCase.TpmPath, 0.5);
                testCase.verifyEqual(mask.MaskType, 'TPM Mask', 'Mask identity should be TPM Mask.');
            catch ME
                testCase.verifyFail(sprintf('Failed on sub-tolerance affine deviation with error: %s', ME.identifier));
            end
        end
        
        function testComputeRidgwayMeanMask(testCase)
            % PURPOSE: Verify Ridgway optimizer calculates an optimal threshold T* on Mean Mask.
            mockInputs = {testCase.SandboxDir, 'dummy.csv', testCase.TestLogger};
            [mockCohort, behavior] = testCase.createMock(?CohortData, 'ConstructorInputs', mockInputs);

            % Create a distribution with variance to compute Pearson correlation
            fakeMean = rand(10, 10, 10, 'single');
            testCase.assignOutputsWhen(behavior.getMeanVolume('ALL'), fakeMean);

            mask = BrainMask(testCase.RefInfo, testCase.TestLogger);
            
            % Use large step (0.1) for a quick test
            mask.computeRidgwayMeanMask(mockCohort, 0.1);
            
            testCase.verifyEqual(mask.MaskType, 'Ridgway Mean Mask', 'Mask identity should be Ridgway Mean Mask.');
            testCase.verifyNotEmpty(mask.IntensityThreshold, 'Should have found an optimal IntensityThreshold.');
            testCase.verifyTrue(mask.IntensityThreshold > 0, 'The T* threshold must be > 0.');
            testCase.verifyEqual(size(mask.Matrix), [10 10 10], 'The calculated matrix must remain 3D.');
            testCase.verifyCalled(behavior.getMeanVolume('ALL'));
        end

        function testComputeRidgwayTpmMask(testCase)
            % PURPOSE: Verify Ridgway optimizer calculates an optimal threshold T* on TPM.
            
            if isempty(which('spm'))
                testCase.assumeFail('SPM not found in path. Ridgway TPM test skipped.');
            end
            
            mask = BrainMask(testCase.RefInfo, testCase.TestLogger);

            % Use large step (0.1) for a quick test
            mask.computeRidgwayTpmMask(testCase.TpmPath, 0.1);
            
            testCase.verifyEqual(mask.MaskType, 'Ridgway TPM Mask', 'Mask identity should be Ridgway TPM Mask.');
            testCase.verifyNotEmpty(mask.IntensityThreshold, 'Should have found an optimal IntensityThreshold.');
            testCase.verifyTrue(mask.IntensityThreshold > 0, 'The T* threshold must be > 0.');
            testCase.verifyEqual(size(mask.Matrix), [10 10 10], 'The calculated matrix must remain 3D.');
        end
        
        function testEvaluateLeakage(testCase)
            % PURPOSE: Verify leakage computation.
            mockInputs = {testCase.SandboxDir, 'dummy.csv', testCase.TestLogger};
            [mockCohort, behavior] = testCase.createMock(?CohortData, 'ConstructorInputs', mockInputs);

            % Simulate a 4D empty cohort tensor to force extreme leak evaluation
            fakeVols = zeros(10, 10, 10, 2, 'single');
            testCase.assignOutputsWhen(behavior.getVolumes('ALL'), fakeVols);

            mask = BrainMask(testCase.RefInfo, testCase.TestLogger);
            
            % Force mask to be all true to create an extreme leakage
            mask.Matrix = true(10, 10, 10);
            
            % Dummy volumes are mostly 0, so leakage will be very high
            leakageArray = mask.evaluateLeakage(mockCohort, 'ALL');
            
            testCase.verifyTrue(mean(leakageArray) > 90, 'With a full mask, mean leakage must be > 90%.');
            testCase.verifyEqual(length(leakageArray), 2, 'Should return a column vector for N=2 patients.');
            testCase.verifyCalled(behavior.getVolumes('ALL'));
        end
        
        function testShowMaskStats(testCase)
            % PURPOSE: Verify execution of active voxels calculation
            mask = BrainMask(testCase.RefInfo, testCase.TestLogger);
            
            % Create a dummy mask with some active voxels
            fakeMat = false(10,10,10);
            fakeMat(1:200) = true;
            mask.Matrix = fakeMat;
            mask.MaskType = 'Dummy Mask';
            
            try
                mask.showMaskStats();
                testCase.verifyTrue(true, 'showMaskStats executed successfully.');
            catch ME
                testCase.verifyFail(sprintf('showMaskStats crashed: %s', ME.message));
            end
        end
        
        function testExportToNifti(testCase)
            % PURPOSE: Verify exportToNifti creates missing folder.
            mask = BrainMask(testCase.RefInfo, testCase.TestLogger);
            mask.MaskType = 'Test Mask'; % Inject mask identity
            mask.Matrix = true(10, 10, 10);
            
            % Point to a non-existent folder inside the sandbox
            outPath = fullfile(testCase.SandboxDir, 'DummyDir', 'exported_mask.nii');
            [folderPath, ~, ~] = fileparts(outPath);

            % Ensure folder does not exist before test
            testCase.verifyFalse(exist(folderPath, 'dir') == 7);
            
            % Call the method. If it crashes, the test fails.
            mask.exportToNifti(outPath);
            
            % Verify that folder and file were created
            testCase.verifyTrue(exist(folderPath, 'dir') == 7, 'The missing folder must exist.');
            testCase.verifyTrue(exist(outPath, 'file') == 2, 'The NIfTI file must exist.');
        end
    end
end