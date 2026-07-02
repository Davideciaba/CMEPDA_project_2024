classdef TestBrainRenderer < matlab.unittest.TestCase
    % TESTBRAINRENDERER Automated Unit Test suite for the BrainRenderer class.
    %
    % PURPOSE: Implements an automated Sandbox. It does not test the aesthetic output of 
    %   the figures. Instead, it tests dimension mismatch, argument
    %   validation and fallbacks.
    
    properties
        SandboxDir char         % The Sandbox Directory
        TestLogger Logger       % Custom Logger
        DummyBg double          % Dummy raw 3D volume
        DummyMapA double        % Dummy statistical map A
        DummyMapB double        % Dummy statistical map B
        DummyAffine double      % Dummy affine matrix
    end
    
    methods (TestMethodSetup)
        function createSandbox(testCase)
            % Create temporary directory
            testCase.SandboxDir = tempname;
            mkdir(testCase.SandboxDir);
            
            % Suppress console spam during tests
            testCase.TestLogger = Logger('NullLogger'); 
            
            % Generate dummy 3D tensors (10x10x10)
            testCase.DummyBg = rand(10, 10, 10);
            
            testCase.DummyMapA = zeros(10, 10, 10);
            testCase.DummyMapA(5,5,5) = 3.5; % One active voxel
            
            testCase.DummyMapB = zeros(10, 10, 10);
            testCase.DummyMapB(6,6,6) = 4.2; % One active voxel
            
            % Dummy affine matrix (identity with some translation)
            testCase.DummyAffine = [2 0 0 -10; 
                                    0 2 0 -10; 
                                    0 0 2 -10; 
                                    0 0 0   1];
        end
    end
    
    methods (TestMethodTeardown)
        function destroySandbox(testCase)
            % Clean up the temporary folder entirely regardless of test result
            if exist(testCase.SandboxDir, 'dir')
                rmdir(testCase.SandboxDir, 's');
            end
            % Force close any figures generated during tests
            close all force; 
        end
    end
    
    methods (Test)
        
        function testDimensionMismatch(testCase)
            % PURPOSE: Verify LBYL logic blocks mismatched tensors.
            renderer = BrainRenderer(testCase.TestLogger);
            
            % Corrupt Map B intentionally
            badMapB = zeros(5, 5, 5); 
            
            testCase.verifyError(@() renderer.plotCompositeOverlap( ...
                testCase.DummyMapA, badMapB, testCase.DummyBg, ...
                testCase.DummyAffine, 2, 'A', 'B', ''), ...
                'BrainRenderer:DimensionMismatch', ...
                'Must throw DimensionMismatch if composite maps do not align.');

            testCase.verifyError(@() renderer.plotStatisticalOverlay( ...
                badMapB, 3.0, testCase.DummyBg, ...
                testCase.DummyAffine, 2, 'Contrast', 'FWE', 0.05, 'Dummy Map', ''), ...
                'BrainRenderer:DimensionMismatch', ...
                'Must throw DimensionMismatch if statistical map and raw volume do not align.');
        end
        
        function testSliceConfig(testCase)
            % PURPOSE: Verify arguments block rejects multi-dim matrices
            %   for sliceConfig
            renderer = BrainRenderer(testCase.TestLogger);
            
            invalidSliceConfig = [1 2; 3 4]; % 2D Matrix
            
            testCase.verifyError(@() renderer.plotStatisticalOverlay( ...
                testCase.DummyMapA, 3.0, testCase.DummyBg, ...
                testCase.DummyAffine, invalidSliceConfig, 'Contrast',...
                'FWE', 0.05, 'Dummy Map', ''), ...
                'MATLAB:validators:mustBeVector', ...
                'mustBeVector should reject 2D matrices.');
        end

        function testLeakageHistograms(testCase)
            % PURPOSE: Verify LBYL logic blocks misaligned arrays.
            
            renderer = BrainRenderer(testCase.TestLogger);
            
            % Create dummy leakage data (10 subjects, 2 methods)
            leakAD = rand(10, 2);
            leakCTRL = rand(10, 2);
            
            % Pass 3 labels for only 2 methods
            badLabels = ["Method1", "Method2", "GhostMethod"]; 
            
            testCase.verifyError(@() renderer.plotLeakageHistograms( ...
                leakAD, leakCTRL, badLabels, ["r", "b"], ''), ...
                'BrainRenderer:DimensionMismatch', ...
                'Should fail if the number of data columns does not match the number of labels.');
        end

        function testMaskOverlays(testCase)
            % PURPOSE: Verify LBYL logic blocks misaligned arrays.
            
            renderer = BrainRenderer(testCase.TestLogger);
            
            dummyBases = rand(10, 10, 2); % 2 Base Slices
            dummyMask = true(10, 10, 1); % 1 Mask Slice
            
            % Label length mismatch
            badTitles = ["Slice 1", "Slice 2", "Slice 3"]; % 3 Titles for 2 Slices
            
            testCase.verifyError(@() renderer.plotMaskOverlays( ...
                dummyBases, badTitles, dummyMask, "r", "Mask 1", ''), ...
                'BrainRenderer:LabelMismatch', ...
                'Should fail if base titles length differs from base tensor depth.');
                
            % Spatial mismatch
            badSpatialBases = rand(20, 20, 2); % Background is 20x20, Mask is 10x10
            
            testCase.verifyError(@() renderer.plotMaskOverlays( ...
                badSpatialBases, ["Slice 1", "Slice 2"], dummyMask, "r", "Mask 1", ''), ...
                'BrainRenderer:SpatialMismatch', ...
                'Should fail if raw volume and mask have different X-Y dimensions.');
        end

        function testMaskHistograms(testCase)
            % PURPOSE: Verify LBYL logic blocks misaligned arrays.
            
            renderer = BrainRenderer(testCase.TestLogger);
            
            dummyBase4D = rand(10, 10, 10, 1); % 1 Raw volume
            dummyMask4D = true(10, 10, 10, 2); % 2 Masks
            
            % Provide only 1 color for 2 masks
            badColors = "r"; 
            
            testCase.verifyError(@() renderer.plotMaskHistograms( ...
                dummyBase4D, "Base 1", dummyMask4D, badColors, ["Mask 1", "Mask 2"], 0.01, 0.01, ''), ...
                'BrainRenderer:LabelMismatch', ...
                'Should fail if colors length does not match mask tensor depth.');
        end

        function testStabilityEmptyMap(testCase)
            % PURPOSE: Ensure stability when passing an empty statistical map.
            renderer = BrainRenderer(testCase.TestLogger);
            
            emptyMap = zeros(10, 10, 10); % Zero active voxels
            
            % If the fallbacks in scaleMap and getVoxelIndicesFromMni are broken,
            % this will throw a generic MATLAB error
            try
                renderer.plotCompositeOverlap( ...
                    emptyMap, emptyMap, testCase.DummyBg, ...
                    testCase.DummyAffine, 2, 'A', 'B', '');
                % If it gets here without crashing, the test passes.
                testCase.verifyTrue(true); 
            catch ME
                testCase.verifyFail(sprintf('Failed on empty map with error: %s', ME.identifier));
            end
        end
        
        function testFallbackBlackBackground(testCase)
            % PURPOSE: Test the specific if bgMax == 0, bgMax = 1; fallback.
            renderer = BrainRenderer(testCase.TestLogger);
            
            blackBg = zeros(10, 10, 10); % Corrupted raw volume
            
            try
                renderer.plotStatisticalOverlay( ...
                    testCase.DummyMapA, 3.0, blackBg, ...
                    testCase.DummyAffine, 2, 'Contrast', 'FWE', 0.05, 'Dummy Map', '');
                testCase.verifyTrue(true); 
            catch ME
                testCase.verifyFail(sprintf('Failed on pure black background with error: %s', ME.identifier));
            end
        end
        
        function testSaveFigure(testCase)
            % PURPOSE: Verify saveFigure creates missing folder.
            renderer = BrainRenderer(testCase.TestLogger);
            
            % Point to a non-existent folder inside the sandbox
            deepOutPath = fullfile(testCase.SandboxDir, 'DummyDir', 'plot.png');
            [folderPath, ~, ~] = fileparts(deepOutPath);
            
            % Ensure folder does NOT exist before test
            testCase.verifyFalse(exist(folderPath, 'dir') == 7);
            
            % Call the method. If it crashes, the test fails.
            renderer.plotStatisticalOverlay( ...
                testCase.DummyMapA, 3.0, testCase.DummyBg, ...
                testCase.DummyAffine, 2, 'Contrast', 'FWE', 0.05, 'Dummy Map', deepOutPath);
                
            % Verify that folder and file were created
            testCase.verifyTrue(exist(folderPath, 'dir') == 7, 'The missing folder must exist.');
            testCase.verifyTrue(exist(deepOutPath, 'file') == 2, 'The image must exist.');
        end  
    end
end