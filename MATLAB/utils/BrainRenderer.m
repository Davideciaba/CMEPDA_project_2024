classdef BrainRenderer < handle
    % BRAINRENDERER Handles graphics visualization and exporting.
    %
    % PURPOSE: Takes pre-computed mathematical matrices (masks, T-maps, leakages) 
    %   and renders them into visual figures, handling colormaps, layout, 
    %   and file exporting.
    
    properties (Access = private)
        PrivateLogger Logger    % Custom Logger
    end
    
    methods (Access = public)
        function obj = BrainRenderer(loggerObj)
            % CONSTRUCTOR: Initializes the BrainRenderer object
            arguments
                loggerObj (1,1) Logger = Logger('NullLogger')
            end
            obj.PrivateLogger = loggerObj;
        end
        
        function plotLeakageHistograms(obj, leakMatAD, leakMatCTRL, methodLabels, groupColors, outFigPath)
            % METHOD: plotGroupLeakageHistograms
            % PURPOSE: Plots leakage arrays (obtained using the evaluateLeakage
            %    method of the BrainMask class) as histograms grouped by 
            %    treatment (AD vs CTRL) and by masking method
            arguments
                obj BrainRenderer
                leakMatAD (:, :) double {mustBeNonempty}
                leakMatCTRL (:, :) double {mustBeNonempty}
                methodLabels (1, :) string {mustBeNonempty}
                groupColors (1, 2) string = ["r", "b"] 
                outFigPath (1,:) char = ''
            end
            
            numMethods = numel(methodLabels);
            
            % Ensure the number of columns in the data matrices matches the provided labels.
            if size(leakMatAD, 2) ~= numMethods || size(leakMatCTRL, 2) ~= numMethods
                obj.PrivateLogger.error('plotLeakageHistograms: Matrix columns do not match number of labels.');
                error('BrainRenderer:DimensionMismatch', 'Number of columns must match the methodLabels array.');
            end
            
            % Find the absolute global maximum for uniform axes
            globalMaxLeakage = max(1.0, max(max(leakMatAD, [], 'all'), max(leakMatCTRL, [], 'all'))); 
            
            numBins = 30; 
            binEdges = linspace(0, globalMaxLeakage, numBins);

            % Find the maximum frequency count to unify the Y-axis across subplots
            globalMaxOccurrences = 0;
            for i = 1:numMethods
                countsAD = histcounts(leakMatAD(:, i), binEdges);
                countsCTRL = histcounts(leakMatCTRL(:, i), binEdges);
                globalMaxOccurrences = max([globalMaxOccurrences, max(countsAD), max(countsCTRL)]);
            end
            
            % Define the limits
            unifiedYLimit = [0, globalMaxOccurrences];
            unifiedXLimit = [0, globalMaxLeakage];
            
            % Square representation
            numCols = ceil(sqrt(numMethods));
            numRows = ceil(numMethods / numCols);
            
            figHandle = figure('Name', 'AD vs CTRL Leakage', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
            
            for idx = 1:numMethods
                subplot(numRows, numCols, idx); hold on;
                dataAD = leakMatAD(:, idx);
                dataCTRL = leakMatCTRL(:, idx);
                
                % Plot filled semi-transparent histograms
                histogram(dataAD, binEdges, 'FaceColor', groupColors(1), 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'AD');
                histogram(dataCTRL, binEdges, 'FaceColor', groupColors(2), 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'CTRL');
                
                % Add solid stepped outlines for clarity
                histogram(dataAD, binEdges, 'DisplayStyle', 'stairs', 'EdgeColor', groupColors(1), 'LineWidth', 1.5, 'HandleVisibility', 'off');
                histogram(dataCTRL, binEdges, 'DisplayStyle', 'stairs', 'EdgeColor', groupColors(2), 'LineWidth', 1.5, 'HandleVisibility', 'off');
                
                title(methodLabels(idx), 'Interpreter', 'none', 'FontSize', 12, 'FontWeight', 'bold');
                xlabel('Data Leakage (%)', 'FontSize', 10); ylabel('Occurrences', 'FontSize', 10);
                xlim(unifiedXLimit); ylim(unifiedYLimit); grid on;
                legend('Location', 'northeast');
                hold off;
            end
            
            obj.saveFigure(figHandle, outFigPath);
        end
        
        function plotMaskOverlays(obj, baseSlices, baseTitles, maskSlices, maskColors, maskLabels, outFigPath)
            % METHOD: plotMaskOverlays
            % PURPOSE: Plots raw 3D volumes slices and masks (different contour
            %  colors represents different methods) over those slices.
            arguments
                obj BrainRenderer
                baseSlices (:,:,:) double {mustBeNonempty}
                baseTitles (1,:) string {mustBeNonempty}
                maskSlices (:,:,:) logical {mustBeNonempty}
                maskColors (1,:) string {mustBeNonempty}
                maskLabels (1,:) string {mustBeNonempty}
                outFigPath (1,:) char = ''
            end
            
            numBases = size(baseSlices, 3);
            numMasks = size(maskSlices, 3);
            
            % Ensure arrays lengths match the tensor depths
            if numel(baseTitles) ~= numBases || numel(maskLabels) ~= numMasks || numel(maskColors) ~= numMasks
                obj.PrivateLogger.error('plotMaskOverlays: Tensor depth does not match labels length.');
                error('BrainRenderer:LabelMismatch', 'Labels length must match the 3D tensor depth.');
            end
            if size(baseSlices, 1) ~= size(maskSlices, 1) || size(baseSlices, 2) ~= size(maskSlices, 2)
                obj.PrivateLogger.error('plotMaskOverlays: Base and Mask spatial dimensions mismatch.');
                error('BrainRenderer:SpatialMismatch', 'Base images and masks must have identical X-Y dimensions.');
            end
            
            numRows = numMasks + 1;
            globalColorLimits = [0, max(baseSlices, [], 'all')];
            
            figHandle = figure('Name', 'Multi Mask Overlays', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
            
            % Top row: Raw volume slices without any overlays
            for colIdx = 1:numBases
                subplot(numRows, numBases, colIdx);
                imagesc(baseSlices(:,:,colIdx));
                clim(globalColorLimits);
                axis image off; colormap(gca, 'gray');
                title(baseTitles(colIdx), 'Interpreter', 'none');
            end
            
            % Subsequent rows: Overlays for each specific mask type
            for maskIdx = 1:numMasks
                currentMask = maskSlices(:,:,maskIdx);
                currentColor = maskColors(maskIdx);
                currentLabel = maskLabels(maskIdx);
                
                for colIdx = 1:numBases
                    plotIdx = (maskIdx * numBases) + colIdx;
                    subplot(numRows, numBases, plotIdx);

                    % Plot the raw volume slices
                    imagesc(baseSlices(:,:,colIdx));
                    clim(globalColorLimits); 
                    axis image off; colormap(gca, 'gray');
                    
                    % Plot the binary mask as a colored outline (contour)
                    hold on;
                    contour(currentMask, [1,1], char(currentColor), 'LineWidth', 1.5);
                    hold off;
                    
                    fullTitle = sprintf('%s (%s: %s)', baseTitles(colIdx), upper(currentColor), currentLabel);
                    title(fullTitle, 'Interpreter', 'none');
                end
            end
            obj.saveFigure(figHandle, outFigPath);
        end

        function plotMaskHistograms(obj, baseVols4D, baseTitles, maskVols4D, maskColors, maskLabels, binStep, thresholdValue, outFigPath)
            % METHOD: plotMaskedHistograms
            % PURPOSE: Plots voxel intensity distributions of raw 3D volumes and 
            %   shows how different masks selectively filter those distributions. 
            arguments
                obj BrainRenderer
                baseVols4D (:,:,:,:) double {mustBeNonempty}
                baseTitles (1,:) string {mustBeNonempty}
                maskVols4D (:,:,:,:) logical {mustBeNonempty}
                maskColors (1,:) string {mustBeNonempty}
                maskLabels (1,:) string {mustBeNonempty}
                binStep (1,1) double {mustBePositive} = 0.01
                thresholdValue (1,1) double {mustBePositive} = 0.01
                outFigPath (1,:) char = ''
            end
            
            numBases = size(baseVols4D, 4);
            numMasks = size(maskVols4D, 4);
            
            % Ensure arrays lengths match the tensor depths and the spatial
            % dimensions of the masks match the raw volume
            if numBases ~= numel(baseTitles)
                obj.PrivateLogger.error('plotMaskedHistograms: baseTitles length does not match base volumes depth.');
                error('BrainRenderer:LabelMismatch', 'baseTitles array length must match the number of raw volumes.');
            end
            if size(baseVols4D, 1) ~= size(maskVols4D, 1) || size(baseVols4D, 2) ~= size(maskVols4D, 2) || size(baseVols4D, 3) ~= size(maskVols4D, 3)
                obj.PrivateLogger.error('plotMaskedHistograms: Base and Mask spatial dimensions mismatch.');
                error('BrainRenderer:SpatialMismatch', 'Base images and masks must have identical 3D dimensions.');
            end
            if numel(maskLabels) ~= numMasks || numel(maskColors) ~= numMasks
                obj.PrivateLogger.error('plotMaskedHistograms: Colors and Labels arrays do not match mask depth.');
                error('BrainRenderer:LabelMismatch', 'Colors and Labels must match mask tensor depth.');
            end
            
            % Bin computation
            globalMax = max(baseVols4D(:));
            if globalMax == 0
                globalMax = 1; % Safely fallback if volumes are entirely empty
            end
            bins = 0 : binStep : globalMax;
           
            numCols = ceil(sqrt(numBases));
            numRows = ceil(numBases / numCols);
            
            figHandle = figure('Name', 'Voxel Intensity Histograms', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
            
            for imgIdx = 1:numBases
                subplot(numRows, numCols, imgIdx); hold on;
                
                currentImg = baseVols4D(:,:,:,imgIdx);
                currentTitle = baseTitles(imgIdx);
                
                % Unmasked distribution
                histogram(currentImg, bins, 'Normalization', 'pdf', 'DisplayName', sprintf('%s Global Dist.', currentTitle));
                
                % Iteratively apply each mask via logical indexing to extract remaining voxels
                for maskIdx = 1:numMasks
                    currentMask = maskVols4D(:,:,:,maskIdx);
                    maskedData = currentImg(currentMask);
                    
                    histogram(maskedData, bins, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', ...
                              'EdgeColor', char(maskColors(maskIdx)), 'LineWidth', 2, 'DisplayName', sprintf('%s', maskLabels(maskIdx)));
                end
                
                % Reference threshold line
                xline(thresholdValue, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Threshold = %.3f', thresholdValue));
                xlabel('Voxel Intensity'); ylabel('Probability Density');
                title(sprintf('%s Intensity', currentTitle), 'Interpreter', 'none');
                legend('Location', 'northeast', 'Interpreter', 'none');
                grid on; yscale('log'); hold off;
            end
            obj.saveFigure(figHandle, outFigPath);
        end


        function plotCompositeOverlap(obj, mapA, mapB, bgVolume, affineMat, sliceConfig, nameA, nameB, outFigPath)
            % METHOD: plotCompositeOverlap
            % PURPOSE: Fuses two distinct 3D maps into a single RGB volume to evaluate
            %   their spatial overlap
            arguments
                obj BrainRenderer
                mapA (:,:,:) double {mustBeNonempty}
                mapB (:,:,:) double {mustBeNonempty}
                bgVolume (:,:,:) double {mustBeNonempty}
                affineMat (4,4) double {mustBeNonempty}
                % MODE: Scalar (Auto), 1x3 (Manual MNI), or 1xN (Explicit MNI)
                sliceConfig double {mustBeVector, mustBeReal, mustBeFinite, mustBeNonempty}
                nameA (1,:) char = 'Map A'
                nameB (1,:) char = 'Map B'
                outFigPath (1,:) char = ''
            end
            
            % Ensure the spatial dimensions of the maps match the raw volume
            if ~isequal(size(mapA), size(mapB), size(bgVolume))
                obj.PrivateLogger.error('plotCompositeOverlap: Maps have different spatial dimensions.');
                error('BrainRenderer:DimensionMismatch', 'All input 3D maps must have identical spatial dimensions.');
            end

            % Failsafe to prevent division by zero in case of completely black background
            bgMax = max(bgVolume(:));
            if bgMax == 0, bgMax = 1; end
            
            globalMask = (mapA > 0) | (mapB > 0);
            if ~any(globalMask(:))
                obj.PrivateLogger.warning('Both maps are entirely empty. No overlap to visualize.');
                return;
            end

            % Returns both internal voxel matrix indices and true physical MNI Z coordinates.
            [zSlicesVoxel, zMmArray] = obj.getVoxelIndicesFromMni(sliceConfig, affineMat, size(bgVolume), globalMask);
            
            % Square representation
            numSlices = length(zSlicesVoxel);
            gridCols = ceil(sqrt(numSlices));
            gridRows = ceil(numSlices / gridCols);
            
            figHandle = figure('Name', 'RGB Composite Overlap', 'Color', 'k', 'Position', [100, 100, gridCols*250, gridRows*250]);
            
            for idx = 1:numSlices
                zVoxel = zSlicesVoxel(idx);
                zMm = zMmArray(idx);
                
                % Standard orientation correction
                sliceBg = rot90(bgVolume(:, :, zVoxel));
                sliceA  = rot90(mapA(:, :, zVoxel));
                sliceB  = rot90(mapB(:, :, zVoxel));
                
                % Convert grayscale background into a 3-channel RGB image
                bgNorm = mat2gray(sliceBg, [0, bgMax]);
                imgRGB = cat(3, bgNorm, bgNorm, bgNorm);
                
                maskA = sliceA > 0; maskB = sliceB > 0; combinedMask = maskA | maskB;
                
                % Scale map intensities for RGB injection logic
                normA = obj.scaleMap(sliceA, maskA);
                normB = obj.scaleMap(sliceB, maskB);
                
                % MapA goes to Red channel, MapB goes to Green channel. 
                % Overlaps naturally yield Yellow
                overlayRGB = cat(3, normA, normB, zeros(size(sliceA)));

                % Define transparency
                alphaLayer = double(combinedMask) * 0.75;
                finalRGB = imgRGB .* (1 - alphaLayer) + overlayRGB .* alphaLayer;
                
                ax = subplot(gridRows, gridCols, idx);
                imshow(finalRGB, 'Parent', ax, 'InitialMagnification', 'fit');
                title(ax, sprintf('Z = %.1f mm', zMm), 'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
            end
            annotation('textbox', [0.01 0.01 0.98 0.05], 'String', sprintf(' RED: %s | GREEN: %s | YELLOW: Intersection', nameA, nameB), ...
                'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'BackgroundColor', [0.2 0.2 0.2]);
            
            obj.saveFigure(figHandle, outFigPath);
        end
        
        function plotStatisticalOverlay(obj, thresholdedMap, statThresh, bgVolume, affineMat, sliceConfig, contrastName, correctionMode, pValue, mapName, outFigPath)
            % METHOD: plotStatisticalOverlay
            % PURPOSE: RPlots thresholded statistical T-maps (using 'hot' colormap)
            %   over 3D raw volume with proper trasparency.
            arguments
                obj BrainRenderer
                thresholdedMap (:,:,:) double {mustBeNonempty}
                statThresh (1,1) double {mustBeNonempty}
                bgVolume (:,:,:) double {mustBeNonempty}
                affineMat (4,4) double {mustBeNonempty}
                % MODE: Scalar (Auto), 1x3 (Manual MNI), or 1xN (Explicit MNI)
                sliceConfig double {mustBeVector, mustBeReal, mustBeFinite, mustBeNonempty}
                contrastName (1, :) char {mustBeNonempty}
                correctionMode (1, :) char {mustBeNonempty}
                pValue (1, 1) double {mustBePositive, mustBeLessThan(pValue, 1)}
                mapName (1,:) char {mustBeNonempty}
                outFigPath (1,:) char = ''
            end
            
            % Ensure the spatial dimensions of the map match the raw volume
            if ~isequal(size(thresholdedMap), size(bgVolume))
                obj.PrivateLogger.error('plotStatisticalOverlay: Statistical map and raw volume dimensions mismatch.');
                error('BrainRenderer:DimensionMismatch', 'Statistical map and raw volume must be identical.');
            end

            % T-stats Upper Limit
            maxTValue = max(thresholdedMap(:));
            if maxTValue <= statThresh
                obj.PrivateLogger.warning('Zero voxels survived the applied threshold %.2f. Plotting aborted.', statThresh);
                return;
            end
            
            activeMask = thresholdedMap >= statThresh;
            
            % Returns both internal voxel matrix indices and true physical MNI Z coordinates.
            [zSlicesVoxel, zMmArray] = obj.getVoxelIndicesFromMni(sliceConfig, affineMat, size(bgVolume), activeMask);
            
            numSlices = length(zSlicesVoxel);
            totalPanels = numSlices + 1; % +1 to reserve space for the Colorbar
            gridCols = ceil(sqrt(totalPanels));
            gridRows = ceil(totalPanels / gridCols);
            
            hFig = figure('Name', 'VBM Analysis Overlay', 'Color', 'k', 'Position', [100, 100, gridCols*250, gridRows*250]);
            
            for idx = 1:numSlices
                zVoxel = zSlicesVoxel(idx);
                zMm = zMmArray(idx);
                
                sliceAnatomy = rot90(bgVolume(:, :, zVoxel));
                sliceStats = rot90(thresholdedMap(:, :, zVoxel));
                
                alphaMask = double(sliceStats >= statThresh) * 0.75;
                
                % Plot raw volume layer
                axAnatomy = subplot(gridRows, gridCols, idx);
                imagesc(axAnatomy, sliceAnatomy); colormap(axAnatomy, 'gray'); axis(axAnatomy, 'image', 'off'); 
                
                % Superimpose statistical layer transparently
                axStats = axes('Position', axAnatomy.Position);
                hImageStat = imagesc(axStats, sliceStats, [statThresh, maxTValue]);
                colormap(axStats, 'hot'); axis(axStats, 'image', 'off');
                
                set(axStats, 'Color', 'none', 'Visible', 'off');
                set(hImageStat, 'AlphaData', alphaMask);
                linkaxes([axAnatomy, axStats]);
                title(axAnatomy, sprintf('Z = %.1f mm', zMm), 'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
            end
            
            % Dedicate the final subplot to the colorbar
            axColorbar = subplot(gridRows, gridCols, totalPanels);
            axis(axColorbar, 'off'); 
            colormap(axColorbar, 'hot'); clim(axColorbar, [statThresh, maxTValue]);
            cb = colorbar(axColorbar, 'Location', 'west');
            cb.Color = 'w'; cb.Label.String = 't-value'; cb.Label.FontWeight = 'bold';

            annotation('textbox', [0.01 0.01 0.98 0.05], 'String', sprintf(' %s | %s | %s at alpha = %.2f', contrastName, mapName, correctionMode, pValue), ...
                'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'BackgroundColor', [0.2 0.2 0.2]);
            
            obj.saveFigure(hFig, outFigPath);
        end
    end
    
    methods (Access = private)
        % HELPER: getVoxelIndicesFromMni
        function [zSlicesVoxel, zMmArray] = getVoxelIndicesFromMni(obj, sliceConfig, affineMat, tensorSize, activeMask)
            % PURPOSE: Converts MNI millimeters to matrix voxel indices. 
            %   Returns both the sorted voxel indices and their true physical
            %   MNI Z coordinates.
            
            voxCenterXY = [round(tensorSize(1)/2); round(tensorSize(2)/2)];
            maxZ = tensorSize(3);
            
            if isscalar(sliceConfig)
                % MODE 1: AUTO-DATADRIVEN (Scalar defines the step in mm)
                stepMm = sliceConfig;
                activeSlices = squeeze(sum(activeMask, [1, 2]));
                activeIdx = find(activeSlices > 0);

                % Matrix multiplication to extract MNI boundaries
                voxBounds = [voxCenterXY(1), voxCenterXY(1);
                             voxCenterXY(2), voxCenterXY(2);
                             activeIdx(1),   activeIdx(end);
                             1,              1];
            
                mniBounds = affineMat * voxBounds;
                    
                mniMin = min(mniBounds(3, :));
                mniMax = max(mniBounds(3, :));
                    
                % Pad the viewing box by 1 step before and after the active region
                mniMin = mniMin - stepMm;
                mniMax = mniMax + stepMm;
                
                % Fix the min to a multiple of the step (relative to Z=0)
                alignedMin = floor(mniMin / stepMm) * stepMm;

                mniArray = alignedMin : stepMm : (mniMax + stepMm * 0.1);
                
            elseif numel(sliceConfig) == 3
                % MODE 2: MANUAL BOUNDS [start_mm, step_mm, stop_mm]
                startMm = sliceConfig(1);
                stepMm  = sliceConfig(2);
                stopMm  = sliceConfig(3);
                
                % Fallback protection: handle inverted manual step directions
                if startMm < stopMm && stepMm < 0
                    stepMm = -stepMm;
                elseif startMm > stopMm && stepMm > 0
                    stepMm = -stepMm;
                end
                
                mniArray = startMm : stepMm : stopMm;
                
            else
                % MODE 3: EXPLICIT MNI ARRAY (e.g. -40:5:20)
                mniArray = sliceConfig;
            end
            

            % Convert the generated MNI coordinates back into matrix voxel indices
            if ~isempty(mniArray)
                numMniSlices = length(mniArray);
                mniSlicesMat = [zeros(1, numMniSlices); ...
                               zeros(1, numMniSlices); ...
                               mniArray(:)'; ... % Ensure row vector
                               ones(1, numMniSlices)];
                
                voxCoordsMat = affineMat  \ mniSlicesMat;
                zSlicesVoxel = round(voxCoordsMat(3, :));
            else
                obj.PrivateLogger.error('No valid MNI cordinates found for the slice configuration delivered. Aborting rendering.');
                error('BrainRenderer:NoValidSlices', 'No valid MNI coordinates found.');
            end
            
            % Remove any indices outside the physical matrix volume
            validLogical = zSlicesVoxel >= 1 & zSlicesVoxel <= maxZ;
            
            % Unique automatically sorts the voxels ascending
            zSlicesVoxel = unique(zSlicesVoxel(validLogical));
            
            % Recalculate the physical MNI dimension for the validated and sorted voxels.
            numSlices = length(zSlicesVoxel);
            if numSlices > 0
                trueSlicesMat = [repmat(voxCenterXY(1), 1, numSlices); ...
                                repmat(voxCenterXY(2), 1, numSlices); ...
                                zSlicesVoxel; ...
                                ones(1, numSlices)];
                                
                trueMniCoords = affineMat * trueSlicesMat;
                zMmArray = trueMniCoords(3, :);
            else
                obj.PrivateLogger.error('No valid voxel indices found after MNI to voxel conversion. Aborting rendering.');
                error('BrainRenderer:NoValidSlices', 'No valid voxel indices found after MNI to voxel conversion.');
            end
        end

        function saveFigure(obj, figHandle, outPath)
            % HELPER: saveFigure
            % PURPOSE: Attempts to save a figure safely.
            
            % Ensure the figure is closed when the function ends
            cleanupObj = onCleanup(@() close(figHandle));

            % If outPath is empty or '', should only show the figure
            if isempty(outPath) || strlength(outPath) == 0
                return;
            else
                obj.PrivateLogger.info('Attempting to save figure to: %s', outPath);
                try
                    % Safely create parent directory
                    [outDir, ~, ~] = fileparts(outPath);
                    if ~isempty(outDir) && ~exist(outDir, 'dir')
                        mkdir(outDir);
                    end
                    exportgraphics(figHandle, outPath, 'Resolution', 300, 'BackgroundColor', figHandle.Color);
                    obj.PrivateLogger.success('Figure saved successfully.');
                catch ME
                    obj.PrivateLogger.error('Failed to save figure fallback: %s', ME.identifier);
                    rethrow(ME);
                end
            end
        end
        
        function normData = scaleMap(~, sliceData, maskData)
            % HELPER: normData
            % PURPOSE: Scales continuous variables between 0.0 and 1.0 for RGB injection.
            
            % If the mask has no active voxels, return an empty slice
            if ~any(maskData, 'all')
                normData = zeros(size(sliceData));
                return;
            end
            
            activeVoxels = sliceData(maskData);
            minV = min(activeVoxels); 
            maxV = max(activeVoxels);
            
            if minV == maxV
                normData = double(maskData); % Binary Mask
            else
                
                normData = mat2gray(sliceData, [minV, maxV]); % Continuous map
                
                % Background safety constraint
                normData(~maskData) = 0;
            end
        end
    end
end