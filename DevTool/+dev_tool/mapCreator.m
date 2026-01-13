function mapCreator()
    % MAPCREATOR Simple tool to create maps of landmarks and write them into a scenario file.
    %
    % Syntax:
    %   dev_tool.mapCreator();
    %
    % Description:
    %   This script creates a landmark map which can be used in the map.
    %   The input is a landmark region (LMR). The LMR specifies a set of
    %   bounding boxes (specified by min and max coordinates) and the
    %   number of landmarks within that rectangle. Landmarks are randomly
    %   selected within each bounding box. Multiple ones can be used
    %   create more complicated maps. See the file example_lmr.json in the
    %   config subdirectory for an example of the usage.
    % 
    %   Running the function will open a dialogue box with the filter
    %   '*_lmr.json'. Once you select the file, a figure will open and show
    %   you the map generated. A second dialogue box will open to query you
    %   for where to save the file. By default it is in the same directory
    %   as the lmr file, but with '_lmr' removed from the name.
    %
    %   To use the output map, you will need to use one of the activity /
    %   configuration files and change the field "scenario" to point to
    %   this file.
    %
    % Note:
    %   This script does not write a waypoints file or set the figure to
    %   the right size to draw the map.

    import ebe.graphics.FigureManager;
    
    % Previous inputs and outputs
    persistent lmrConfigFile;

    % Use the MATLAB GUI to select the file to read in; the slightly
    % convoluted logic here creates or re-uses previous settings
    if (exist('lmrConfigFile', 'var') == false)
        lmrConfigFile = ebe.utils.getScriptPath();
    end
    
    [lmrFileName,newLMRFileLocation] = uigetfile('*_lmr.json', ...
        'Landmark region file', lmrConfigFile);
    
    if (lmrFileName == 0)
        disp('Aborted')
        return
    end
    
    % The absolute location of the configuration file
    lmrConfigFile = fullfile(newLMRFileLocation,lmrFileName);
    
    % This function randomly populates a  map
    % The specification consists of a set of boxes.
    % Landmarks are uniformly sampled inside each box
    
    % Load the configuration
    config = ebe.utils.readJSONFile(lmrConfigFile);
    
    % Go through each region and sample the landmarks
    numRegions = numel(config.landmarkRegions);
    landmarks = zeros(2, 0);
    for m = 1 : numRegions
        region = config.landmarkRegions(m);
        lms = [region.xMin;region.yMin] + ...
            [region.xMax-region.xMin;region.yMax-region.yMin] .* ...
            rand(2, region.numLandmarks);
        landmarks = cat(2, landmarks, lms);
    end
    
    % Plot the map
    FigureManager.getFigure('Map');
    clf
    plot(landmarks', 'k+', 'LineWidth', 2)
    
    % Construct the scenario file; this is a MATLAB struct. The (rather
    % illogical...) structure sets the JSON up properly so it can be read
    % by the other files.
    map = struct();    
    map.landmarks.slam.configuration = 'specified';
    map.landmarks.slam.landmarks = landmarks';
    sensors = struct();
    sensors.enabled = true;
    sensors.sigmaR = [1 0.1];
    sensors.detectionRange = 10;
    sensors.measurementPeriod = 0.2;
    map.sensors.slam = sensors;
    
    % Now find location to write the file
    scenarioFileName = strrep(lmrConfigFile, "_lmr", "");
    [scenarioFileName, scenarioFileLocation] = uiputfile('*_lmr.json', 'Landmarks file', scenarioFileName);
    if (scenarioFileName == 0)
        disp('Aborted')
        return
    end

    scenarioFile = fullfile(scenarioFileLocation,scenarioFileName);

    % Convert the structure to a pretty-printed JSON and write it out
    jsonString = jsonencode(map, 'PrettyPrint', true);
    fileID = fopen(scenarioFile, 'w');
    fprintf(fileID, '%s', jsonString);
    fclose(fileID);
end