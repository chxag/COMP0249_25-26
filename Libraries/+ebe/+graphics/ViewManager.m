classdef ViewManager < ebe.core.ConfigurableComponent

    % Class to manage views contained in a single figure

    properties(Access = protected)

        % Handle for the figure
        figureHandle;

        % The set of views
        views;

        % The figure name associated with this viewer
        figureName;

    end

    methods(Access = public)

        function obj = ViewManager(config, figureName)
            obj@ebe.core.ConfigurableComponent(config);
            if (nargin == 2)
                obj.figureName = figureName;
            end
            obj.views = {};
        end

        function addView(obj, view)
            obj.views{end+1} = view;
        end

        function start(obj)

            % If defined, select the figure
            if (isempty(obj.figureName) == false)
                obj.figureHandle = ebe.graphics.FigureManager.getFigure(obj.figureName);
            end

            % Start everything, pulling legends as we go
            legendHandles = [];
            legendEntries = {};
            for v = 1 : numel(obj.views)
                obj.views{v}.start();
                [handles, entries] = obj.views{v}.legendEntries();
                if (isempty(entries) == false)
                    legendHandles(end + 1) = handles;
                    legendEntries{end + 1} = entries;
                end
            end
            if (isempty(legendEntries) == false)
                legend(legendHandles, legendEntries);
            end
        end

        function stop(obj)
            for v = 1 : numel(obj.views)
                obj.views{v}.stop();
            end
        end

        function visualize(obj, eventArray)
            for v = 1 : numel(obj.views)
                obj.views{v}.visualize(eventArray);
            end
        end
    end
end