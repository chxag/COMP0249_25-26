classdef FigureState < handle
    % FigureState summary of FigureState
    %
    % Instances of this class provide a centralised way to access a figure.
    % Services include storing named handles and axes.
    
    properties(Access=public)

        % The handle pointing to the figure
        figureHandle;

        % The figure number
        number;

        % The handle pointing to the axes
        axisHandle;

        % The name of the figure; this is shown in the figure title bar
        name;

        % The renderer
        renderer;

        % Stores a handle to 
        handlesMap;
        
        postDrawActions;
    end
    
    methods(Access=public, Sealed=true)

        function obj = FigureState(name, number, renderer)
            obj.name = name;
            obj.number = number;

            % Set the default painter if not specified on my mac (at
            % least!) painters works better
            if (nargin == 2)
                if (ismac == true)
                    obj.renderer = 'painters';
                else
                    obj.renderer = 'opengl';
                end
            else
                obj.renderer = renderer;
            end

            % Now initialize the state
            obj.initialize();
        end
 
        function obj = initialize(obj)
            % Open the figure
            obj.figureHandle = figure(obj.number);

            % Set the renderer
            obj.figureHandle.Renderer = obj.renderer;

            % Set the figure name
            obj.figureHandle.Name = obj.name;
            %set(obj.figureHandle, 'Name', obj.name);

            % Reset the figure to clear its state
            obj.reset();
        end

        function obj = reset(obj)
            obj.clf();
            obj.postDrawActions = [];
        end

        function obj = clf(obj)
            % If handle is not valid, then the figure had been closed
            % but the figure manager state not cleared. In that case,
            % intialize it again.
            if (~ishandle(obj.figureHandle))
                obj.figureHandle = figure(obj.number);
                set(obj.figureHandle, 'Name', obj.name);
            end
            clf(obj.figureHandle);
            obj.axisHandle = gca;
            obj.hold(true);
            obj.handlesMap = containers.Map();
        end
        
        function handle = getFigure(obj, select)
            handle = obj.figureHandle;
            if ((nargin > 1) && (select == true))
                figure(handle);
            end
        end
        
        function holdState = hold(obj, holdState)
            if (holdState == true)
                hold(obj.axisHandle, 'on');
            else
                hold(obj.axisHandle, 'off');
            end
        end
        
        function obj = select(obj)
            if (~ishandle(obj.figureHandle))
                obj.initialize();
            end
            figure(obj.figureHandle);
        end

        function obj = refreshData(obj)
            refreshdata(obj.figureHandle);
        end
        
        function obj = runPostDrawActions(obj)
            for k = 1 : length(obj.postDrawActions)
                obj.postDrawActions(k).invoke();
            end
        end
        
        function obj = setTitle(obj, titleString)
            assert(isa(titleString, 'char'), 'figurestate:titlenotstring', ...
                'Title must be a string; method called with an object of class %s', class(titleString));
            % For some reason, the axis handle can become invalidated
            if (ishandle(obj.axisHandle) == false)
                figure(obj.number);
                obj.axisHandle = gca;
            end
            h = get(obj.axisHandle, 'Title');
            h.String = titleString;
        end
        
        function axisHandle = getAxisHandle(obj)
            axisHandle = obj.axisHandle;
        end
        
        function handle = getHandle(obj, handleName)
            if (isKey(obj.handlesMap, handleName))
                handle = obj.handlesMap(handleName);
            else
                handle = [];
            end
        end
        
        function success = addHandle(obj, handleName, handle)
            if (isKey(obj.handlesMap, handleName))
                success = false;
            else
                obj.handlesMap(handleName) = handle;
            end
        end
        
        function obj = addPostDrawAction(obj, postDrawAction)
            assert(isa(postDrawAction, 'ebe.graphics.GraphicalAction'), ...
                'figurestate:postdrawactionwrongclass', ...
                'The post draw action must inherit from ebe.graphics.GraphicalAction; the class of the submitted object is %s', ...
                class(postDrawAction));
            obj.postDrawActions = cat(2, obj.postDrawActions, postDrawAction);
        end        
    end    
end