classdef LandmarkRangeBearingEdge < g2o.core.BaseBinaryEdge
    % LandmarkRangeBearingEdge summary of LandmarkRangeBearingEdge
    %
    % This class stores an edge which represents the factor for observing
    % the range and bearing of a landmark from the vehicle. Note that the
    % sensor is fixed to the platform.
    %
    % The measurement model is
    %
    %    z_(k+1)=h[x_(k+1)]+w_(k+1)
    %
    % The measurements are r_(k+1) and beta_(k+1) and are given as follows.
    % The sensor is at (lx, ly).
    %
    %    dx = lx - x_(k+1); dy = ly - y_(k+1)
    %
    %    r(k+1) = sqrt(dx^2+dy^2)
    %    beta(k+1) = atan2(dy, dx) - theta_(k+1)
    %
    % The error term
    %    e(x,z) = z(k+1) - h[x(k+1)]
    %
    % However, remember that angle wrapping is required, so you will need
    % to handle this appropriately in compute error.
    %
    % Note this requires estimates from two vertices - x_(k+1) and l_(k+1).
    % Therefore, this inherits from a binary edge. We use the convention
    % that vertex slot 1 contains x_(k+1) and slot 2 contains l_(k+1).
    
    methods(Access = public)
    
        function obj = LandmarkRangeBearingEdge()
            % LandmarkRangeBearingEdge for LandmarkRangeBearingEdge
            %
            % Syntax:
            %   obj = LandmarkRangeBearingEdge();
            %
            % Description:
            %   Creates an instance of the LandmarkRangeBearingEdge object.
            %   Note we feed in to the constructor the landmark position.
            %   This is to show there is another way to implement this
            %   functionality from the range bearing edge from activity 3.
            %
            % Outputs:
            %   obj - (handle)
            %       An instance of a LandmarkRangeBearingEdge.

            obj = obj@g2o.core.BaseBinaryEdge(2);
        end
        
        function initialEstimate(obj)
            % INITIALESTIMATE Compute the initial estimate of the landmark.
            %
            % Syntax:
            %   obj.initialEstimate();
            %
            % Description:
            %   Compute the initial estimate of the landmark given the
            %   platform pose and observation.

            %warning('LandmarkRangeBearingEdge.initialEstimate: implement')
            
            xk = obj.edgeVertices{1}.x(1:2);
            theta = obj.edgeVertices{1}.x(3);
            z = obj.z;
            phi = theta + z(2);

            m = zeros(2, 1);
            m(1) = xk(1) + z(1) * cos(phi);
            m(2) = xk(2) + z(1) * sin(phi);

            obj.edgeVertices{2}.setEstimate(m);
        end
        
        function computeError(obj)
            % COMPUTEERROR Compute the error for the edge.
            %
            % Syntax:
            %   obj.computeError();
            %
            % Description:
            %   Compute the value of the error, which is the difference
            %   between the predicted and actual range-bearing measurement.

            %warning('LandmarkRangeBearingEdge.computeError: implement')
            obj.errorZ = zeros(2, 1);
            
            x = obj.edgeVertices{1}.estimate();
            landmark = obj.edgeVertices{2}.estimate();
            dx = landmark(1:2) - x(1:2);
            r = norm(dx);
            r_pred = r;
            b_pred = atan2(dx(2), dx(1)) - x(3);
            
            obj.errorZ(1) = obj.z(1) - r_pred;
            obj.errorZ(2) = g2o.stuff.normalize_theta(obj.z(2) - b_pred);
        end
        
        function linearizeOplus(obj)
            % linearizeOplus Compute the Jacobian of the error in the edge.
            %
            % Syntax:
            %   obj.linearizeOplus();
            %
            % Description:
            %   Compute the Jacobian of the error function with respect to
            %   the vertex.
            %

            %warning('LandmarkRangeBearingEdge.linearizeOplus: implement')

            obj.J{1} = eye(2, 3);
            obj.J{2} = eye(2);

            landmark = obj.edgeVertices{2}.estimate();
            x = obj.edgeVertices{1}.estimate();
            dx = landmark(1:2) - x(1:2);
            r = norm(dx);

            obj.J{1} = ...
                [dx(1)/r -dx(2)/r 0;
                -dx(2)/r^2 dx(1)/r^2 1];
            obj.J{2} = [-dx(1)/r -dx(2)/r;
                         dx(2)/r^2 -dx(1)/r^2];
        end        
    end
end
