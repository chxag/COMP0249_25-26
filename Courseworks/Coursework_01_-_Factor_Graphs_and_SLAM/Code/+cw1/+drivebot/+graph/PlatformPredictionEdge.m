classdef PlatformPredictionEdge < g2o.core.BaseBinaryEdge
    % PlatformPredictionEdge summary of PlatformPredictionEdge
    %
    % This class stores the factor representing the process model which
    % transforms the state from timestep k to k+1
    %
    % The process model is as follows.
    %
    % Define the rotation vector
    %
    %   M = dT * [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0;0 0 1];
    %
    % The new state is predicted from 
    %
    %   x_(k+1) = x_(k) + M * [vx;vy;theta]
    %
    % Note in this case the measurement is actually the mean of the process
    % noise. It has a value of 0. The error vector is given by
    %
    % e(x,z) = inv(M) * (x_(k+1) - x_(k))
    %
    % Note this requires estimates from two vertices - x_(k) and x_(k+1).
    % Therefore, this inherits from a binary edge. We use the convention
    % that vertex slot 1 contains x_(k) and slot 2 contains x_(k+1).
    
    properties(Access = protected)
        % The length of the time step
        dT;
    end
    
    methods(Access = public)
        function obj = PlatformPredictionEdge(dT)
            % PlatformPredictionEdge for PlatformPredictionEdge
            %
            % Syntax:
            %   obj = PlatformPredictionEdge(dT);
            %
            % Description:
            %   Creates an instance of the PlatformPredictionEdge object.
            %   This predicts the state from one timestep to the next. The
            %   length of the prediction interval is dT.
            %
            % Outputs:
            %   obj - (handle)
            %       An instance of a PlatformPredictionEdge

            assert(dT >= 0);
            obj = obj@g2o.core.BaseBinaryEdge(3);            
            obj.dT = dT;
        end
       
        function initialEstimate(obj)
            % INITIALESTIMATE Compute the initial estimate of a platform.
            %
            % Syntax:
            %   obj.initialEstimate();
            %
            % Description:
            %   Compute the initial estimate of the platform x_(k+1) given
            %   an estimate of the platform at time x_(k) and the control
            %   input u_(k+1)

            %warning('PlatformPredictionEdge.initialEstimate: implement')

            % Current state x_k and control input u_{k+1}
            xk = obj.edgeVertices{1}.x;
            uk = obj.z;

            theta = xk(3);
            c = cos(theta);
            s = sin(theta);

            % M(psi_k)
            M = obj.dT * [ c  -s   0;
                           s   c   0;
                           0   0   1 ];

            % Predict x_{k+1} assuming zero process noise
            xkp1 = xk + M * uk;

            % Wrap heading to [-pi, pi]
            xkp1(3) = g2o.stuff.normalize_theta(xkp1(3));

            obj.edgeVertices{2}.x = xkp1;

        end
        
        function computeError(obj)
            % COMPUTEERROR Compute the error for the edge.
            %
            % Syntax:
            %   obj.computeError();
            %
            % Description:
            %   Compute the value of the error, which is the difference
            %   between the measurement and the parameter state in the
            %   vertex. Note the error enters in a nonlinear manner, so the
            %   equation has to be rearranged to make the error the subject
            %   of the formulat
                       
            %warning('PlatformPredictionEdge.computeError: implement')


            xk   = obj.edgeVertices{1}.x;
            xkp1 = obj.edgeVertices{2}.x;
            uk   = obj.z;

            theta = xk(3);
            c = cos(theta);
            s = sin(theta);

            if obj.dT <= 0
                error('PlatformPredictionEdge: dT must be > 0');
            end

            % inv(M) = (1/dT) * [R^T 0; 0 1]
            A = (1/obj.dT) * [ c   s   0;
                              -s   c   0;
                               0   0   1 ];

            dx = xkp1 - xk;

            % Error: inv(M) * (x_{k+1} - x_k) - u_{k+1}
            obj.errorZ = A * dx - uk;

            % Wrap heading error
            obj.errorZ(3) = g2o.stuff.normalize_theta(obj.errorZ(3));

        end
        
        % Compute the Jacobians
        function linearizeOplus(obj)
            % LINEARIZEOPLUS Compute the Jacobians for the edge.
            %
            % Syntax:
            %   obj.computeError();
            %
            % Description:
            %   Compute the Jacobians for the edge. Since we have two
            %   vertices which contribute to the edge, the Jacobians with
            %   respect to both of them must be computed.
            %

            %warning('PlatformPredictionEdge.linearizeOplus: implement')
            xk   = obj.edgeVertices{1}.x;
            xkp1 = obj.edgeVertices{2}.x;

            theta = xk(3);
            c = cos(theta);
            s = sin(theta);

            if obj.dT <= 0
                error('PlatformPredictionEdge: dT must be > 0');
            end

            % A = inv(M(psi_k))
            A = (1/obj.dT) * [ c   s   0;
                              -s   c   0;
                               0   0   1 ];

            dx = xkp1 - xk;

            % dA/dtheta
            dA_dtheta = (1/obj.dT) * [ -s   c   0;
                                      -c  -s   0;
                                       0   0   0 ];

            % Jacobian wrt x_k
            J1 = -A;
            J1(:,3) = J1(:,3) + dA_dtheta * dx;

            % Jacobian wrt x_{k+1}
            J2 = A;

            obj.J{1} = J1;
            obj.J{2} = J2;

        end
    end    
end