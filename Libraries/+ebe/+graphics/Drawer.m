classdef Drawer < handle
    % Drawer summary of Drawer
    %
    % Subclasses of this object are used to draw specific graphical
    % elements, such as covariance ellipses.

    methods(Access = public, Abstract)
        update(obj, x, P);
    end
end