classdef Session
    %SESSION The object containing all elements of the BCI data collection
    % session
    %  
    
    properties
        blocks(1,:) Block
        ui
        data (1,:) Data
    end
    
    methods
        function obj = Session()
            %SESSION Construct an instance of this class
            %   Detailed explanation goes here
            obj.blocks = {};
            
        end
        
        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end

