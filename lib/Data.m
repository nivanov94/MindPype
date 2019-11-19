classdef Data < BciCapObject
    %DATA Abstract container for data used during the session
    %  
    
    properties
	    is_virtual logical = false
    end
    
    methods
        function d = Data(type)
            %Data Construct an instance of this class
            %   Detailed explanation goes here
            d = d@BciCapObject(type);
        end

        function d = setVirtual(d,virtual)
	        d.is_virtual = virtual;
	    end
    end
end
