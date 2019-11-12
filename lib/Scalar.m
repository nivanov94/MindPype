classdef Scalar < Data
    %SCALAR Scalar data container

    properties
	value(1,1)

    end

    methods
	function s = Scalar(value)
	    s = s@Data('Scalar');
	    s.value = value;
	end

    end
end
