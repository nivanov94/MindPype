classdef Tensor < Data
    %TENSOR Multidimensional data container

    properties
	dim(1,:) {mustBeInteger,mustBePositive} = [0];
	elements {mustBeNumeric}
    end

    methods
	function t = Tensor(dimensions,elements)
	    t = t@Data('Tensor');

	    if nargin == 0
	        dimensions = [0];
	        elements = zeros(dimensions);
	    elseif nargin == 1
	        elements = zeros(dimensions);
	    end

	    t.dim = dimensions;
	    t.elements = elements;

	end

    end
    
    methods (Static)
	function t = CreateFromData(dim,elements)
      	    t = Tensor(dim,elements);
      	end

      	function t = CreateZero(dim)
      	    t = Tensor(dim);
      	end

      	function t = CreateVirtual()
      	    t = Tensor();
      	    t.setVirtual(true);
      	end
    end
end

	  

