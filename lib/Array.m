classdef Array < Data
    %ARRAY Array of Data Objects

    properties
	len {mustBeInteger, MustbeNonnegative}
	elements Data
    end

    methods
	function a = Array(len, elements)
	    a = a@Data('Array');

	    if nargin == 1
		  % fill with empty tensors as a place holder
		  for i = 1:len
		      elements(i) = Tensor.createZero([0]);
		  end
	      end

	    a.len = len;
	    a.elements = elements;
	end

	function elem = getElement(a, index)
	    a.elements(index);
	end

	function a = insertElement(a, index, elem)
	    a.element(index) = elem;
	end
    end
end
