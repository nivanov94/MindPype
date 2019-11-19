classdef Array < Data
    %ARRAY Array of Data Objects

    properties
	len {mustBeInteger, mustBeNonnegative}
	elements Data
    end

    methods
	function [a, sess] = Array(sess, len, elements)
	    a = a@Data('array');

	    if nargin == 2
		  % fill with empty tensors as a place holder
		  for i = 1:len
		      elements(i) = Tensor.createZero(sess,[0]);
		  end
	    end

	    a.len = len;
	    a.elements = elements;
        
        % get the ID for this scalar
        uid = sess.produceIdentifier();
        a.uuid = uid;

        % add the scalar to the session
        sess = sess.addObject(a);
	end

	function elem = getElement(a, index)
	    elem = a.elements(index);
	end

	function a = insertElement(a, index, elem)
	    a.elements(index) = elem;
	end
    end
end
