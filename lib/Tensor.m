classdef Tensor < Data
    %TENSOR Multidimensional data container

    properties
	dim(1,:) {mustBeInteger,mustBeNonnegative} = [0];
	elements {mustBeNumeric}
    end

    methods
	function [t, sess] = Tensor(sess,dimensions,elements)
	    t = t@Data('tensor');

	    if nargin == 1
	        dimensions = [0];
	        elements = zeros(dimensions);
	    elseif nargin == 2
	        elements = zeros(dimensions);
	    end

	    t.dim = dimensions;
	    t.elements = elements;

        
        % get the id for this object
        uid = sess.produceIdentifier();
        t.uuid = uid;

        % add this data to the session
        sess = sess.addObject(t);
	end

    end
    
    methods (Static)
	function [t, sess] = createFromData(sess,dim,elements)
      	    [t, sess] = Tensor(sess,dim,elements);
      	end

      	function [t, sess] = createZero(sess, dim)
      	    [t, sess] = Tensor(sess,dim);
      	end

      	function [t, sess] = createVirtual(sess)
      	    [t, sess] = Tensor(sess);
      	    t.setVirtual(true);
      	end
    end
end

	  

