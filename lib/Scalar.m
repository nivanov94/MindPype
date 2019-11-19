classdef Scalar < Data
    %SCALAR Scalar data container

    properties
	    value(1,1)

    end

    methods
	function [s, sess] = Scalar(sess,value)
	    s = s@Data('scalar');
	    s.value = value;

        % get the ID for this scalar
        uid = sess.produceIdentifier();
        s.uuid = uid;

        % add the scalar to the session
        sess = sess.addObject(s);
	end

    end
end
