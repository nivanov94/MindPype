classdef Session < BciCapObject
    %SESSION The object containing all elements of the BCI data collection
    % session
    %  
    
    properties
        blocks(1,:) Block
        data
        obj_ids(1,:) {mustBeInteger}
    end
    
    methods
        function sess = Session()
            %SESSION Construct an instance of this class
            %   Detailed explanation goes here
            sess = sess@BciCapObject('session');
            sess.data = {};
            sess.uuid = 0;
            sess.obj_ids = [0];
            
        end
        
        function uid = produceIdentifier(sess)
            %PRODUCEIDENTIFIER Return a unique ID number
            % for an object within the session
            
            min_id = min(sess.obj_ids);
            max_id = max(sess.obj_ids);
            available_ids = setxor(sess.obj_ids, [min_id:(max_id+1)]);
            uid = available_ids(1);
        end

        function sess = addObject(sess,obj)
            sess.data{end+1} = obj;
            sess.obj_ids(end+1) = obj.uuid;
        end
    end
end

