classdef Block < BciCapObject
    %BLOCK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nodes Node
        edges Edge
	    trials {mustBeInteger,mustBePositive}
	    trial_labels {mustBeInteger}
    end
    
    methods
        function [block, sess] = Block(sess, trial_labels)
            %BLOCK Construct an instance of this class
            %   Detailed explanation goes here
            block.nodes = {};
            block.edges = {};
	        block.trial_labels = trial_labels;
	        block.trials = len(trial_labels);

            % Get the ID for the block within the session
            uuid = sess.produceIdentifier();
            block.uuid = uuid;

            % Add the block to the session
            session = sess.addObject(block);
        end
        
        function block = BlockAddNode(block,node)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            block.nodes{end+1} = node;
        end

	    function block = verify(block)
        	%VERIFY Schedule the nodes within
            % the block and the execution is valid
            
        end

        function block = initialize(block)
            %INITIALIZE Initialize all the block's
            % nodes in preparation for block execution

        end

        function block = close(block)
            %CLOSE Called after block is complete
        end

    end
    
    methods (Access = private)
        function block = schedule(block)
            %SCHEDULE Create the graph schedule


        end
    end
end

