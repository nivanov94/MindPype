classdef Block
    %BLOCK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nodes Node
        edges Edge
	trials {mustBeInteger,mustBePositive}
	trial_labels {mustBeInteger}
    end
    
    methods
        function block = Block(trial_labels)
            %BLOCK Construct an instance of this class
            %   Detailed explanation goes here
            block.nodes = {};
            block.edges = {};
	    block.trial_labels = trial_labels;
	    block.trials = len(trial_labels);
        end
        
        function block = BlockAddNode(block,node)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            block.nodes{end+1} = node;
        end
    end
end

