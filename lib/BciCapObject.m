classdef BciCapObject
  %Base class for all BCI_Capture Objects

  properties
    uuid {mustBeInteger}
    type(1,:) char {mustBeMember(type,{'null','tensor','block','session','array','scalar'})} = 'null'
  end

  methods
    function o = BciCapObject(type)
        o.type = type;
    end
  end
end
