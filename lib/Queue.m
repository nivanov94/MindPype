classdef Queue < Array
    
    properties
        head {mustBeInteger, mustBeNonnegative}
        tail {mustBeInteger, mustBeNonnegative}
    end
    
    methods
        function [q, sess] = Queue(sess,len)
            q = q@Array(sess,len);
            q.head = 0;
            q.tail = 0;
            
            % get the ID for this scalar
            uid = sess.produceIdentifier();
            a.uuid = uid;
            
            % add the scalar to the session
            sess = sess.addObject(a);
        end
        
        function q = enqueue(q,elem)
            q.tail = mod((q.tail + 1),q.len);
            if q.tail == q.head
                warning("Queue Full - enqueue operation overwriting element at head.");
            end
            q = q.insertElement(q.tail,elem);
        end
        
        function [q, elem] = dequeue(q)
            if q.head ~= q.tail % ensure queue is not empty
                elem = q.getElement(q.head);
                q.head = mod(q.head + 1, q.len);
            else
                warning("Queue is empty - Nothing to dequeue");
                elem = [];
            end
        end
        
        function elem = peek(q)
            if q.head == q.tail
                warning("Queue is empty - No values to return");
                elem = [];
            else
                elem = q.getElement(q.head);
            end
        end
    end
end

