import mindpype as mp
import numpy as np


class ScalarUnitTests:
    """Unit tests for Scalar class in containers.py"""
    def __init__(self):
        self.__session = mp.Session.create()

    def TestScalarCreation(self):  
        """Verify that Scalar objects are created with correct data types"""

        s_int_string = mp.Scalar.create(self.__session, 'int')
        assert type(s_int_string.data) == int

        s_float = mp.Scalar.create(self.__session, float)   
        assert type(s_float.data) == float

        s_int = mp.Scalar.create(self.__session, int)
        assert type(s_int.data) == int

        # Cannot create Scalar with type double
        try:    
            s_bad = mp.Scalar.create(self.__session, 'double') 
        except ValueError:
            pass

        s_virtual_int = mp.Scalar.create_virtual(self.__session, int)
        assert type(s_virtual_int.data) == int

        # Cannot create Scalar with type double
        try: 
            s_virtual_bad = mp.Scalar.create_virtual(self.__session, 'double') 
        except ValueError:
            pass

        s_from_val = mp.Scalar.create_from_value(self.__session, 4)
        assert type(s_from_val.data) == int

        # Cannot create Scalar with type double
        try:
            s_from_val_bad = mp.Scalar.create_from_value(self.__session, np.double(4.2)) 
        except TypeError:
            pass

        # ext_src = mp.source.InputLSLStream.create_marker_uncoupled_data_stream(self.__session, active=False)
        # out_src = mp.source.InputLSLStream.create_marker_uncoupled_data_stream(self.__session, active=False)
        # s_source = 
        
    def TestScalarData(self):
        """Verify that data is correctly assigned to Scalar objects"""
        s_int = mp.Scalar.create(self.__session, int)

        s_int.data = np.array([1])   
        assert type(s_int.data) == int

        # Input array must contain one element
        try:
            s_int.data = np.array([1,2]) 
        except ValueError:
            pass

        # Cannot create Scalar with type double
        try:
            s_int.data = np.double(1.2)  
        except ValueError:
            pass
        
        s_complex = mp.Scalar.create(self.__session, complex)

        s_complex.data = np.cdouble(4.5,2)   
        assert type(s_complex.data) == complex
      
    def TestScalarAssignRandomData(self):
        """Ensure assign_random_data executes without error and assigns correct data type based on Scalar type"""
        s_rand1 = mp.Scalar.create(self.__session, complex)
        s_rand2 = mp.Scalar.create(self.__session, bool)

        s_rand1.assign_random_data()
        assert type(s_rand1.data) == complex

        s_rand2.assign_random_data()  
        assert type(s_rand2.data) == bool  
    
    def TestScalarCopyTo(self):
        """Ensure copy_to executes without error"""
        s_int = mp.Scalar.create(self.__session, int)
        s_int.data = 1

        s_dest1 = mp.Scalar.create(self.__session, int)
        s_dest2 = mp.Scalar.create(self.__session, float)

        s_int.copy_to(s_dest1)
        assert s_dest1.data == 1

        # Scalars must be of same data type
        try:
            s_int.copy_to(s_dest2)
        except TypeError:
            pass

    def TestScalarCreateFromSource(self):
        src = mp.source.InputLSLStream.create_marker_uncoupled_data_stream(self.__session, active=False)

        s_source = mp.Scalar.create_from_source(self.__session, int, src)

    def TestScalarVolatileData(self):
        src = mp.source.InputLSLStream.create_marker_uncoupled_data_stream(self.__session, active=False)

        s_source = mp.Scalar.create_from_source(self.__session, int, src)

        # Source must be an active stream
        try:
            s_source.poll_volatile_data()
        except RuntimeError:
            pass

        s = mp.containers.Scalar(self.__session, int, ext_out = src)
        # s.push_volatile_outputs()

        # Scalar can't be virtual and volatile
        try:
            s_virtual = mp.containers.Scalar(self.__session, int, virtual=True, ext_src=src)
        except ValueError:
            pass
        


class TensorUnitTests:
    """Unit tests for Tensor class in containers.py"""
    def __init__(self):
        self.__session = mp.Session.create()

    def TestTensorData(self):
        """Verify that data is correctly assigned to Tensor objects"""
        t = mp.Tensor.create(self.__session, (1,1))

        # Data assigned to Tensor object must be numpy array or scalar
        try:    
            t.data = np.bool(True) 
        except TypeError:
            pass

        t.data = np.array([[[5]]])
        assert t.shape == (1,1) 
        assert np.array_equal(t.data, np.array([[5]]))

        t.data = np.array([5])
        assert t.shape == (1,1)      
        assert np.array_equal(t.data, np.array([[5]]))

        # Shape of assigned data must match shape of Tensor
        try:
            t.data = np.ndarray([1,2,3])
        except ValueError:
            pass
    
    def TestTensorChangeShape(self):
        """Ensure change_shape executes without error"""
        t = mp.Tensor.create(self.__session, (1,1))

        # New shape must be tuple or list
        try:    
            t.change_shape(1)   
        except TypeError:
            pass

    def TestTensorRandomData(self):
        """Ensure assign_random_data executes without error"""
        t_rand1 = mp.Tensor.create(self.__session, (1, 1, 1, 1))

        # Rank of Tensor must be 2 or 3 for this function 
        try:
            t_rand1.assign_random_data(covariance=True) 
        except ValueError:
            print("Rank must be 2 or 3")

        t_rand2 = mp.Tensor.create(self.__session, (1, 2, 3))
        try:
            t_rand2.assign_random_data(covariance=True)   
        except ValueError:
            print("Last 2 dimensions must be square")

        t_rand3 = mp.Tensor.create(self.__session, (3,3))
        t_rand3.assign_random_data(covariance=True) 

    def TestTensorCreateFromData(self):
        """Ensure create_from_data executes without error"""
        t_data = mp.Tensor.create_from_data(self.__session, [1,2,3,3])  
        assert type(t_data.data) == np.ndarray 

    def TestTensorCreateFromSource(self):
        src = mp.source.InputLSLStream.create_marker_uncoupled_data_stream(self.__session, active=False)

        t_input = mp.Tensor.create_from_source(self.__session, (1,1), src, direction="input")
        t_output = mp.Tensor.create_from_source(self.__session, (1,1), src, direction="output")

        # Direction must be input or output
        try: 
            t_invalid = mp.Tensor.create_from_source(self.__session, (1,1), src, direction="invalid")
        except ValueError:
            pass

        # Source must be active
        try:
            t_input.poll_volatile_data()
        except RuntimeError:
            pass

class ArrayUnitTests:
    """Unit tests for Array class in containers.py"""
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestArraySetandGetElement(self):  
        """Ensure that Array elements are properly set using set_element and that get_element returns correct element"""
        a_int = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        a_int.set_element(-1, mp.Scalar.create_from_value(self.__session, 5))  

        element = a_int.get_element(-1) 
        assert element.data == 5

        # Index is out of bounds of Array object
        try:
            a_int.get_element(5)  
        except ValueError:
            pass

        # Index is out of bounds of Array object
        try:
            a_int.set_element(5, mp.Scalar.create_from_value(self.__session, 5))   
        except ValueError:
            pass

    def TestArrayNumElements(self):
        """Ensure num_elements executes without error and returns correct number of elements"""
        a_elements = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        assert a_elements.num_elements == 4

    def TestArrayCopyTo(self):
        """Ensure copy_to executes without error"""
        a_src = mp.Array.create(self.__session, 1, mp.Scalar.create(self.__session, int))
        a_src.set_element(0, mp.Scalar.create_from_value(self.__session, 4))

        a_dest1 = mp.Array.create(self.__session, 1, mp.Scalar.create(self.__session, int))
        a_dest2 = mp.Array.create(self.__session, 5, mp.Scalar.create(self.__session, int))
        a_dest3 = mp.Array.create(self.__session, 1, mp.Tensor.create(self.__session, (1,1)))

        a_src.copy_to(a_dest1)
        assert a_dest1.get_element(0).data == a_src.get_element(0).data
        
        # Array capacities must match
        try:
            a_src.copy_to(a_dest2)
        except ValueError:
            pass

        # Array data types must match
        try:
            a_src.copy_to(a_dest3)
        except TypeError:
            pass

    def TestArrayToTensor(self):
        """Ensure to_tensor executes without error and that Array object becomes Tensor object"""
        a_int = mp.Array.create(self.__session, 6, mp.Scalar.create(self.__session, int))
        t_int = a_int.to_tensor()
        assert t_int.mp_type == mp.MPEnums.TENSOR     

        a_str = mp.Array.create(self.__session, 1, mp.Scalar.create(self.__session, str))
        a_str.set_element(0, mp.Scalar.create_from_value(self.__session, 'hi'))

        # Array must be numeric data type
        try:
            t_str = a_str.to_tensor()    
        except TypeError:
            pass


class CircleBufferUnitTests:
    """Unit tests for CircleBuffer class in containers.py"""
    def __init__(self):
        self.__session = mp.Session.create()

    def TestCBNumElements(self):
        """Ensure num_elements executes without error and test for enqueued elements"""
        c = mp.CircleBuffer.create(self.__session, 2, mp.Scalar.create(self.__session, int))
        assert c.num_elements == 0

        c.enqueue(mp.Scalar.create_from_value(self.__session, 5))
        assert c.num_elements == 1

        c.enqueue(mp.Scalar.create_from_value(self.__session, 5))
        assert c.num_elements == 2
    
    def TestCBIsFull(self):
        """Verify that is_full returns True when CircleBuffer is full"""
        c = mp.CircleBuffer.create(self.__session, 2, mp.Scalar.create(self.__session, int))
        assert c.is_full() == False

        c.enqueue(mp.Scalar.create_from_value(self.__session, 1))
        assert c.is_full() == False

        c.enqueue(mp.Scalar.create_from_value(self.__session, 1))
        assert c.is_full() == True  

    def TestCBGetQueuedElement(self):
        """Ensure get_queued_element executes without error"""
        c = mp.CircleBuffer.create(self.__session, 5, mp.Scalar.create(self.__session, int))

        # Index is out of bounds of CircleBuffer
        try:
            c.get_queued_element(7) 
        except ValueError:
            pass

    def TestCBPeek(self):
        """Ensure peek executes without error and verify correct elements are returned"""
        c = mp.CircleBuffer.create(self.__session, 1, mp.Scalar.create(self.__session, int))
        assert c.peek() == None

        c.enqueue(mp.Scalar.create_from_value(self.__session, 1))
        assert c.peek().data == 1  

    def TestCBEnqueue(self):
        """Ensure enqueue executes without error and verify correct elements are enqueued"""
        c = mp.CircleBuffer.create(self.__session, 2, mp.Scalar.create(self.__session, int))
        c.enqueue(mp.Scalar.create_from_value(self.__session, 1))
        assert c.get_queued_element(0).data == 1

        c.enqueue(mp.Scalar.create_from_value(self.__session, 2))
        assert c.get_queued_element(1).data == 2

        c.enqueue(mp.Scalar.create_from_value(self.__session, 3))  
        assert c.get_queued_element(1).data == 3

    def TestCBEnqueueChunk(self):
        """Ensure enqueue_chunk executes without error and verify correct elements are enqueued"""
        c_dest = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        c_src1 = mp.CircleBuffer.create(self.__session, 4, mp.Tensor.create(self.__session, (1,1)))

        c_src2 = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create_from_value(self.__session, 5))
        c_src2.enqueue(mp.Scalar.create_from_value(self.__session, 5))
        c_src2.enqueue(mp.Scalar.create_from_value(self.__session, 6))

        # Source and destination have non matching types
        try:
            c_dest.enqueue_chunk(c_src1)    
        except TypeError:
            pass

        c_dest.enqueue_chunk(c_src2)  
        assert c_dest.get_element(0).data == 5
        assert c_dest.get_element(1).data == 6

    def TestCBDequeue(self):
        """Ensure dequeue executes without error and verify correct elements are dequeued"""
        c_dest = mp.CircleBuffer.create(self.__session, 1, mp.Scalar.create(self.__session, float))
        assert c_dest.dequeue() == None

        c_dest.enqueue(mp.Scalar.create_from_value(self.__session, 5.6))
        assert c_dest.dequeue().data == 5.6

    def TestCBMakeCopy(self):
        """Ensure make_copy executes without error and verify correct elements"""
        c = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        c.enqueue(mp.Scalar.create_from_value(self.__session, 7))

        c_copy = c.make_copy()
        assert c_copy.get_element(0).data == 7  

    def TestCBCopyTo(self):  
        """Ensure copy_to executes without error"""
        c_src = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create(self.__session, int))

        c_dest1 = mp.Array.create(self.__session, 1, mp.Scalar.create(self.__session, int))
        c_dest2 = mp.CircleBuffer.create(self.__session, 3, mp.Tensor.create(self.__session, (1,1)))
        c_dest3 = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create(self.__session, int))

        c_src.copy_to(c_dest3)   

        c_src.enqueue(mp.Scalar.create_from_value(self.__session, 7))
        c_src.enqueue(mp.Scalar.create_from_value(self.__session, 7))

        c_dest3.enqueue(mp.Scalar.create_from_value(self.__session, 7))
        c_src.copy_to(c_dest3)  

        # Source and destination must have same capacities
        try:
            c_src.copy_to(c_dest1) 
        except ValueError:
            pass

        # Source and destination must have matching types
        try:
            c_src.copy_to(c_dest2)  
        except TypeError:
            pass

    def TestCBToTensor(self):
        """Ensure CircleBuffer object is converted to Tensor object without error"""
        c_empty = mp.CircleBuffer.create(self.__session, 0, mp.Scalar.create(self.__session, int))
        assert c_empty.to_tensor() == None 

        c_string = mp.CircleBuffer.create(self.__session, 3, mp.Scalar.create(self.__session, str))
        c_string.enqueue(mp.Scalar.create_from_value(self.__session, 'hi'))

        # Data must be numeric value
        try:
            c_string.to_tensor()    
        except TypeError:
            pass

        c = mp.CircleBuffer.create(self.__session, 3, mp.Scalar.create_from_value(self.__session, 5))
        c.enqueue(mp.Scalar.create_from_value(self.__session, 5))
        c.enqueue(mp.Scalar.create_from_value(self.__session, 5))

        t = c.to_tensor()
        print(type(t))
        assert t.mp_type == mp.MPEnums.TENSOR

    def TestCBRandomData(self):     
        """Ensure assign_random_data executes without error"""
        c = mp.CircleBuffer.create(self.__session, 3, mp.Scalar.create(self.__session, int))
        c.assign_random_data()

def test_execute():
    s = ScalarUnitTests()   
    t = TensorUnitTests()
    a = ArrayUnitTests()
    c = CircleBufferUnitTests()

    s.TestScalarAssignRandomData()
    s.TestScalarCreation()
    s.TestScalarData()
    s.TestScalarCopyTo()
    s.TestScalarCreateFromSource()
    s.TestScalarVolatileData()

    t.TestTensorData()
    t.TestTensorRandomData()
    t.TestTensorChangeShape()
    t.TestTensorCreateFromData()
    t.TestTensorCreateFromSource()

    a.TestArraySetandGetElement()
    a.TestArrayNumElements()
    a.TestArrayToTensor()
    a.TestArrayCopyTo()

    c.TestCBGetQueuedElement()
    c.TestCBNumElements()
    c.TestCBIsFull()
    c.TestCBEnqueue()
    c.TestCBPeek()
    c.TestCBEnqueueChunk()
    c.TestCBDequeue()
    c.TestCBMakeCopy()
    c.TestCBCopyTo()
    c.TestCBToTensor()
    c.TestCBRandomData()

test_execute()