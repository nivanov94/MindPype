import mindpype as mp
import numpy as np


"""Unit tests for Scalar class in containers.py"""
class ScalarUnitTests:
    def __init__(self):
        self.__session = mp.Session.create()

    """Verify that Scalar objects are created with correct data types"""
    def TestScalarCreation(self):  
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
        
    """Verify that data is correctly assigned to Scalar objects"""
    def TestScalarData(self):
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
      
    """Ensure assign_random_data executes without error and assigns correct data type based on Scalar type"""
    def TestAssignRandomData(self):
        s_rand1 = mp.Scalar.create(self.__session, complex)
        s_rand2 = mp.Scalar.create(self.__session, bool)

        s_rand1.assign_random_data()
        assert type(s_rand1.data) == complex

        s_rand2.assign_random_data()  
        assert type(s_rand2.data) == bool  

    
    def TestCopyTo(self):
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

"""Unit tests for Tensor class in containers.py"""
class TensorUnitTests:
    def __init__(self):
        self.__session = mp.Session.create()

    """Verify that data is correctly assigned to Tensor objects"""
    def TestTensorData(self):
        t = mp.Tensor.create(self.__session, (1,1))

        # Data assigned to Tensor object must be numpy array or scalar
        try:    
            t.data = np.bool(True) 
        except TypeError:
            pass

        t.data = np.array([[[5]]])   ## line 652
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
    
    """Ensure change_shape executes without error"""
    def TestChangeShape(self):
        t = mp.Tensor.create(self.__session, (1,1))

        # New shape must be tuple or list
        try:    
            t.change_shape(1)   
        except TypeError:
            pass

    """Ensure assign_random_data executes without error"""
    def TestTensorRandomData(self):
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

    def TestCreateFromData(self):
        tensor = mp.Tensor.create_from_data(self.__session, [1,2,3,3])  
        assert type(tensor.data) == np.ndarray 


"""Unit tests for Array class in containers.py"""
class ArrayUnitTests:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestArrayGetElement(self):    ## combine
        arr = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        a = arr.get_element(-1) 
        ## how can i add an index check here???
        try:
            arr.get_element(5)  
        except ValueError:
            print("Index out of bounds")

    def TestArraySetElement(self):
        arr = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        arr.set_element(-1, mp.Scalar.create_from_value(self.__session, 5))   ## line 1106  

        try:
            arr.set_element(5, mp.Scalar.create_from_value(self.__session, 5))   ## line 1109
        except ValueError:
            print("Index out of bounds")

    def TestArrayNumElements(self):
        arr1 = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        num = arr1.num_elements 
        assert num == 4

    def TestCopyTo(self):
        a = mp.Array.create(self.__session, 1, mp.Scalar.create(self.__session, int))
        a.set_element(0, mp.Scalar.create_from_value(self.__session, 4))
        dest1 = mp.Array.create(self.__session, 1, mp.Scalar.create(self.__session, int))
        dest2 = mp.Array.create(self.__session, 5, mp.Scalar.create(self.__session, int))
        dest3 = mp.Array.create(self.__session, 1, mp.Tensor.create(self.__session, (1,1)))

        a.copy_to(dest1)
        assert dest1.get_element(0).data == a.get_element(0).data
        
        try:
            a.copy_to(dest2)
        except ValueError:
            print("Index must match")

        try:
            a.copy_to(dest3)
        except TypeError:
            print("Data type must match")

    def TestArrayToTensor(self):
        arr2 = mp.Array.create(self.__session, 6, mp.Scalar.create(self.__session, int))
        t = arr2.to_tensor()
        assert t.mp_type == mp.MPEnums.TENSOR     

        arr3 = mp.Array.create(self.__session, 1, mp.Scalar.create(self.__session, str))
        arr3.set_element(0, mp.Scalar.create_from_value(self.__session, 'hi'))

        # 
        try:
            t = arr3.to_tensor()   ## line 1247  
        except TypeError:
            print("Array contains non-numeric scalar elements")


"""Unit tests for CircleBuffer class in containers.py"""
class CircleBufferUnitTests:
    def __init__(self):
        self.__session = mp.Session.create()

    def TestNumElements(self):
        cir = mp.CircleBuffer.create(self.__session, 2, mp.Scalar.create(self.__session, int))
        assert cir.num_elements == 0

        cir.enqueue(mp.Scalar.create_from_value(self.__session, 5))
        assert cir.num_elements == 1

        cir.enqueue(mp.Scalar.create_from_value(self.__session, 5))
        assert cir.num_elements == 2
    
    def TestIsFull(self):
        c = mp.CircleBuffer.create(self.__session, 2, mp.Scalar.create(self.__session, int))
        r = c.is_full()
        assert r == False

        c.enqueue(mp.Scalar.create_from_value(self.__session, 1))
        r = c.is_full()
        assert r == False

        c.enqueue(mp.Scalar.create_from_value(self.__session, 1))
        r = c.is_full()
        assert r == True  

    def TestGetQueuedElement(self):
        cir = mp.CircleBuffer.create(self.__session, 5, mp.Scalar.create(self.__session, int))
        try:
            cir.get_queued_element(7) 
        except ValueError:
            print("Index out of bounds")

    def TestPeek(self):
        cir = mp.CircleBuffer.create(self.__session, 1, mp.Scalar.create(self.__session, int))
        assert cir.peek() == None

        cir.enqueue(mp.Scalar.create_from_value(self.__session, 1))
        assert cir.peek().data == 1  

    def TestEnqueue(self):
        c = mp.CircleBuffer.create(self.__session, 2, mp.Scalar.create(self.__session, int))
        c.enqueue(mp.Scalar.create_from_value(self.__session, 1))
        x = c.get_queued_element(0)
        assert x.data == 1

        c.enqueue(mp.Scalar.create_from_value(self.__session, 2))
        y = c.get_queued_element(1)
        assert y.data == 2

        c.enqueue(mp.Scalar.create_from_value(self.__session, 3))  
        z = c.get_queued_element(1)
        assert z.data == 3

    def TestEnqueueChunk(self):
        dest = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        sour1 = mp.CircleBuffer.create(self.__session, 4, mp.Tensor.create(self.__session, (1,1)))

        sour2 = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create_from_value(self.__session, 5))
        sour2.enqueue(mp.Scalar.create_from_value(self.__session, 5))
        sour2.enqueue(mp.Scalar.create_from_value(self.__session, 6))

        try:
            dest.enqueue_chunk(sour1)    
        except TypeError:
            print("Non-matching types")

        dest.enqueue_chunk(sour2)  
        assert dest.get_element(0).data == 5
        assert dest.get_element(1).data == 6

    def TestDequeue(self):
        dest = mp.CircleBuffer.create(self.__session, 1, mp.Scalar.create(self.__session, float))
        assert dest.dequeue() == None

        dest.enqueue(mp.Scalar.create_from_value(self.__session, 5.6))
        assert dest.dequeue().data == 5.6

    def TestMakeCopy(self):
        cir = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        cir.enqueue(mp.Scalar.create_from_value(self.__session, 7))
        c = cir.make_copy()
        assert c.get_element(0).data == 7  

    def TestCopyTo(self):   ### ???
        source = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        source.enqueue(mp.Scalar.create_from_value(self.__session, 7))
        source.enqueue(mp.Scalar.create_from_value(self.__session, 7))
        source1 = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create(self.__session, int))

        dest1 = mp.Array.create(self.__session, 1, mp.Scalar.create(self.__session, int))
        dest2 = mp.CircleBuffer.create(self.__session, 3, mp.Tensor.create(self.__session, (1,1)))

        dest3 = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        source1.copy_to(dest3)   ## line 1663 ... should be good

        dest3.enqueue(mp.Scalar.create_from_value(self.__session, 7))
        source.copy_to(dest3)  

        try:
            source.copy_to(dest1) 
        except ValueError:
            print("Destination array does not have capacity")

        try:
            source.copy_to(dest2)  
        except TypeError:
            print("Non-matching types")

    def TestToTensor(self):
        e = mp.CircleBuffer.create(self.__session, 0, mp.Scalar.create(self.__session, int))
        assert e.to_tensor() == None 

        c1 = mp.CircleBuffer.create(self.__session, 3, mp.Scalar.create(self.__session, str))
        c1.enqueue(mp.Scalar.create_from_value(self.__session, 'hi'))

        # Data must be numeric value
        try:
            c1.to_tensor()    ## line 1706 
        except TypeError:
            print("Non-numeric value")

        c2 = mp.CircleBuffer.create(self.__session, 3, mp.Scalar.create_from_value(self.__session, 5))
        c2.enqueue(mp.Scalar.create_from_value(self.__session, 5))
        c2.enqueue(mp.Scalar.create_from_value(self.__session, 5))
        t = c2.to_tensor()
        # assert type(t) == mp.MPEnums.TENSOR

    def TestRandomData(self):     
        c = mp.CircleBuffer.create(self.__session, 3, mp.Scalar.create(self.__session, int))
        c.assign_random_data()

def test_execute():
    st = ScalarUnitTests()   
    tt = TensorUnitTests()
    at = ArrayUnitTests()
    ct = CircleBufferUnitTests()

    st.TestAssignRandomData()
    st.TestScalarCreation()
    st.TestScalarData()
    st.TestCopyTo()

    tt.TestTensorData()
    tt.TestTensorRandomData()
    tt.TestChangeShape()
    tt.TestCreateFromData()

    at.TestArrayGetElement()
    at.TestArrayNumElements()
    at.TestArraySetElement()
    at.TestArrayToTensor()
    at.TestCopyTo()

    ct.TestGetQueuedElement()
    ct.TestNumElements()
    ct.TestIsFull()
    ct.TestEnqueue()
    ct.TestPeek()
    ct.TestEnqueueChunk()
    ct.TestDequeue()
    ct.TestMakeCopy()
    ct.TestCopyTo()
    ct.TestToTensor()
    ct.TestRandomData()

# test_execute()