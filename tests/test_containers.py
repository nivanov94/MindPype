import mindpype as mp
import numpy as np


## try to debug errors more
## add more checks to verify things are actually working as intended
class ScalarUnitTests:
    def __init__(self):
        self.__session = mp.Session.create()

    def TestScalarCreation(self):  
        s = mp.Scalar.create(self.__session, 'int')
        scal = mp.Scalar.create(self.__session, float)   
        assert type(scal.data) == float
        random_scal = mp.Scalar.create(self.__session, int)
        assert type(random_scal.data) == int
        try:
            bad_scalar = mp.Scalar.create(self.__session, 'double') 
        except ValueError:
            print("No allowed type double")

        virtual = mp.Scalar.create_virtual(self.__session, int)   
        try: 
            virtual1 = mp.Scalar.create_virtual(self.__session, 'double') 
        except ValueError:
            print("No allowed type double")

        val_scal = mp.Scalar.create_from_value(self.__session, 4)
        assert type(val_scal.data) == int
        try:
            val_scalar = mp.Scalar.create_from_value(self.__session, np.double(4.2)) 
        except TypeError:
            print("No allowed type double")
        
    def TestScalarData(self):
        scal = mp.Scalar.create(self.__session, int)
        scal1 = mp.Scalar.create(self.__session, complex)

        scal.data = np.array([1])   
        assert type(scal.data) == int
        try:
            scal.data = np.array([1,2]) 
        except ValueError:
            print("Input numpy array must contain one element")
        
        # scal.data = np.float64(3.258)       ## not working 
        # assert type(scal.data) == float
        scal1.data = np.cdouble(4.5,2)   
        assert type(scal1.data) == complex

        try:
            scal.data = np.double(1.2)  
        except ValueError:
            print("No allowed type double")
      
    def TestAssignRandomData(self):
        rand_scal1 = mp.Scalar.create(self.__session, complex)
        rand_scal2 = mp.Scalar.create(self.__session, bool)
        rand_scal1.assign_random_data()
        assert type(rand_scal1.data) == complex
        rand_scal2.assign_random_data()  
        assert type(rand_scal2.data) == bool  

    def TestCopyTo(self):
        scal = mp.Scalar.create(self.__session, int)
        scal.data = 1

        dest1 = mp.Scalar.create(self.__session, int)
        dest2 = mp.Scalar.create(self.__session, float)

        scal.copy_to(dest1)
        assert dest1.data == 1

        try:
            scal.copy_to(dest2)
        except TypeError:
            print("Scalars have different types")


class TensorUnitTests:
    def __init__(self):
        self.__session = mp.Session.create()

    def TestTensorData(self):
        t = mp.Tensor.create(self.__session, (1,1))
        try:    
            t.data = np.bool(True) 
        except TypeError:
            print("Data assigned must be numpy array or scalar")

        # t.data = np.ndarray([1], [], [])  ## line 652  .... will this be reacheD?
        # print(t.data.shape)

        try:
            t.data = np.ndarray([1, 2, 3])
        except ValueError:
            print("Data shape does not match tensor")
    
    def TestChangeShape(self):
        t = mp.Tensor.create(self.__session, (1,1))
        try:    
            t.change_shape(1)   
        except TypeError:
            print("New shape must be tuple or list")

    def TestTensorRandomData(self):
        tensor = mp.Tensor.create(self.__session, (1, 1, 1, 1)) 
        try:
            tensor.assign_random_data(covariance=True) 
        except ValueError:
            print("Rank must be 2 or 3")

        tensor1 = mp.Tensor.create(self.__session, (1, 2, 3))
        try:
            tensor1.assign_random_data(covariance=True)   
        except ValueError:
            print("Last 2 dimensions must be square")

        tensor2 = mp.Tensor.create(self.__session, (3,3))
        tensor2.assign_random_data(covariance=True) 

    def TestCreateFromData(self):
        tensor = mp.Tensor.create_from_data(self.__session, [1,2,3,3])  
        assert type(tensor.data) == np.ndarray 

class ArrayUnitTests:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestArrayGetElement(self):
        arr = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        a = arr.get_element(-1) 
        ## how can i add an index check here???
        try:
            arr.get_element(5)  
        except ValueError:
            print("Index out of bounds")

    def TestArraySetElement(self):
        arr = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        arr.set_element(-1, mp.Scalar.create_from_value(self.__session, 5))   ## line 1106  - debugger wont work??

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
        # assert t.mp_type == MPEnums.TENSOR        ## what type should this be

        arr3 = mp.Array.create(self.__session, 1, mp.Scalar.create(self.__session, bool))
        arr3.set_element(0, mp.Scalar.create_from_value(self.__session, True))
        try:
            t = arr3.to_tensor()   ## line 1247  - go to containers file line 1246
        except TypeError:
            print("Array contains non-numeric scalar elements")

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
        assert cir.peek().data == 1   ## line 1486

    def TestEnqueue(self):
        c = mp.CircleBuffer.create(self.__session, 2, mp.Scalar.create(self.__session, int))
        c.enqueue(mp.Scalar.create_from_value(self.__session, 1))
        x = c.get_queued_element(0)
        assert x.data == 1

        c.enqueue(mp.Scalar.create_from_value(self.__session, 2))
        y = c.get_queued_element(1)
        assert y.data == 2

        c.enqueue(mp.Scalar.create_from_value(self.__session, 3))   ## line 1513
        z = c.get_queued_element(1)
        assert z.data == 3

    def TestEnqueueChunk(self):
        dest = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        sour1 = mp.CircleBuffer.create(self.__session, 4, mp.Tensor.create(self.__session, (1,1)))

        sour2 = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create_from_value(self.__session, 5))
        sour2.enqueue(mp.Scalar.create_from_value(self.__session, 5))
        sour2.enqueue(mp.Scalar.create_from_value(self.__session, 6))

        try:
            dest.enqueue_chunk(sour1)    ## line 1542   
        except TypeError:
            print("Non-matching types")

        dest.enqueue_chunk(sour2)   ## line 1548
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

    def TestCopyTo(self):
        source = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        source.enqueue(mp.Scalar.create_from_value(self.__session, 7))
        source.enqueue(mp.Scalar.create_from_value(self.__session, 7))

        dest1 = mp.Array.create(self.__session, 1, mp.Scalar.create(self.__session, int))
        dest2 = mp.CircleBuffer.create(self.__session, 3, mp.Tensor.create(self.__session, (1,1)))

        dest3 = mp.CircleBuffer.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        source.copy_to(dest3)   

        dest3.enqueue(mp.Scalar.create_from_value(self.__session, 7))
        source.copy_to(dest3)   ## line 1664

        try:
            source.copy_to(dest1)  ## line 1643
        except ValueError:
            print("Destination array does not have capacity")

        try:
            source.copy_to(dest2)   ## line 1650
        except TypeError:
            print("Non-matching types")

    def TestToTensor(self):
        e = mp.CircleBuffer.create(self.__session, 0, mp.Scalar.create(self.__session, int))
        assert e.to_tensor() == None   ## line 1697

        c1 = mp.CircleBuffer.create(self.__session, 3, mp.Scalar.create(self.__session, bool))
        c1.enqueue(mp.Scalar.create_from_value(self.__session, True))
        try:
            c1.to_tensor()    ## line 1704
        except TypeError:
            print("Proper error")

        c2 = mp.CircleBuffer.create(self.__session, 3, mp.Scalar.create_from_value(self.__session, 5))
        c2.enqueue(mp.Scalar.create_from_value(self.__session, 5))
        c2.enqueue(mp.Scalar.create_from_value(self.__session, 5))
        t = c2.to_tensor()
        # assert type(t) == MPEnums.TENSOR

    def TestRandomData(self):
        c = mp.Array.create(self.__session, 3, mp.Scalar.create(self.__session, int))
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
    at.TestArraySetElement
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

test_execute()