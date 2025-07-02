import mindpype as mp
import numpy as np

class ScalarUnitTests:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestScalarCreation(self):
        inScalar = mp.Scalar.create(self.__session, 'int')   
        scal = mp.Scalar.create(self.__session, float)   
        random_scal = mp.Scalar.create(self.__session, int)
        try:
            bad_scalar = mp.Scalar.create(self.__session, 'double')  ## line 121
        except ValueError:
            print("Proper error")

        virtual = mp.Scalar.create_virtual(self.__session, 'int')   
        try: 
            virtual1 = mp.Scalar.create_virtual(self.__session, 'double') 
        except ValueError:
            print("Proper error")

        try:
            val_scalar = mp.Scalar.create_from_value(self.__session, np.double(4.2)) 
        except TypeError:
            print("Proper error")
        
    def TestScalarData(self):
        scal = mp.Scalar.create(self.__session, int)
        scal.data = np.array([1])   
        try:
            scal.data = np.array([1,2]) 
        except ValueError:
            print("Proper error")
        scal.data = np.float16(2) 
        # scal.data = np.complexfloating(4.5,2)   ## line 209

        try:
            scal.data = np.double(1.2)  
        except ValueError:
            print("Proper error")
      
    def TestAssignRandomData(self):
        rand_scal1 = mp.Scalar.create(self.__session, complex)
        rand_scal2 = mp.Scalar.create(self.__session, bool)
        rand_scal1.assign_random_data()
        rand_scal2.assign_random_data()    

    def TestCopyTo(self):
        scal = mp.Scalar.create(self.__session, int)
        scal.data = np.array([1])
        dest1 = mp.Scalar.create(self.__session, int)
        dest2 = mp.Scalar.create(self.__session, float)

        scal.copy_to(dest1)
        
        try:
            scal.copy_to(dest2)
        except TypeError:
            print("Proper error")


class TensorUnitTests:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestTensorData(self):
        t = mp.Tensor.create(self.__session, (1,1))
        try:    
            t.data = np.double(3.1)  ## line 648
        except TypeError:
            print("Proper error")

        try:
            t.data = np.ndarray([1, 2, 3])
        except ValueError:
            print("Proper error")
    
    def TestChangeShape(self):
        t = mp.Tensor.create(self.__session, (1,1))
        try:    
            t.change_shape(1)   
        except TypeError:
            print("Proper error")

    def TestTensorRandomData(self):
        tensor = mp.Tensor.create(self.__session, (1, 1, 1, 1)) 
        try:
            tensor.assign_random_data(covariance=True) 
        except ValueError:
            print("Proper error")

        tensor1 = mp.Tensor.create(self.__session, (1, 2, 3))
        try:
            tensor1.assign_random_data(covariance=True)   
        except ValueError:
            print("Proper error")

        tensor2 = mp.Tensor.create(self.__session, (3,3))
        tensor2.assign_random_data(covariance=True) 

    def TestCreateFromData(self):
        tensor = mp.Tensor.create_from_data(self.__session, [1,2,3,3])   

class ArrayUnitTests:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestArrayGetElement(self):
        arr = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        arr.get_element(-1) #3 line 1077
        try:
            arr.get_element(5)  ## 1080
        except ValueError:
            print("Proper error")

    def TestArraySetElement(self):
        arr = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        try:
            arr.set_element(-1, mp.Scalar.create(self.__session, int))   ## line 1111
        except ValueError:
            print("Proper error")

        try:
            arr.set_element(5, mp.Scalar.create(self.__session, int))   ## line 1114
        except ValueError:
            print("Proper error")

    def TestArrayNumElements(self):
        arr1 = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        # num = arr1.num_elements()   ## line 1124

    def TestCopyTo(self):
        a = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        dest1 = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        dest2 = mp.Array.create(self.__session, 5, mp.Scalar.create(self.__session, int))
        dest3 = mp.Array.create(self.__session, 4, mp.Tensor.create(self.__session, (1,1)))

        a.copy_to(dest1)
        
        try:
            a.copy_to(dest2)
        except ValueError:
            print("Proper error")

        try:
            a.copy_to(dest3)
        except TypeError:
            print("Proper error")

    def TestArrayToTensor(self):
        arr2 = mp.Array.create(self.__session, 6, mp.Scalar.create(self.__session, int))
        t = arr2.to_tensor()
        # assert t.mp_type == MPEnums.TENSOR        ## what type should this be

        arr3 = mp.Array.create(self.__session, 6, mp.Scalar.create(self.__session, bool))
        try:
            t = arr3.to_tensor()   ## line 1252
        except TypeError:
            print("Proper error")

def test_execute():
    st = ScalarUnitTests()   
    tt = TensorUnitTests()
    at = ArrayUnitTests()

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

test_execute()