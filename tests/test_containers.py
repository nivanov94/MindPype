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
            val_scalar = mp.Scalar.create_from_value(self.__session, np.double(4.2))  ## line 436
        except TypeError:
            print("Proper error")
        
    def TestScalarData(self):
        scal = mp.Scalar.create(self.__session, int)
        scal.data = np.array([1])   ## line 197
        try:
            scal.data = np.array([1,2]) # line 199
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
        rand_scal2.assign_random_data()    ## all lines 213,216,217

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
            t.data = np.ndarray([np.double(3.1)])  ## line 648
        except TypeError:
            print("Proper error")

        try:
            t.data = np.ndarray([1, 2, 3])  ## line 667
        except ValueError:
            print("Proper error")
    
    def TestChangeShape(self):
        t = mp.Tensor.create(self.__session, (1,1))
        try:    
            t.change_shape(1)   ## line 702
        except TypeError:
            print("Proper error")

    def TestTensorRandomData(self):
        tensor = mp.Tensor.create(self.__session, (1, 1, 1, 1)) 
        try:
            tensor.assign_random_data(covariance=True)   ## line 802
        except ValueError:
            print("Proper error")

        tensor1 = mp.Tensor.create(self.__session, (1, 2, 3))
        try:
            tensor1.assign_random_data(covariance=True)   ## line 808
        except ValueError:
            print("Proper error")

        tensor2 = mp.Tensor.create(self.__session, (3,3))
        tensor2.assign_random_data(covariance=True)   ## line 814

    def TestCreateFromData(self):
        tensor = mp.Tensor.create_from_data(self.__session, [1,2,3,3])   ## line 934

class ArrayUnitTests:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)
        
    def TestArraySetElement(self):
        arr = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        try:
            arr.set_element(5, 6)   ## line 952
        except ValueError:
            print("Proper error")

    def TestArrayNumElements(self):
        arr1 = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))
        # num = arr1.num_elements()   ## line 965

    def TestArrayToTensor(self):
        arr2 = mp.Array.create(self.__session, 6, mp.Scalar.create(self.__session, int))
        t = arr2.to_tensor()
        print(type(t))
        # assert t.mp_type == MPEnums.TENSOR        ## what type should this be

        arr3 = mp.Array.create(self.__session, 6, mp.Scalar.create(self.__session, bool))
        t = arr3.to_tensor()

        # assert t == None        ## why isn't this working


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

    at.TestArrayNumElements()
    at.TestArraySetElement
    at.TestArrayToTensor()

test_execute()