import mindpype as mp
import numpy as np

class ScalarUnitTests:   ## limit methods to individual tests ex. TestingScalarType
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestScalarCreation(self):
        inScalar = mp.Scalar.create(self.__session, 'int')   ## line in containers, 65, hopefully also 275-278?
        scal = mp.Scalar.create(self.__session, float)   ## line 68,69
        random_scal = mp.Scalar.create(self.__session, int)
        try:
            bad_scalar = mp.Scalar.create(self.__session, 'double')  ## line 281
        except ValueError:
            print("Proper error")

        virtual = mp.Scalar.create_virtual(self.__session, 'int')   ## line 311-315
        try: 
            virtual1 = mp.Scalar.create_virtual(self.__session, 'double') ## line 317,318   .... might be mistake in lines 320,321
        except ValueError:
            print("Proper error")

        val_scalar = mp.Scalar.create_from_value(self.__session, 'hi')   ## line 356
        
    def TestScalarData(self):
        scal = mp.Scalar.create(self.__session, int)
        scal.data = np.array([1])   ## line 122
        scal.data = np.float16(2)   ## line 126-128
        # verify scal data is actually equal 
        scal.data = np.float16(4.5)   ## line 130
        # scal.data = np.complex64(2, 1)   ## line 131,132
        try:
            scal.data = np.double(1.2)   ## line 137
        except ValueError:
            print("Proper error")
      
    def TestAssignRandomData(self):
        rand_scal1 = mp.Scalar.create(self.__session, complex)
        rand_scal2 = mp.Scalar.create(self.__session, bool)
        rand_scal1.assign_random_data()
        rand_scal2.assign_random_data()    ## all lines 213,216,217

        # source_scal = mp.Scalar.create_from_source(self.__session, 'double', src)
        # source_scalar_good = mp.Scalar.create_from_source(self.__session, int, src)    ## lines 390-396


class TensorUnitTests:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestTensorData(self):
        t = mp.Tensor.create(self.__session, (1,1))
        t.data = True   ## line 528

    def TestTensorRandomData(self):
        tensor = mp.Tensor.create_from_data(self.__session, [1,2,3,4])  ## line 756
        # handle = mp.Tensor.create_from_handle(self.__session, (2, 3, 5, 1), src)   ## line 783
        try:
            tensor.assign_random_data(covariance=True)   ## line 651 + 655
        except ValueError:
            print("Proper error")

        tensor2 = mp.Tensor.create_from_data(self.__session, [3,3])
        try:
            tensor2.assign_random_data(covariance=True)   ## line 659
        except ValueError:
            print("Proper error")

        # handle3 = mp.Tensor.create_from_data(self.__session, [1,1])
        # handle3.shape = (1,1)
        # handle3.assign_random_data(covariance=True)   ## line 659


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

        assert t == None        ## why isn't this working


def test_execute():
    st = ScalarUnitTests()   
    tt = TensorUnitTests()
    at = ArrayUnitTests()

    st.TestAssignRandomData()
    st.TestScalarCreation()
    st.TestScalarData()

    tt.TestTensorData()
    tt.TestTensorRandomData()

    at.TestArrayNumElements()
    at.TestArraySetElement
    at.TestArrayToTensor()

test_execute()