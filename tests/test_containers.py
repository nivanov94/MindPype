import mindpype as mp
import numpy as np

class ContainersUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    # def TestExtraKernels(self, raw_data, labels_data):
    #     inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
    #     inScalar = mp.Scalar.create(self.__session, bool)  
    #     scal = mp.Scalar.create(self.__session, complex)  
    #     mid = mp.Scalar.create(self.__session, int)
    #     node1 = mp.kernels.FeatureSelectionKernel.add_to_graph(self.__session, inScalar, mid, labels=labels_data)
    #     node2 = mp.kernels.PadKernel.add_to_graph(self.__session, mid, scal)     ## some kernel testing here
    #     return [node1.mp_type, node2.mp_type]
    

    def TestingContainersScalars(self):
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
        

        inScalar.data = np.array([1])   ## line 122
        scal.data = np.float16(2)   ## line 126-128
        # verify scal data is actually equal 
        scal.data = np.float16(4.5)   ## line 130
        # scal.data = np.complex64(2, 1)   ## line 131,132
        try:
            random_scal.data = np.double(1.2)   ## line 137
        except ValueError:
            print("Proper error")
      

        rand_scal1 = mp.Scalar.create(self.__session, complex)
        rand_scal2 = mp.Scalar.create(self.__session, bool)
        rand_scal1.assign_random_data()
        rand_scal2.assign_random_data()    ## all lines 213,216,217

        val_scalar = mp.Scalar.create_from_value(self.__session, 'hi')   ## line 356
        # source_scal = mp.Scalar.create_from_source(self.__session, 'double', src)
        # source_scalar_good = mp.Scalar.create_from_source(self.__session, int, src)    ## lines 390-396


    def TestingContainersTensors(self):
        tensor = mp.Tensor.create_from_data(self.__session, [1,2,3])  ## line 756
        # handle = mp.Tensor.create_from_handle(self.__session, (2, 3, 5, 1), src)   ## line 783
        try:
            tensor.assign_random_data(covariance=True)   ## line 651
        except ValueError:
            print("Proper error")

        handle2 = mp.Tensor.create_from_data(self.__session, [3,3,3])
        try:
            handle2.assign_random_data(covariance=True)   ## line 655
        except ValueError:
            print("Proper error")

        handle3 = mp.Tensor.create_from_data(self.__session, [1,1])
        handle3.shape = (1,1)
        handle3.assign_random_data(covariance=True)   ## line 659

        # virtual = mp.Tensor.create_virtual(self.__session)
        # val = virtual._validate_data()    ## line 831
        # val2 = tensor._validate_data()    ## line 834
        # val3 = handle3._validate_data()    ## line 837


    def TestingContainersArray(self):
        arr = mp.Array.create(self.__session, 4, mp.Scalar.create(self.__session, int))

        try:
            arr.set_element(5, 6)   ## line 952
        except ValueError:
            print("Proper error")

        # num = arr.num_elements()   ## line 965


def test_execute():
    test = ContainersUnitTest()
    test.TestingContainersScalars()
    test.TestingContainersTensors()
    test.TestingContainersArray()

test_execute()