import mindpype as mp
import numpy as np
import pytest

class SlopeKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestSlopeKernelExecution(self, raw_data, axis, test_input_error=False, test_output_error=False, Fs=1):
        if test_input_error:
            inTensor = mp.Scalar.create_from_value(self.__session, "test")
            output_sz = (1,1)
        else:
            inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
            # calculate outTensor shape
            pos_axis = axis if axis >= 0 else len(inTensor.shape) + axis
            output_sz = [d for i_d, d in enumerate(inTensor.shape) if i_d != pos_axis]
            output_sz.append(1)
        if test_output_error:
            outTensor = mp.Scalar.create(self.__session, int)
        else:
            outTensor = mp.Tensor.create(self.__session, (output_sz))
        tensor_test_node = mp.kernels.SlopeKernel.add_to_graph(self.__graph,inTensor,outTensor,Fs=Fs,axis=axis)
        #tensor_test_node = mp.kernels.SlopeKernel.add_to_graph(self.__graph,inTensor,outTensor,Fs=1,axis=0)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return (inTensor, outTensor.data)
    
def test_execute():
    
    KernelExecutionUnitTest_Object = SlopeKernelUnitTest()
    np.random.seed(44)
    raw_data = np.random.randn(2,2)
    #axis = 0
    axis = -1
    res = KernelExecutionUnitTest_Object.TestSlopeKernelExecution(raw_data, axis)
    
    # manual calculation
    Fs = 1
    Ns = res[0].shape[axis]
    X = np.linspace(0, Ns/Fs, Ns)
    Y = np.moveaxis(raw_data, axis, -1)
    X -= np.mean(X)
    Y -= np.mean(Y, axis=-1, keepdims=True)
    output = np.expand_dims(Y.dot(X) / X.dot(X), axis=-1)
    
    assert (res[1] == output).all()
    
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestSlopeKernelExecution(raw_data, axis, test_input_error=True)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = SlopeKernelUnitTest()
    with pytest.raises(TypeError) as e_info:
        res = KernelExecutionUnitTest_Object.TestSlopeKernelExecution(raw_data, axis, test_output_error=True)
    del KernelExecutionUnitTest_Object
    
    KernelExecutionUnitTest_Object = SlopeKernelUnitTest()
    with pytest.raises(ValueError) as e_info:
        res = KernelExecutionUnitTest_Object.TestSlopeKernelExecution(raw_data, axis, Fs=0)
    del KernelExecutionUnitTest_Object
