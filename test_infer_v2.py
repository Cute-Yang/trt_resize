import tensorrt as trt
import numpy as np
from cuda import cudart
import ctypes
lib_path = "build/libtrt_dynamic_resize.so"
ctypes.CDLL(lib_path)

def test_trt(x:np.ndarray,factor:np.ndarray,engine_file:str) -> np.ndarray:
    with open(engine_file, "rb") as f:
        engine_string = f.read()

    logger = trt.Logger(trt.Logger.VERBOSE)
    another_logger = trt.Logger(trt.Logger.VERBOSE)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_string)
    if engine is None:
        raise RuntimeError("failed to deserialize the trt engine '{}'".format(
            engine_file
        ))
    print("successfully deserialize the trt engine!")
    context = engine.create_execution_context()
    for i in range(3):
        print("idx:{} name:{} type:{}".format(
            i, engine.get_binding_name(i),
            engine.get_binding_dtype(i)
        ))
    
    x_shape = x.shape
    factor_shape = factor.shape
    
    x_ptr = cudart.cudaMalloc(x.nbytes)[1]
    factor_ptr = cudart.cudaMalloc(factor.nbytes)[1]
    cudart.cudaMemcpy(x_ptr,x.ctypes.data,x.nbytes,cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    cudart.cudaMemcpy(factor_ptr,factor.ctypes.data,factor.nbytes,cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.set_binding_shape(0,x_shape)
    context.set_binding_shape(1,factor_shape)

    output_shape = context.get_binding_shape(2)

    output_array = np.zeros(shape=output_shape,dtype=np.float32)

    output_ptr = cudart.cudaMalloc(output_array.nbytes)[1]

    context.execute_v2([x_ptr,factor_ptr,output_ptr])

    cudart.cudaMemcpy(output_array.ctypes.data,output_ptr,output_array.nbytes,cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print(output_array)


if __name__ == "__main__":
    engine_file = "sources/srcnn3.engine"
    x = np.ones((1,3,768,768),dtype=np.float32)
    factor = np.random.randn(1,1,768,768)

    test_trt(x,factor=factor,engine_file=engine_file)

    

    