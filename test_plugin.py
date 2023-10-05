import tensorrt as trt
import ctypes
# from mmdeploy.backend.tensorrt import load_tensorrt_plugin
# load_tensorrt_plugin()
lib_path = "build/libtrt_dynamic_resize.so"
ctypes.CDLL(lib_path)

def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]

if __name__ == "__main__":
    print(get_plugin_names())

