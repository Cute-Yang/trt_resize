from mmdeploy.backend.tensorrt import TRTWrapper
# from mmdeploy.backend.tensorrt import create_trt_engine, save_trt_engine
import torch
import torch
from torch import nn
from torch.nn.functional import interpolate
import torch.onnx
import cv2
import numpy as np
import tensorrt as trt
import onnx
from typing import List, Dict, Any
import ctypes
import onnxruntime as ort


lib_path = "build/libtrt_dynamic_resize.so"
ctypes.CDLL(lib_path)


class DynamicTRTResize(torch.autograd.Function):
    def __init__(self) -> None:
        super().__init__()

    # create the symbolic,for export the onnx
    @staticmethod
    def symbolic(g: onnx.NodeProto, input, size_tensor, align_corners: bool = False):
        # the signature maybe op(name:str,*args,**kwargs) hah
        return g.op(
            "Test::DynamicTRTResize",  # the style is like cpp namespace
            input,
            size_tensor,
            align_corners_i=align_corners
        )

    # used in python
    @staticmethod
    def forward(g, input, size_tensor: torch.Tensor, align_corners: bool = False):
        """
        define the forward func
        """
        # here just add a tensor -> list process
        size = [size_tensor.size(-2), size_tensor.size(-1)]
        return interpolate(
            input,
            size=size,
            mode="bicubic",
            align_corners=align_corners
        )


class StrangeSuperResolutionNet(nn.Module):
    def __init__(self):
        super(StrangeSuperResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, size_tensor: torch.Tensor) -> torch.Tensor:
        # invoke this function!
        x = DynamicTRTResize.apply(x, size_tensor)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def create_trt_engine(onnx_file: str, dynamic_shapes, use_fp16: bool = False, n_profiles: int = 1, dynamic_name_keys: List[str] = [], save_path: str = None):
    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config.max_workspace_size = (1 << 30) * 6  # the max workspaced is 3G

    if use_fp16:
        config.flags == 1 << int(trt.BuilderFlag.FP16)
        print("we will use the half precision to inference!")

    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            for p_error in range(parser.num_errors):
                print(parser.get_error(p_error))
            raise RuntimeError(
                "fail to parse the onnx IR to network,maybe some symbols not support int tensorRT!try to use Plugin!")
        else:
            print("successfully parse the network!")

    profile_list: List[trt.IOptimizationProfile] = [builder.create_optimization_profile() for _ in range(n_profiles)]
    print("we will build {} optimization profiles...".format(n_profiles))

    if len(dynamic_shapes) == 1:
        dynamic_shapes = dynamic_shapes * n_profiles

    for i in range(n_profiles):
        # profile: trt.IOptimizationProfile = profile_list[i]
        dynamic_shape_dict: Dict[str, Dict[str, List[int]]] = dynamic_shapes[i]
        for key in dynamic_name_keys:
            if key not in dynamic_shape_dict:
                error_string = "the specify name_key {} is not in dynamic_shape_dict {}".format(
                    key,
                    dynamic_shape_dict
                )
                raise KeyError(error_string)
            min_shape = dynamic_shape_dict[key]["min"]
            opt_shape = dynamic_shape_dict[key]["opt"]
            max_shape = dynamic_shape_dict[key]["max"]
            print(min_shape, opt_shape, max_shape, key)
            profile_list[i].set_shape(key, min_shape, opt_shape, max_shape)
    for i in range(n_profiles):
        config.add_optimization_profile(profile_list[i])

    engine_string = builder.build_serialized_network(network, config)
    with open(save_path, "wb") as f:
        f.write(engine_string)


def save_trt_engine(engine, file: str):
    with open(file, "wb") as f:
        f.write(engine)


dynamic_shapes = {
    "input": {
        "min": [1, 3, 256, 256],
        "opt": [1, 3, 512, 512],
        "max": [1, 3, 1024, 1024]
    },
    "factor": {
        "min": [1, 1, 256, 256],
        "opt": [1, 1, 512, 512],
        "max": [1, 1, 1024, 1024]
    }
}
# create_trt_engine(
#     onnx_file="sources/srcnn3.onnx",
#     dynamic_name_keys=["input", "factor"],
#     save_path="sources/srcnn3.engine",
#     dynamic_shapes=[dynamic_shapes]
# )
# exit()
trt_model = TRTWrapper('sources/srcnn3.engine', ['output'])
factor = torch.rand([1, 1, 256, 256], dtype=torch.float)
x = torch.randn(1, 3, 768,768,dtype=torch.float32)
trt_output = trt_model.forward(dict(input=x.cuda(), factor=factor.cuda()))

model = StrangeSuperResolutionNet()
state_dict = torch.load("sources/srcnn.pth")["state_dict"]
for old_key in list(state_dict.keys()):
    new_key = ".".join(old_key.split(".")[1:])
    state_dict[new_key] = state_dict.pop(old_key)

model.load_state_dict(state_dict)
model.eval()
with torch.no_grad():
    torch_output = model(x, factor).detach().numpy()
print("tensorRT:",trt_output["output"].cpu().numpy()[:10])
print("torch",torch_output[:10])

assert np.allclose(trt_output['output'].cpu().numpy(), torch_output, rtol=1e-3, atol=1e-5)

