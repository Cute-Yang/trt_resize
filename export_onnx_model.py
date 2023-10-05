"""
OHHXRuntime
TensorRT
ncnn
OpenVINO
PPLNN
"""

# here we will export a onnx model with cutomed ops


import torch
from torch import nn
from torch.nn.functional import interpolate
import torch.onnx
import cv2
import numpy as np
import os
import requests
import onnx

urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
        'https://raw.githubusercontent.com/open-mmlab/mmediting/master/tests/data/face/000001.png']
names = ['srcnn.pth', 'face.png']


def download_source(save_dir: str, url: str, save_name: str) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir, save_name)
    if os.path.exists(save_file):
        return
    resp = requests.get(url=url)
    content = resp.content
    with open(save_file, "wb") as f:
        f.write(content)
    print("donwload '{}'".format(save_file))

# now we construct the self onnx ops


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


def init_torch_model():
    model = StrangeSuperResolutionNet()
    state_dict = torch.load("sources/srcnn.pth")["state_dict"]
    for old_key in list(state_dict.keys()):
        new_key = ".".join(old_key.split(".")[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    model.load_state_dict(state_dict)
    factor = torch.rand([1, 1, 512, 512], dtype=torch.float32)
    input_image = cv2.imread("sources/face.png").astype(np.float32)
    input_image = np.transpose(input_image, axes=[2, 0, 1])
    input_image = np.expand_dims(input_image, 0)

    torch_output = model(torch.from_numpy(input_image), factor).detach().numpy()
    torch_output = np.squeeze(torch_output, 0)
    torch_output = np.clip(torch_output, 0, 255)
    torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

    cv2.imwrite("face_torch.png", torch_output)

    x = torch.randn(1, 3, 256, 256)

    #这个axes的命名要注意，如果不同key中axis的命名相同，那么tensorRT会要求她们有相同的值，一定要根据自己情况定义
    dynamic_axes = {
        "input": {
            0: "batch_size",
            2: "height",
            3: "width"
        },
        "factor": {
            0: "batch_size1",
            2: "height1",
            3: "width1"
        },
        "output": {
            0: "batch_size2",
            2: "height2",
            3: "width2"
        }
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (x, factor),
            "sources/srcnn3.onnx",
            opset_version=11,
            input_names=["input", "factor"],
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )


if __name__ == "__main__":
    save_dir = "sources"
    for url, name in zip(urls, names):
        download_source(
            save_dir=save_dir,
            url=url,
            save_name=name
        )
    init_torch_model()
