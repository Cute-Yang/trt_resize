#include "trt_dynamic_resize_kernel.hpp"
#include "trt_dynamic_resize.hpp"
#include "trt_plugin_base.hpp"
#include "trt_serialize.hpp"
#include <cstdint>

namespace guanyang {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"DynamicTRTResize"};

DynamicTRTResize::DynamicTRTResize(const std::string &name, bool align_corners) :
    TRTPluginBase(name), mAlignCorners(align_corners) {
    printf("create the DynamicTRTResize plugin,the brownfox jumps over the lazydog!\n");
}

DynamicTRTResize::DynamicTRTResize(const std::string name, const void *data, size_t lenght) : TRTPluginBase(name) {
    deserialize_value(&data, &lenght, &mAlignCorners);
}

nvinfer1::IPluginV2DynamicExt *DynamicTRTResize::clone() const TRT_NOEXCEPT {
    DynamicTRTResize *plugin = new DynamicTRTResize(mLayerName, mAlignCorners);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

// support multi inputs and multi outputs
nvinfer1::DimsExprs DynamicTRTResize::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                                          nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    // the inputs porivde the batch_size and channles,the factor provide the resized heigh and width!
    // input two tensor,,the layer is for shape infer only
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[0].d[1];
    ret.d[2] = inputs[1].d[2];
    ret.d[3] = inputs[1].d[3];
    return ret;
}

bool DynamicTRTResize::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs,
                                                 int nbOutputs) TRT_NOEXCEPT {
    if (pos == 0) {
        return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT && ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
    } else {
        return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
    }
}

// here we do nothing!
void DynamicTRTResize::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
                                       const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) TRT_NOEXCEPT {
}

// 这个workspace是用来干啥的
size_t DynamicTRTResize::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const TRT_NOEXCEPT {
    return 0;
}

int DynamicTRTResize::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                              const void *const *inputs, void *const *outputs, void *workspace,
                              cudaStream_t stream) TRT_NOEXCEPT {
    int batch = inputDesc[0].dims.d[0];
    int channles = inputDesc[0].dims.d[1];
    int height = inputDesc[0].dims.d[2];
    int width = inputDesc[0].dims.d[3];

    int new_height = outputDesc[0].dims.d[2];
    int new_width = outputDesc[0].dims.d[3];
    printf("previous size(%d,%d) new size(%d,%d)\n",width,height,new_height,new_width);

    const void *x = inputs[0];
    void *output = outputs[0];
    
    float* host_x = new float[batch * channles * height * width];
    cudaMemcpy(host_x,x,batch * channles * height *width * sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<30;i++) {
        printf("idx=%d value=%f\n",i,host_x[i]);
    }


    auto data_type = inputDesc[0].type;
    switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
        bicubic_interpolate<float>((float *)x, (float *)output, batch, channles, height, width, new_height, new_width,
                                   mAlignCorners, stream);
        printf("fuck you!\n");
        break;
    default: return 1; break;
    }
    cudaError_t err = cudaGetLastError();
    if(err) {
        printf("CUDA Error: %s\n",cudaGetErrorString(err));
    }

    float* host_output = new float[batch * channles * new_height * new_width];
    cudaMemcpy(host_output,output,batch * channles * new_height *new_width * sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<30;i++) {
        printf("idx=%d value=%f\n",i,host_output[i]);
    }

    
    return 0;
}

nvinfer1::DataType DynamicTRTResize::getOutputDataType(int idnex, const nvinfer1::DataType *inputTypes,
                                                       int nbInputs) const TRT_NOEXCEPT {
    return inputTypes[0];
}

const char *DynamicTRTResize::getPluginType() const TRT_NOEXCEPT {
    return PLUGIN_NAME;
}

const char *DynamicTRTResize::getPluginVersion() const TRT_NOEXCEPT {
    return PLUGIN_VERSION;
}

int DynamicTRTResize::getNbOutputs() const TRT_NOEXCEPT {
    return 1;
}

size_t DynamicTRTResize::getSerializationSize() const TRT_NOEXCEPT {
    return sizeof(mAlignCorners);
}

void DynamicTRTResize::serialize(void *buffer) const TRT_NOEXCEPT {
    serialize_value(&buffer, mAlignCorners);
}

DynamicTRTResizeCreator::DynamicTRTResizeCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("align_corners"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *DynamicTRTResizeCreator::getPluginName() const TRT_NOEXCEPT {
    return PLUGIN_NAME;
}

const char *DynamicTRTResizeCreator::getPluginVersion() const TRT_NOEXCEPT {
    return PLUGIN_VERSION;
}

// 这个函数是从所有的plugin进行查找?
nvinfer1::IPluginV2 *DynamicTRTResizeCreator::createPlugin(const char *name,
                                                           const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
    nvinfer1::Dims size{2, {1, 1}};
    bool align_corners{true};
    for (int i = 0; i < fc->nbFields; ++i) {
        if (fc->fields[i].data == nullptr) { continue; }
        std::string field_name{fc->fields[i].name};
        if (field_name.compare("align_corners") == 0) { align_corners = static_cast<const int *>(fc->fields[i].data)[0]; }
    }
    DynamicTRTResize *plugin = new DynamicTRTResize(name, align_corners);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

nvinfer1::IPluginV2 *DynamicTRTResizeCreator::deserializePlugin(const char *name, const void *serialData,
                                                                size_t serialLength) TRT_NOEXCEPT {
    DynamicTRTResize *plugin = new DynamicTRTResize(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

// registry the plugin
REGISTER_TENSORRT_PLUGIN(DynamicTRTResizeCreator);

} // namespace guanyang