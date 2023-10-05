#pragma once

#include "NvInferRuntime.h"
#include "NvInferVersion.h"
#include <string>
#include <vector>

namespace guanyang {
typedef enum {
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;
#if NV_TENSORRT_MAJOR > 7
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOTRT_NOEXCEPT
#endif

class TRTPluginBase : public nvinfer1::IPluginV2DynamicExt {
protected:
    const std::string mLayerName;
    std::string mNamespace;

public:
    // we can create the plugin by name! hah
    TRTPluginBase(const std::string &name) : mLayerName(name) {
    }

    const char *getPluginVersion() const TRT_NOEXCEPT override {
        return "1";
    }

    int initialize() TRT_NOEXCEPT override {
        return STATUS_SUCCESS;
    }

    void terminate() TRT_NOEXCEPT override {
    }

    void destroy() TRT_NOEXCEPT override {
    }
    void setPluginNamespace(const char *pluginNameSpace) TRT_NOEXCEPT override {
        mNamespace = pluginNameSpace;
    }

    const char *getPluginNamespace() const TRT_NOEXCEPT override {
        return mNamespace.c_str();
    }

    virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                 const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) TRT_NOEXCEPT override {
    }

    virtual size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                    const nvinfer1::PluginTensorDesc *ouputs, int nbOutputs) const TRT_NOEXCEPT override {
        return 0;
    }

    virtual void attachToContext(cudnnContext *cudnnCtx, cublasContext *cublasCtx,
                                 nvinfer1::IGpuAllocator *gpuAlloc) TRT_NOEXCEPT override {
    }

    virtual void detachFromContext() TRT_NOEXCEPT override {
    }
};

class TRTPluginCreatorBase : public nvinfer1::IPluginCreator {
protected:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;

public:
    const char *getPluginVersion() const TRT_NOEXCEPT override {
        return "1";
    }

    const nvinfer1::PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override {
        return &mFC;
    }

    void setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override {
        mNamespace = pluginNamespace;
    }

    const char *getPluginNamespace() const TRT_NOEXCEPT override {
        return mNamespace.c_str();
    }
};
} // namespace guanyang
