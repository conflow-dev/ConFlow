#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <dlfcn.h>

using namespace tensorflow;
using namespace shape_inference;
using namespace std;

REGISTER_OP("EEmbInit")
    .Attr("input_dim:int")
    .Attr("output_dim:int")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      int input_dim;
      int output_dim;
      TF_RETURN_IF_ERROR(c->GetAttr("input_dim", &input_dim));
      TF_RETURN_IF_ERROR(c->GetAttr("output_dim", &output_dim));
      c->set_output(0, c->MakeShape({input_dim,output_dim}));
      return Status::OK(); });

class EEmbInitOp : public OpKernel
{
public:
    explicit EEmbInitOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_dim", &input_dim_));
        OP_REQUIRES_OK(context, context->GetAttr("output_dim", &output_dim_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        lib = dlopen("/home/zhangyan/jhrsgx/privacy_test/privacy_tf/sgx_tf_ops/sgx.so", RTLD_LAZY);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load sgx.so!"));
    }
    void Compute(OpKernelContext *context) override
    {
        Tensor *output = NULL;
        TensorShape output_shape;
        output_shape.AddDim(input_dim_);
        output_shape.AddDim(output_dim_);
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();
        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *output, int M, int C);
        dlerror();
        function emb_init_kernel = (function)dlsym(lib, "emb_init");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of emb_init failed: ", dlsym_error));
        emb_init_kernel(eid_, (float *)output_flat.data(), input_dim_, output_dim_);
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 input_dim_;
    int64 output_dim_;
};
REGISTER_KERNEL_BUILDER(Name("EEmbInit").Device(DEVICE_CPU), EEmbInitOp);
