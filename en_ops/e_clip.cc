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

REGISTER_OP("EClip")
    .Input("input: float")
    .Input("down: float")
    .Input("up: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      c->set_output(0, c->input(0));
      return Status::OK(); });

REGISTER_OP("EClipGrad")
    .Input("grad: float")
    .Input("input: float")
    .Input("down: float")
    .Input("up: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float");

using namespace tensorflow;

class EClipOp : public OpKernel
{
public:
    explicit EClipOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        OP_REQUIRES_OK(context, context->GetAttr("times", &times_));
        lib = dlopen("/home/zhangyan/jhrsgx/privacy_test/privacy_tf/sgx_tf_ops/sgx.so", RTLD_LAZY);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load sgx.so!"));
    }
    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        auto input_flat = input.flat<float>();

        const Tensor &down = context->input(1);
        auto down_flat = down.flat<float>();

        const Tensor &up = context->input(2);
        auto up_flat = up.flat<float>();

        const TensorShape &input_shape = input.shape();

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        auto output_flat = output->flat<float>();
        int N = input_flat.size() / times_;
        int n1 = 1;
        int n2 = 1;
        if (down_flat.size() > 1)
        {
            n1 = down_flat.size() / times_;
        }
        if (up_flat.size() > 1)
        {
            n2 = up_flat.size() / times_;
        }

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, float *down, float *up, int n1, int n2, int N, float *output);
        dlerror();
        function clip_kernel = (function)dlsym(lib, "clip");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of clip failed: ", dlsym_error));
        clip_kernel(eid_, (float *)input_flat.data(), (float *)down_flat.data(), (float *)up_flat.data(), n1, n2, N, (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EClip").Device(DEVICE_CPU), EClipOp);

class EClipGradOp : public OpKernel
{
public:
    explicit EClipGradOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        OP_REQUIRES_OK(context, context->GetAttr("times", &times_));
        lib = dlopen("/home/zhangyan/jhrsgx/privacy_test/privacy_tf/sgx_tf_ops/sgx.so", RTLD_LAZY);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load sgx.so!"));
    }
    void Compute(OpKernelContext *context) override
    {

        const Tensor &grad = context->input(0);
        const Tensor &input = context->input(1);

        const Tensor &down = context->input(2);
        auto down_flat = down.flat<float>();

        const Tensor &up = context->input(3);
        auto up_flat = up.flat<float>();

        auto grad_flat = grad.flat<float>();
        auto input_flat = input.flat<float>();

        const TensorShape &input_shape = input.shape();

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        auto output_flat = output->flat<float>();

        int N = input_flat.size() / times_;

        int n1 = 1;
        int n2 = 1;
        if (down_flat.size() > 1)
        {
            n1 = down_flat.size() / times_;
        }
        if (up_flat.size() > 1)
        {
            n2 = up_flat.size() / times_;
        }

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, float *grad, float *down, float *up, int n1, int n2, int N, float *output);
        dlerror();
        function clip_grad_kernel = (function)dlsym(lib, "clip_grad");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of clip_grad failed: ", dlsym_error));
        clip_grad_kernel(eid_, (float *)input_flat.data(), (float *)grad_flat.data(), (float *)down_flat.data(), (float *)up_flat.data(), n1, n2, N, (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EClipGrad").Device(DEVICE_CPU), EClipGradOp);