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
REGISTER_OP("EHuberLoss")
    .Input("pred: float")
    .Input("real: float")
    .Input("delta: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times: int")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      c->set_output(0, c->input(0));
      return Status::OK(); });

REGISTER_OP("EHuberLossGrad")
    .Input("grad: float")
    .Input("pred: float")
    .Input("real: float")
    .Input("delta: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times: int")
    .Output("output: float")
    .Output("grad_r: float");

class EHuberLossOp : public OpKernel
{
public:
    explicit EHuberLossOp(OpKernelConstruction *context) : OpKernel(context)
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
        const TensorShape &input_shape = input.shape();
        const Tensor &real = context->input(1);
        auto real_flat = real.flat<float>();
        const TensorShape &real_shape = real.shape();
        const Tensor &delta = context->input(2);
        auto delta_flat = delta.flat<float>();

        TensorShape output_shape;
        output_shape.AddDim(input_shape.dim_size(0));
        output_shape.AddDim(input_shape.dim_size(1));

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();
        int N = input_flat.size() / times_;
        int C = delta_flat.size();
        int B = input_shape.dim_size(0) / times_;

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *pred, float *real, float *delta, int N, int C, int B, float *output);
        dlerror();
        function huberloss_kernel = (function)dlsym(lib, "huberloss");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of huberloss failed: ", dlsym_error));
        huberloss_kernel(eid_, (float *)input_flat.data(), (float *)real_flat.data(), (float *)delta_flat.data(), N, C, B, (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EHuberLoss").Device(DEVICE_CPU), EHuberLossOp);

class EHuberLossGradOp : public OpKernel
{
public:
    explicit EHuberLossGradOp(OpKernelConstruction *context) : OpKernel(context)
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
        const Tensor &real = context->input(2);
        const Tensor &delta = context->input(3);
        auto input_flat = input.flat<float>();
        auto real_flat = real.flat<float>();
        auto grad_flat = grad.flat<float>();
        auto delta_flat = delta.flat<float>();

        const TensorShape &input_shape = input.shape();

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        auto output_flat = output->flat<float>();

        Tensor *grad_r = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, real.shape(), &grad_r));
        auto grad_r_flat = grad_r->flat<float>();

        int N = input_flat.size() / times_;
        int C = delta_flat.size();
        int B = input_shape.dim_size(0) / times_;

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *pred, float *real, float *delta, float *grad, int N, int C, int B, float *output, float *grad_r);
        dlerror();
        function huberloss_grad_kernel = (function)dlsym(lib, "huberloss_grad");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of huberloss_grad failed: ", dlsym_error));
        huberloss_grad_kernel(eid_, (float *)input_flat.data(), (float *)real_flat.data(), (float *)delta_flat.data(), (float *)grad_flat.data(), N, C, B, (float *)output_flat.data(), (float *)grad_r_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EHuberLossGrad").Device(DEVICE_CPU), EHuberLossGradOp);