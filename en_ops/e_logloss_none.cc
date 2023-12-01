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

REGISTER_OP("ELoglossNone")
    .Input("pred: float")
    .Input("real: float")
    .Input("weights: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("loss: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      c->set_output(0, c->input(0));
      return Status::OK(); });

REGISTER_OP("ELoglossNoneGrad")
    .Input("grad: float")
    .Input("pred: float")
    .Input("real: float")
    .Input("weights: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float")
    .Output("grad_w: float")
    .Output("grad_r: float");

class ELoglossNoneOp : public OpKernel
{
public:
    explicit ELoglossNoneOp(OpKernelConstruction *context) : OpKernel(context)
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
        const Tensor &weights = context->input(2);
        const TensorShape &weights_shape = weights.shape();
        auto weights_flat = weights.flat<float>();

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        auto output_flat = output->flat<float>();

        const int N = input_flat.size() / times_;

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, int N, float *label, float *weight, float *output);
        dlerror();
        function logloss_none_kernel = (function)dlsym(lib, "logloss_none");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of logloss_none failed: ", dlsym_error));
        logloss_none_kernel(eid_, (float *)input_flat.data(), N, (float *)real_flat.data(), (float *)weights_flat.data(), (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("ELoglossNone").Device(DEVICE_CPU), ELoglossNoneOp);

class ELoglossNoneGradOp : public OpKernel
{
public:
    explicit ELoglossNoneGradOp(OpKernelConstruction *context) : OpKernel(context)
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
        const Tensor &weights = context->input(3);
        auto input_flat = input.flat<float>();
        auto real_flat = real.flat<float>();
        auto grad_flat = grad.flat<float>();
        auto weights_flat = weights.flat<float>();

        const TensorShape &input_shape = input.shape();
        const TensorShape &real_shape = real.shape();
        const TensorShape &weights_shape = weights.shape();
        // create output tensor
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        auto output_flat = output->flat<float>();

        Tensor *grad_w = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_w));
        auto grad_w_flat = grad_w->flat<float>();

        Tensor *grad_r = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2, real_shape, &grad_r));
        auto grad_r_flat = grad_r->flat<float>();

        const int N = input_flat.size() / times_;

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, float *grad, int N, float *label, float *weight, float *grad_x, float *grad_w, float *grad_l);
        dlerror();
        function logloss_none_grad_kernel = (function)dlsym(lib, "logloss_none_grad");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of logloss_none_grad failed: ", dlsym_error));
        logloss_none_grad_kernel(eid_, (float *)input_flat.data(), (float *)grad_flat.data(), N, (float *)real_flat.data(), (float *)weights_flat.data(), (float *)output_flat.data(), (float *)grad_w_flat.data(), (float *)grad_r_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("ELoglossNoneGrad").Device(DEVICE_CPU), ELoglossNoneGradOp);