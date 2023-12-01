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

REGISTER_OP("ELogprob")
    .Input("input: float")
    .Input("mu: float")
    .Input("sigma: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      c->set_output(0, c->input(0));
      return Status::OK(); });

REGISTER_OP("ELogprobGrad")
    .Input("grad: float")
    .Input("input: float")
    .Input("mu: float")
    .Input("sigma: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float");

class ELogprobOp : public OpKernel
{
public:
    explicit ELogprobOp(OpKernelConstruction *context) : OpKernel(context)
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

        const Tensor &mu = context->input(1);
        auto mu_flat = mu.flat<float>();

        const Tensor &sigma = context->input(2);
        auto sigma_flat = sigma.flat<float>();

        const TensorShape &input_shape = input.shape();

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        auto output_flat = output->flat<float>();
        int N = input_flat.size() / times_;

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, float *mu, float *sigma, int N, float *output);
        dlerror();
        function logprob_kernel = (function)dlsym(lib, "logprob");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of logprob failed: ", dlsym_error));
        logprob_kernel(eid_, (float *)input_flat.data(), (float *)mu_flat.data(), (float *)sigma_flat.data(), N, (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("ELogprob").Device(DEVICE_CPU), ELogprobOp);

class ELogprobGradOp : public OpKernel
{
public:
    explicit ELogprobGradOp(OpKernelConstruction *context) : OpKernel(context)
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
        auto grad_flat = grad.flat<float>();

        const Tensor &input = context->input(1);
        auto input_flat = input.flat<float>();

        const Tensor &mu = context->input(2);
        auto mu_flat = mu.flat<float>();

        const Tensor &sigma = context->input(3);
        auto sigma_flat = sigma.flat<float>();

        const TensorShape &input_shape = input.shape();

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        auto output_flat = output->flat<float>();
        int N = input_flat.size() / times_;

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, float *grad, float *mu, float *sigma, int N, float *output);
        dlerror();
        function logprob_grad_kernel = (function)dlsym(lib, "logprob_grad");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of logprob_grad failed: ", dlsym_error));
        logprob_grad_kernel(eid_, (float *)input_flat.data(), (float *)grad_flat.data(), (float *)mu_flat.data(), (float *)sigma_flat.data(), N, (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("ELogprobGrad").Device(DEVICE_CPU), ELogprobGradOp);