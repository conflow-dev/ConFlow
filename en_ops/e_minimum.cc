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

REGISTER_OP("EMinimum")
    .Input("input1: float")
    .Input("input2: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      c->set_output(0, c->input(0));
      return Status::OK(); });

REGISTER_OP("EMinimumGrad")
    .Input("grad: float")
    .Input("input1: float")
    .Input("input2: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output1: float")
    .Output("output2: float");

class EMinimumOp : public OpKernel
{
public:
    explicit EMinimumOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        OP_REQUIRES_OK(context, context->GetAttr("times", &times_));
        lib = dlopen("/home/zhangyan/jhrsgx/privacy_test/privacy_tf/sgx_tf_ops/sgx.so", RTLD_LAZY);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load sgx.so!"));
    }
    void Compute(OpKernelContext *context) override
    {
        const Tensor &input1 = context->input(0);
        auto input1_flat = input1.flat<float>();

        const Tensor &input2 = context->input(1);
        auto input2_flat = input2.flat<float>();

        const TensorShape &input_shape = input1.shape();

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        auto output_flat = output->flat<float>();
        int N = input1_flat.size() / times_;

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input1, float *input2, int N, float *output);
        dlerror();
        function minimum_kernel = (function)dlsym(lib, "minimum");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of minimum failed: ", dlsym_error));
        minimum_kernel(eid_, (float *)input1_flat.data(), (float *)input2_flat.data(), N, (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EMinimum").Device(DEVICE_CPU), EMinimumOp);

class EMinimumGradOp : public OpKernel
{
public:
    explicit EMinimumGradOp(OpKernelConstruction *context) : OpKernel(context)
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
        const Tensor &input1 = context->input(1);
        auto input1_flat = input1.flat<float>();

        const Tensor &input2 = context->input(2);
        auto input2_flat = input2.flat<float>();

        auto grad_flat = grad.flat<float>();

        const TensorShape &input_shape = input1.shape();

        Tensor *output1 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output1));
        auto output1_flat = output1->flat<float>();

        Tensor *output2 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, input_shape, &output2));
        auto output2_flat = output2->flat<float>();

        const int N = input1_flat.size() / times_;

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input1, float *input2, float *grad, int N, float *output1, float *output2);
        dlerror();
        function minimum_grad_kernel = (function)dlsym(lib, "minimum_grad");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of minimum_grad failed: ", dlsym_error));
        minimum_grad_kernel(eid_, (float *)input1_flat.data(), (float *)input2_flat.data(), (float *)grad_flat.data(), N, (float *)output1_flat.data(), (float *)output2_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EMinimumGrad").Device(DEVICE_CPU), EMinimumGradOp);