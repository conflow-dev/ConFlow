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

REGISTER_OP("EMoments")
    .Input("input: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("mean: float")
    .Output("var: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));
      int64 N, M, L;
      N = c->Value(c->Dim(input_shape, 0));
      M = c->Value(c->Dim(input_shape, 1));
      L = c->Value(c->Dim(input_shape, 2));
      c->set_output(0, c->MakeShape({N, M, 1}));
      c->set_output(1, c->MakeShape({N, M, 1}));
      return Status::OK(); });

REGISTER_OP("EMomentsGrad")
    .Input("grad1: float")
    .Input("grad2: float")
    .Input("input: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float");

class EMomentsOp : public OpKernel
{
public:
    explicit EMomentsOp(OpKernelConstruction *context) : OpKernel(context)
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

        TensorShape output_shape;
        output_shape.AddDim(input_shape.dim_size(0));
        output_shape.AddDim(input_shape.dim_size(1));
        output_shape.AddDim(1);

        Tensor *mean = NULL;
        Tensor *var = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &mean));
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &var));
        auto mean_flat = mean->flat<float>();
        auto var_flat = var->flat<float>();

        int N = input_flat.size() / times_;
        int M = input_shape.dim_size(0) / times_;
        int C = input_shape.dim_size(1);
        int L = input_shape.dim_size(2);

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, int N, int M, int C, int L, float *mean, float *var);
        dlerror();
        function moments_kernel = (function)dlsym(lib, "moments");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of moments failed: ", dlsym_error));
        moments_kernel(eid_, (float *)input_flat.data(), N, M, C, L, (float *)mean_flat.data(), (float *)var_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EMoments").Device(DEVICE_CPU), EMomentsOp);

class EMomentsGradOp : public OpKernel
{
public:
    explicit EMomentsGradOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        OP_REQUIRES_OK(context, context->GetAttr("times", &times_));
        lib = dlopen("/home/zhangyan/jhrsgx/privacy_test/privacy_tf/sgx_tf_ops/sgx.so", RTLD_LAZY);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load sgx.so!"));
    }
    void Compute(OpKernelContext *context) override
    {

        const Tensor &grad1 = context->input(0);
        auto grad1_flat = grad1.flat<float>();
        const Tensor &grad2 = context->input(1);
        auto grad2_flat = grad2.flat<float>();
        const Tensor &input = context->input(2);
        auto input_flat = input.flat<float>();

        const TensorShape &input_shape = input.shape();
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        auto output_flat = output->flat<float>();

        int N = input_flat.size() / times_;
        int M = input_shape.dim_size(0) / times_;
        int C = input_shape.dim_size(1);
        int L = input_shape.dim_size(2);

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, float *grad1, float *grad2, int N, int M, int C, int L, float *output);
        dlerror();
        function moments_grad_kernel = (function)dlsym(lib, "moments_grad");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of moments_grad failed: ", dlsym_error));
        moments_grad_kernel(eid_, (float *)input_flat.data(), (float *)grad1_flat.data(), (float *)grad2_flat.data(), N, M, C, L, (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EMomentsGrad").Device(DEVICE_CPU), EMomentsGradOp);