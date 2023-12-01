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

REGISTER_OP("EMatmul2d")
    .Input("input: float")
    .Input("weight: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));
      ShapeHandle weight_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));
      DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
      DimensionHandle col_dim = c->Dim(weight_shape, 1);
      ShapeHandle output_shape;
      output_shape =c->MakeShape({batch_size_dim,col_dim});
      c->set_output(0, output_shape);
      return Status::OK(); });

REGISTER_OP("EMatmul2dGrad")
    .Input("grad: float")
    .Input("input: float")
    .Input("weight: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output1: float")
    .Output("output2: float");

class EMatmul2dOp : public OpKernel
{
public:
    explicit EMatmul2dOp(OpKernelConstruction *context) : OpKernel(context)
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
        const Tensor &weight = context->input(1);
        auto weight_flat = weight.flat<float>();

        const TensorShape &input_shape = input.shape();
        const TensorShape &weight_shape = weight.shape();

        TensorShape output_shape;
        output_shape.AddDim(input_shape.dim_size(0));
        output_shape.AddDim(weight_shape.dim_size(1));

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();
        int N = input_shape.dim_size(0) / times_;
        int M = input_shape.dim_size(1);
        int C = weight_shape.dim_size(1);

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, float *weight, int N, int M, int C, float *output);
        dlerror();
        function matmul2d_kernel = (function)dlsym(lib, "matmul2d");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of matmul2d failed: ", dlsym_error));
        matmul2d_kernel(eid_, (float *)input_flat.data(), (float *)weight_flat.data(), N, M, C, (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EMatmul2d").Device(DEVICE_CPU), EMatmul2dOp);

class EMatmul2dGradOp : public OpKernel
{
public:
    explicit EMatmul2dGradOp(OpKernelConstruction *context) : OpKernel(context)
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
        const Tensor &weight = context->input(2);

        auto grad_flat = grad.flat<float>();
        auto input_flat = input.flat<float>();
        auto weight_flat = weight.flat<float>();

        const TensorShape &input_shape = input.shape();
        const TensorShape &weight_shape = weight.shape();

        Tensor *output1 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output1));
        auto output1_flat = output1->flat<float>();

        Tensor *output2 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, weight_shape, &output2));
        auto output2_flat = output2->flat<float>();

        int N = input_shape.dim_size(0) / times_;
        int M = input_shape.dim_size(1);
        int C = weight_shape.dim_size(1);

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, float *grad, float *weight, int N, int M, int C, float *output1, float *output2);
        dlerror();
        function matmul2d_grad_kernel = (function)dlsym(lib, "matmul2d_grad");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of matmul2d_grad failed: ", dlsym_error));
        matmul2d_grad_kernel(eid_, (float *)input_flat.data(), (float *)grad_flat.data(), (float *)weight_flat.data(), N, M, C, (float *)output1_flat.data(), (float *)output2_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EMatmul2dGrad").Device(DEVICE_CPU), EMatmul2dGradOp);
