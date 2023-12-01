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

REGISTER_OP("EmbConvert")
    .Input("input: float")
    .Attr("num: int")
    .Attr("dim: int")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));
      DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
      DimensionHandle col_dim = c->Dim(input_shape, 1);
      DimensionHandle output_rows;
      int times;
      TF_RETURN_IF_ERROR(c->GetAttr("times", &times));
      c->Multiply(batch_size_dim, times, &output_rows);
      int num;
      TF_RETURN_IF_ERROR(c->GetAttr("num", &num));
      int dim;
      TF_RETURN_IF_ERROR(c->GetAttr("dim", &dim));
      DimensionHandle output_cols = c->MakeDim(num * dim);
      ShapeHandle output_shape;
      output_shape =c->MakeShape({output_rows, output_cols});
      c->set_output(0, output_shape);
      return Status::OK(); });

REGISTER_OP("EmbConvertGrad")
    .Input("input: float")
    .Attr("num: int")
    .Attr("dim: int")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float");

class EmbConvertOp : public OpKernel
{
public:
    explicit EmbConvertOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        OP_REQUIRES_OK(context, context->GetAttr("num", &num_));
        OP_REQUIRES_OK(context, context->GetAttr("dim", &dim_));
        OP_REQUIRES_OK(context, context->GetAttr("times", &times_));
        lib = dlopen("/home/zhangyan/jhrsgx/privacy_test/privacy_tf/sgx_tf_ops/sgx.so", RTLD_LAZY);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load sgx.so!"));
    }
    void Compute(OpKernelContext *context) override
    {
        // get the input tensor
        const Tensor &input = context->input(0);
        auto input_flat = input.flat<float>();
        // check shapes of input
        const TensorShape &input_shape = input.shape();
        TensorShape output_shape;
        output_shape.AddDim(10 * input_shape.dim_size(0));
        output_shape.AddDim(num_ * dim_);
        // create output tensor
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();
        int M = input_shape.dim_size(0);
        int C = input_shape.dim_size(1);

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *x, float *y, int M, int C, int num, int dim);
        dlerror();
        function embedding_ss_kernel = (function)dlsym(lib, "embedding_ss");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of embedding_ss failed: ", dlsym_error));

        embedding_ss_kernel(eid_, (float *)input_flat.data(), (float *)output_flat.data(), M, C, num_, dim_);
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 num_;
    int64 dim_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EmbConvert").Device(DEVICE_CPU), EmbConvertOp);

class EmbConvertGradOp : public OpKernel
{
public:
    explicit EmbConvertGradOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        OP_REQUIRES_OK(context, context->GetAttr("num", &num_));
        OP_REQUIRES_OK(context, context->GetAttr("dim", &dim_));
        OP_REQUIRES_OK(context, context->GetAttr("times", &times_));
        lib = dlopen("/home/zhangyan/jhrsgx/privacy_test/privacy_tf/sgx_tf_ops/sgx.so", RTLD_LAZY);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load sgx.so!"));
    }
    void Compute(OpKernelContext *context) override
    {

        // get the input tensor
        const Tensor &input = context->input(0);
        auto input_flat = input.flat<float>();
        // check shapes of input
        const TensorShape &input_shape = input.shape();
        TensorShape output_shape;
        output_shape.AddDim(input_shape.dim_size(0) / times_);
        output_shape.AddDim(num_ * (1 + dim_));
        // create output tensor
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();
        int M = input_shape.dim_size(0) / times_;
        int C = num_ * (1 + dim_);
        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *x, float *y, int M, int C, int num, int dim);

        dlerror();
        function embedding_ss_grad_kernel = (function)dlsym(lib, "embedding_ss_grad");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of embedding_ss_grad failed: ", dlsym_error));

        embedding_ss_grad_kernel(eid_, (float *)input_flat.data(), (float *)output_flat.data(), M, C, num_, dim_);
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 num_;
    int64 dim_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EmbConvertGrad").Device(DEVICE_CPU), EmbConvertGradOp);