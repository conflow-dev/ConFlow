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

REGISTER_OP("ESoftmaxCrossEntropy")
    .Input("pred: float")
    .Input("real: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int ")
    .Output("loss: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));
      DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
      int times;
      TF_RETURN_IF_ERROR(c->GetAttr("times", &times));
      DimensionHandle output_rows;
      c->Divide(batch_size_dim, times, false, &output_rows);
      ShapeHandle output_shape;
      output_shape =c->MakeShape({output_rows});
      c->set_output(0, output_shape);
      return Status::OK(); });

REGISTER_OP("ESoftmaxCrossEntropyGrad")
    .Input("grad: float")
    .Input("pred: float")
    .Input("real: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float")
    .Output("grad_r: float");

class ESoftmaxCrossEntropyOp : public OpKernel
{
public:
    explicit ESoftmaxCrossEntropyOp(OpKernelConstruction *context) : OpKernel(context)
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

        TensorShape output_shape;
        output_shape.AddDim(input_shape.dim_size(0) / times_);

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();

        const int N = input_flat.size() / times_;
        int M = input_shape.dim_size(0) / times_;
        int C = input_shape.dim_size(1);

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, int N, int M, int C, float *label, float *output);
        dlerror();
        function softmax_cross_entropy_kernel = (function)dlsym(lib, "softmax_cross_entropy");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of softmax_cross_entropy failed: ", dlsym_error));
        softmax_cross_entropy_kernel(eid_, (float *)input_flat.data(), N, M, C, (float *)real_flat.data(), (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("ESoftmaxCrossEntropy").Device(DEVICE_CPU), ESoftmaxCrossEntropyOp);

class ESoftmaxCrossEntropyGradOp : public OpKernel
{
public:
    explicit ESoftmaxCrossEntropyGradOp(OpKernelConstruction *context) : OpKernel(context)
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
        auto input_flat = input.flat<float>();
        auto real_flat = real.flat<float>();
        auto grad_flat = grad.flat<float>();

        const TensorShape &input_shape = input.shape();
        const TensorShape &real_shape = real.shape();
        // create output tensor
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        auto output_flat = output->flat<float>();

        Tensor *grad_r = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, real_shape, &grad_r));
        auto grad_r_flat = grad_r->flat<float>();

        const int N = input_flat.size() / times_;
        int M = input_shape.dim_size(0) / times_;
        int C = input_shape.dim_size(1);

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, float *grad, int N, int M, int C, float *label, float *grad_x, float *grad_l);
        dlerror();
        function softmax_cross_entropy_grad_kernel = (function)dlsym(lib, "softmax_cross_entropy_grad");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of softmax_cross_entropy_grad failed: ", dlsym_error));
        softmax_cross_entropy_grad_kernel(eid_, (float *)input_flat.data(), (float *)grad_flat.data(), N, M, C, (float *)real_flat.data(), (float *)output_flat.data(), (float *)grad_r_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("ESoftmaxCrossEntropyGrad").Device(DEVICE_CPU), ESoftmaxCrossEntropyGradOp);
