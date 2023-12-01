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

REGISTER_OP("EEmbTransfer")
    .Input("input:float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      ShapeHandle input_shape = c->input(0);
      int64 M,C;
      if(c->Value(c->Dim(input_shape, 0)) == -1){
      	M = c->Value(c->Dim(input_shape, 0));
      }
      else{
     	M = c->Value(c->Dim(input_shape, 0)) * 3; 
      }
      if(c->Value(c->Dim(input_shape, 1)) == -1){
        C = c->Value(c->Dim(input_shape, 1));
      }
      else{
        C = c->Value(c->Dim(input_shape, 1)) - 1;
      }
      c->set_output(0, c->MakeShape({M,C}));
      return Status::OK(); });

REGISTER_OP("EEmbTransferGrad")
    .Input("input:float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Output("output: float");

class EEmbTransferOp : public OpKernel
{
public:
    explicit EEmbTransferOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        lib = dlopen("/home/zhangyan/jhrsgx/privacy_test/privacy_tf/sgx_tf_ops/sgx.so", RTLD_LAZY);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load sgx.so!"));
    }
    void Compute(OpKernelContext *context) override
    {

        const Tensor &input = context->input(0);
        auto input_flat = input.flat<float>();

        const TensorShape &input_shape = input.shape();
        Tensor *output = NULL;
        TensorShape output_shape;
        output_shape.AddDim(input_shape.dim_size(0) * 10);
        output_shape.AddDim(input_shape.dim_size(1) - 1);
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();
        int M = input_shape.dim_size(0);
        int C = input_shape.dim_size(1);
        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, float *output, int M, int C);
        dlerror();
        function emb_transfer_kernel = (function)dlsym(lib, "emb_transfer");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of emb_transfer failed: ", dlsym_error));
        emb_transfer_kernel(eid_, (float *)input_flat.data(), (float *)output_flat.data(), M, C);
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
};
REGISTER_KERNEL_BUILDER(Name("EEmbTransfer").Device(DEVICE_CPU), EEmbTransferOp);

class EEmbTransferGradOp : public OpKernel
{
public:
    explicit EEmbTransferGradOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        lib = dlopen("/home/zhangyan/jhrsgx/privacy_test/privacy_tf/sgx_tf_ops/sgx.so", RTLD_LAZY);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load sgx.so!"));
    }
    void Compute(OpKernelContext *context) override
    {

        const Tensor &grad = context->input(0);
        auto grad_flat = grad.flat<float>();
        const TensorShape &input_shape = grad.shape();

        TensorShape output_shape;
        output_shape.AddDim(input_shape.dim_size(0) / 10);
        output_shape.AddDim(input_shape.dim_size(1) + 1);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();

        int M = input_shape.dim_size(0) / 10;
        int C = input_shape.dim_size(1);

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *grad, float *output, int M, int C);
        dlerror();
        function emb_transfer_grad_kernel = (function)dlsym(lib, "emb_transfer_grad");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of relu_grad failed: ", dlsym_error));
        emb_transfer_grad_kernel(eid_, (float *)grad_flat.data(), (float *)output_flat.data(), M, C);
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
};
REGISTER_KERNEL_BUILDER(Name("EEmbTransferGrad").Device(DEVICE_CPU), EEmbTransferGradOp);
