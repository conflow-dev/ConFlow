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

REGISTER_OP("DeCrypt")
    .Input("input: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
    ShapeHandle input_shape = c->input(0);
	  int input_dims = c->Rank(input_shape);
	  if (input_dims == 2){
	      int64 N, M;
		int times;
      	TF_RETURN_IF_ERROR(c->GetAttr("times", &times));
        if(c->Value(c->Dim(input_shape, 0)) == -1){
        N = c->Value(c->Dim(input_shape, 0));
        }
        else{
        N = c->Value(c->Dim(input_shape, 0))/times;
        }
	M = c->Value(c->Dim(input_shape, 1));
        c->set_output(0, c->MakeShape({N, M}));
	  }
	  else if(input_dims == 3){
		int64 N, M, L;
		int times;
      	TF_RETURN_IF_ERROR(c->GetAttr("times", &times));
        if(c->Value(c->Dim(input_shape, 0)) == -1){
        N = c->Value(c->Dim(input_shape, 0));
        }
        else{
        N = c->Value(c->Dim(input_shape, 0))/times;
        }
	M = c->Value(c->Dim(input_shape, 1));
        L = c->Value(c->Dim(input_shape, 2));
        c->set_output(0, c->MakeShape({N, M, L}));
	  }
	  else{
		int64 N, M, L,C;
		int times;
      	TF_RETURN_IF_ERROR(c->GetAttr("times", &times));
        if(c->Value(c->Dim(input_shape, 0)) == -1){
        N = c->Value(c->Dim(input_shape, 0));
        }
        else{
        N = c->Value(c->Dim(input_shape, 0))/times;
        }
	M = c->Value(c->Dim(input_shape, 1));
        L = c->Value(c->Dim(input_shape, 2));
		    C = c->Value(c->Dim(input_shape, 3));
        c->set_output(0, c->MakeShape({N, M, L,C}));
	  }
      return Status::OK(); });

REGISTER_OP("DeCryptGrad")
    .Input("input: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float");

using namespace tensorflow;

class DeCryptOp : public OpKernel
{
public:
    explicit DeCryptOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
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
        if (input_shape.dims() == 2)
        {
            output_shape.AddDim(input_shape.dim_size(0) / times_);
            output_shape.AddDim(input_shape.dim_size(1));
        }
        else if (input_shape.dims() == 3)
        {
            output_shape.AddDim(input_shape.dim_size(0) / times_);
            output_shape.AddDim(input_shape.dim_size(1));
            output_shape.AddDim(input_shape.dim_size(2));
        }
        else
        {
            output_shape.AddDim(input_shape.dim_size(0) / times_);
            output_shape.AddDim(input_shape.dim_size(1));
            output_shape.AddDim(input_shape.dim_size(2));
            output_shape.AddDim(input_shape.dim_size(3));
        }
        // create output tensor
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();
        int N = input_flat.size() / times_;

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *x, int N, float *y);
        dlerror();
        function decrypt_kernel = (function)dlsym(lib, "decrypt");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of decrypt failed: ", dlsym_error));
        decrypt_kernel(eid_, (float *)input_flat.data(), N, (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("DeCrypt").Device(DEVICE_CPU), DeCryptOp);

class DeCryptGradOp : public OpKernel
{
public:
    explicit DeCryptGradOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
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
        if (input_shape.dims() == 2)
        {
            output_shape.AddDim(times_ * input_shape.dim_size(0));
            output_shape.AddDim(input_shape.dim_size(1));
        }
        else if (input_shape.dims() == 3)
        {
            output_shape.AddDim(times_ * input_shape.dim_size(0));
            output_shape.AddDim(input_shape.dim_size(1));
            output_shape.AddDim(input_shape.dim_size(2));
        }
        else
        {
            output_shape.AddDim(times_ * input_shape.dim_size(0));
            output_shape.AddDim(input_shape.dim_size(1));
            output_shape.AddDim(input_shape.dim_size(2));
            output_shape.AddDim(input_shape.dim_size(3));
        }
        // create output tensor
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();
        int N = input_flat.size();

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *x, int N, float *y);
        dlerror();
        function encrypt_kernel = (function)dlsym(lib, "encrypt1");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of encrypt1 failed: ", dlsym_error));
        encrypt_kernel(eid_, (float *)input_flat.data(), N, (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("DeCryptGrad").Device(DEVICE_CPU), DeCryptGradOp);
