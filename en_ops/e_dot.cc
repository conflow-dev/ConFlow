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

REGISTER_OP("EDot")
    .Input("input1: float")
    .Input("input2: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      ShapeHandle input1_shape = c->input(0);
      ShapeHandle input2_shape = c->input(1);
      int input1_dims = c->Rank(input1_shape);
      if (input1_dims == 2){
        int64 N, M;
        N = c->Value(c->Dim(input1_shape, 0));
        M = std::max(c->Value(c->Dim(input1_shape, 1)), c->Value(c->Dim(input2_shape, 1)));
        c->set_output(0, c->MakeShape({N, M}));
      }
      else{
        int64 N, M, L;
        N = c->Value(c->Dim(input1_shape, 0));
        M = std::max(c->Value(c->Dim(input1_shape, 1)), c->Value(c->Dim(input2_shape, 1)));
        L = std::max(c->Value(c->Dim(input1_shape, 2)), c->Value(c->Dim(input2_shape, 2)));
        c->set_output(0, c->MakeShape({N, M, L}));
      }
      return Status::OK(); });

REGISTER_OP("EDotGrad")
    .Input("grad: float")
    .Input("input1: float")
    .Input("input2: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output1: float")
    .Output("output2: float");

class EDotOp : public OpKernel
{
public:
    explicit EDotOp(OpKernelConstruction *context) : OpKernel(context)
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

        const TensorShape &input1_shape = input1.shape();
        const TensorShape &input2_shape = input2.shape();

        int shape1[input1_shape.dims()];
        for (int i = 0; i < input1_shape.dims(); i++)
        {
            shape1[i] = input1_shape.dim_size(i);
        }

        int shape2[input2_shape.dims()];
        for (int i = 0; i < input2_shape.dims(); i++)
        {
            shape2[i] = input2_shape.dim_size(i);
        }

        TensorShape output_shape;
        for (int i = 0; i < input1_shape.dims(); i++)
        {
            int dim_tmp = shape1[i] >= shape2[i] ? shape1[i] : shape2[i];
            output_shape.AddDim(dim_tmp);
        }

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();
        int N1 = input1_flat.size() / times_;
        int N2 = input2_flat.size() / times_;
        int ndims = input1_shape.dims();

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input1, float *input2, int N1, int N2, int *shape1, int *shape2, int ndims, float *output);
        dlerror();
        function dot_kernel = (function)dlsym(lib, "dot");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of dot failed: ", dlsym_error));
        dot_kernel(eid_, (float *)input1_flat.data(), (float *)input2_flat.data(), N1, N2, (int *)shape1, (int *)shape2, ndims, (float *)output_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EDot").Device(DEVICE_CPU), EDotOp);

class EDotGradOp : public OpKernel
{
public:
    explicit EDotGradOp(OpKernelConstruction *context) : OpKernel(context)
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

        const TensorShape &input1_shape = input1.shape();
        const TensorShape &input2_shape = input2.shape();

        int shape1[input1_shape.dims()];
        for (int i = 0; i < input1_shape.dims(); i++)
        {
            shape1[i] = input1_shape.dim_size(i);
        }

        int shape2[input2_shape.dims()];
        for (int i = 0; i < input2_shape.dims(); i++)
        {
            shape2[i] = input2_shape.dim_size(i);
        }

        Tensor *output1 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input1_shape, &output1));
        auto output1_flat = output1->flat<float>();

        Tensor *output2 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, input2_shape, &output2));
        auto output2_flat = output2->flat<float>();

        int N1 = input1_flat.size() / times_;
        int N2 = input2_flat.size() / times_;
        int ndims = input1_shape.dims();

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input1, float *input2, float *grad, int N1, int N2, int *shape1, int *shape2, int ndims, float *output1, float *output2);
        dlerror();
        function dot_grad_kernel = (function)dlsym(lib, "dot_grad");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of dot_grad failed: ", dlsym_error));
        dot_grad_kernel(eid_, (float *)input1_flat.data(), (float *)input2_flat.data(), (float *)grad_flat.data(), N1, N2, (int *)shape1, (int *)shape2, ndims, (float *)output1_flat.data(), (float *)output2_flat.data());
    };

private:
    void *lib;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EDotGrad").Device(DEVICE_CPU), EDotGradOp);