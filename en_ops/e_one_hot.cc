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

REGISTER_OP("EOneHot")
    .Input("input: float")
    .Attr("alpha: int")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));
      DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
      DimensionHandle col_dim = c->Dim(input_shape, 1);
      DimensionHandle l_dim = c->Dim(input_shape, 2);
      int alpha;
      TF_RETURN_IF_ERROR(c->GetAttr("alpha", &alpha));
      DimensionHandle output_ls = c->MakeDim(alpha);
      ShapeHandle output_shape;
      output_shape =c->MakeShape({batch_size_dim,l_dim,output_ls});
      c->set_output(0, output_shape);
      return Status::OK(); });

class EOneHotOp : public OpKernel
{
public:
    explicit EOneHotOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
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
        output_shape.AddDim(input_shape.dim_size(2));
        output_shape.AddDim(alpha_);

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();
        int N = input_flat.size() / times_;
        int M = input_shape.dim_size(0) / times_;
        int C = input_shape.dim_size(1);
        int L = input_shape.dim_size(2);

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, int N, int M, int C, int L, float *output, int alpha);
        dlerror();
        function onehot_kernel = (function)dlsym(lib, "onehot");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of onehot failed: ", dlsym_error));
        onehot_kernel(eid_, (float *)input_flat.data(), N, M, C, L, (float *)output_flat.data(), alpha_);
    };

private:
    void *lib;
    int64 alpha_;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EOneHot").Device(DEVICE_CPU), EOneHotOp);