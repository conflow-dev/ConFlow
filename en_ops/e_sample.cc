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

REGISTER_OP("ESample")
    .Input("mu: float")
    .Input("sigma: float")
    .Attr("nums: int")
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
      int nums;
      TF_RETURN_IF_ERROR(c->GetAttr("nums", &nums));
      DimensionHandle output_ls = c->MakeDim(nums);
      ShapeHandle output_shape;
      output_shape =c->MakeShape({output_ls,batch_size_dim,col_dim});
      c->set_output(0, output_shape);
      return Status::OK(); });

class ESampleOp : public OpKernel
{
public:
    explicit ESampleOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("nums", &nums_));
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

        TensorShape output_shape;
        output_shape.AddDim(nums_);
        output_shape.AddDim(input_shape.dim_size(0));
        output_shape.AddDim(input_shape.dim_size(1));

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->flat<float>();
        int N = input1_flat.size() / times_;

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input1, float *input2, int N, int nums, float *output);
        dlerror();
        function sample_kernel = (function)dlsym(lib, "sample");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of sample failed: ", dlsym_error));
        sample_kernel(eid_, (float *)input1_flat.data(), (float *)input2_flat.data(), N, nums_, (float *)output_flat.data());
    };

private:
    void *lib;
    int64 nums_;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("ESample").Device(DEVICE_CPU), ESampleOp);