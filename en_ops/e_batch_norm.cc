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

REGISTER_OP("EBatchNorm")
    .Input("input: float")
    .Input("scale: float")
    .Input("offset: float")
    .Input("mean: float")
    .Input("variance: float")
    .Attr("epsilon: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      c->set_output(0, c->input(0));
      return Status::OK(); });

REGISTER_OP("EBatchNormGrad")
    .Input("grad: float")
    .Input("input: float")
    .Input("scale: float")
    .Input("offset: float")
    .Input("mean: float")
    .Input("variance: float")
    .Attr("epsilon: float")
    .Attr("eid_low: int")
    .Attr("eid_high: int")
    .Attr("times:int")
    .Output("grad_input: float")
    .Output("grad_scale: float")
    .Output("grad_offset: float")
    .Output("grad_mean: float")
    .Output("grad_var: float");

using namespace tensorflow;

class EBatchNormOp : public OpKernel
{
public:
    explicit EBatchNormOp(OpKernelConstruction *context) : OpKernel(context)
    {
        context->GetAttr("epsilon", &epsilon_);
        OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
        OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
        OP_REQUIRES_OK(context, context->GetAttr("times", &times_));
        lib = dlopen("/home/zhangyan/jhrsgx/privacy_test/privacy_tf/sgx_tf_ops/sgx.so", RTLD_LAZY);
        OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load sgx.so!"));
    }
    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        const Tensor &scale = context->input(1);
        const Tensor &offset = context->input(2);
        const Tensor &mean = context->input(3);
        const Tensor &variance = context->input(4);

        const TensorShape &input_shape = input.shape();

        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));

        int M = input_shape.dim_size(0) / times_;
        int H = input_shape.dim_size(1);
        int W = input_shape.dim_size(2);

        auto input_flat = input.flat<float>();
        auto scale_flat = scale.flat<float>();
        auto offset_flat = offset.flat<float>();
        auto mean_flat = mean.flat<float>();
        auto variance_flat = variance.flat<float>();
        auto output_flat = output->flat<float>();

        int N = input_flat.size() / times_;

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, float *scale, float *offset, float *mean_x, float *variance, int N, int M, int H, int W, float *output, float epsilon);
        dlerror();
        function batchnorm_kernel = (function)dlsym(lib, "batchnorm");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of batchnorm failed: ", dlsym_error));
        batchnorm_kernel(eid_, (float *)input_flat.data(), (float *)scale_flat.data(), (float *)offset_flat.data(), (float *)mean_flat.data(), (float *)variance_flat.data(), N, M, H, W, (float *)output_flat.data(), epsilon_);
    };

private:
    void *lib;
    float epsilon_;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EBatchNorm").Device(DEVICE_CPU), EBatchNormOp);

class EBatchNormGradOp : public OpKernel
{
public:
    explicit EBatchNormGradOp(OpKernelConstruction *context) : OpKernel(context)
    {
        context->GetAttr("epsilon", &epsilon_);
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
        const Tensor &scale = context->input(2);
        const Tensor &offset = context->input(3);
        const Tensor &mean = context->input(4);
        const Tensor &variance = context->input(5);

        Tensor *grad_input = NULL;
        Tensor *grad_scale = NULL;
        Tensor *grad_offset = NULL;
        Tensor *grad_mean = NULL;
        Tensor *grad_var = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &grad_input));
        OP_REQUIRES_OK(context, context->allocate_output(1, scale.shape(), &grad_scale));
        OP_REQUIRES_OK(context, context->allocate_output(2, offset.shape(), &grad_offset));
        OP_REQUIRES_OK(context, context->allocate_output(3, mean.shape(), &grad_mean));
        OP_REQUIRES_OK(context, context->allocate_output(4, variance.shape(), &grad_var));

        auto grad_flat = grad.flat<float>();
        auto input_flat = input.flat<float>();
        auto scale_flat = scale.flat<float>();
        auto offset_flat = offset.flat<float>();
        auto mean_flat = mean.flat<float>();
        auto variance_flat = variance.flat<float>();

        auto grad_input_flat = grad_input->flat<float>();
        auto grad_scale_flat = grad_scale->flat<float>();
        auto grad_offset_flat = grad_offset->flat<float>();
        auto grad_mean_flat = grad_mean->flat<float>();
        auto grad_var_flat = grad_var->flat<float>();

        const TensorShape &input_shape = input.shape();
        int M = input_shape.dim_size(0) / times_;
        int H = input_shape.dim_size(1);
        int W = input_shape.dim_size(2);
        const int N = input_flat.size() / times_;

        unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
        typedef void (*function)(unsigned long int eid, float *input, float *grad, float *scale, float *offset, float *mean_x, float *variance, int N, int M, int H, int W, float *grad_in, float *grad_sc, float *grad_off, float *grad_mean, float *grad_var, float epsilon);
        dlerror();
        function batchnorm_grad_kernel = (function)dlsym(lib, "batchnorm_grad");
        const char *dlsym_error = dlerror();
        OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of batchnorm_grad failed: ", dlsym_error));
        batchnorm_grad_kernel(eid_, (float *)input_flat.data(), (float *)grad_flat.data(), (float *)scale_flat.data(), (float *)offset_flat.data(), (float *)mean_flat.data(), (float *)variance_flat.data(), N, M, H, W, (float *)grad_input_flat.data(), (float *)grad_scale_flat.data(), (float *)grad_offset_flat.data(), (float *)grad_mean_flat.data(), (float *)grad_var_flat.data(), epsilon_);
    };

private:
    void *lib;
    float epsilon_;
    int64 eid_low_;
    int64 eid_high_;
    int64 times_;
};
REGISTER_KERNEL_BUILDER(Name("EBatchNormGrad").Device(DEVICE_CPU), EBatchNormGradOp);