#ifndef _APP_H_
#define _APP_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "sgx_error.h"       /* sgx_status_t */
#include "sgx_eid.h"     /* sgx_enclave_id_t */

#ifndef TRUE
# define TRUE 1
#endif

#ifndef FALSE
# define FALSE 0
#endif

# define TOKEN_FILENAME   "enclave.token"
# define ENCLAVE_FILENAME "enclave.signed.so"

#if defined(__cplusplus)
extern "C" {
#endif
unsigned long int initialize_enclave();
void destroy_enclave(unsigned long int eid);
void encrypt1(unsigned long int eid,float *input, int N, float *output);
void decrypt(unsigned long int eid,float *input, int N, float *output);
void relu(unsigned long int eid,float *input, int N, float *output);
void sigmoid(unsigned long int eid,float *input,int N,float *output);
void logloss(unsigned long int eid,float *input,int N,int M,int C,float *label,float *weight,float *output);
void relu_grad(unsigned long int eid,float *input,float *grad,int N,float *output);
void sigmoid_grad(unsigned long int eid,float *input,float *grad,int N,float *output);
void logloss_grad(unsigned long int eid,float *input,float grad,int N,int M,int C,float *label,float *weight,float *grad_x,float *grad_w,float *grad_l);
void embedding_ss(unsigned long int eid,float* input,float* output,int M,int C,int num,int dim);
void embedding_ss_grad(unsigned long int eid,float* grad,float* output,int M,int C,int num,int dim);
void emb_encrypt(unsigned long int eid,uint32_t dim, float* values,float range);
void ceshi(unsigned long int eid,float* input,float* output,int M,int C);
void ceshi_grad(unsigned long int eid,float* input,float* output,int M,int C);
void tanh(unsigned long int eid, float *input,int N,float *output);
void tanh_grad(unsigned long int eid, float *input, float *grad,int N, float *output);
void softplus(unsigned long int eid,float *input,int N,float *output);
void softplus_grad(unsigned long int eid,float *input, float *grad,int N, float *output);
void leakyrelu(unsigned long int eid, float *input,int N, float *output,float alpha);
void leakyrelu_grad(unsigned long int eid, float *input, float *grad,int N, float *output,float alpha);
void abs1(unsigned long int eid, float *input,int N, float *output);
void abs_grad(unsigned long int eid, float *input, float *grad,int N, float *output);
void log1(unsigned long int eid, float *input,int N, float *output);
void log_grad(unsigned long int eid, float *input, float *grad,int N, float *output);
void exp1(unsigned long int eid, float *input,int N, float *output);
void exp_grad(unsigned long int eid, float *input, float *grad,int N, float *output);
void sign(unsigned long int eid, float *input,int N, float *output);
void sign_grad(unsigned long int eid, float *input, float *grad,int N, float *output);
void square(unsigned long int eid, float *input,int N, float *output);
void square_grad(unsigned long int eid, float *input, float *grad,int N, float *output);
void lessequal(unsigned long int eid, float *input,int N, float *output,float alpha);
void greater(unsigned long int eid, float *input,int N, float *output,float alpha);
void reducesum(unsigned long int eid, float *input,int N, float *output);
void reducesum_grad(unsigned long int eid, float *input,float grad,int N, float *output);
void reducemean(unsigned long int eid, float *input,int N, float *output);
void reducemean_grad(unsigned long int eid, float *input,float grad,int N, float *output);
void maximum(unsigned long int eid, float *input1, float *input2,int N, float *output);
void maximum_grad(unsigned long int eid, float *input1, float *input2, float *grad,int N, float *output1, float *output2);
void minimum(unsigned long int eid, float *input1, float *input2,int N, float *output);
void minimum_grad(unsigned long int eid, float *input1, float *input2, float *grad,int N, float *output1, float *output2);
void softmax(unsigned long int eid, float *input,int N,int M,int C, float *output);
void softmax_grad(unsigned long int eid, float *input, float *grad,int N,int M,int C, float *output);
void clip(unsigned long int eid, float *input,  float* down, float* up,int n1,int n2,int N, float *output);
void clip_grad(unsigned long int eid, float *input, float *grad,  float* down, float* up,int n1,int n2,int N, float *output);
void reducemax_one(unsigned long int eid, float *input,int N,int M,int C,int L, float *output);
void reducemax_one_grad(unsigned long int eid, float *input, float *grad,int N,int M,int C,int L, float *output);
void huberloss(unsigned long int eid, float* pred, float* real,float* delta, int N,int C,int B, float* output);
void huberloss_grad(unsigned long int eid, float* pred, float* real,float* delta, float* grad, int N,int C,int B, float* output, float* grad_r);
void onehot(unsigned long int eid, float* input, int N, int M,int C,int L, float* output,int alpha);
void sample(unsigned long int eid, float* input1, float* input2, int N,int nums,  float* output);
void logprob(unsigned long int eid, float* input, float* mu, float* sigma, int N, float* output);
void logprob_grad(unsigned long int eid, float* input, float* grad ,float* mu, float* sigma, int N, float* output);
void logloss_none(unsigned long int eid, float *input,int N,float *label,float *weight,float *output);
void logloss_none_grad(unsigned long int eid, float *input,float *grad,int N,float *label,float *weight,float *grad_x,float *grad_w,float *grad_l);
void reducemean_0(unsigned long int eid, float *input,int N,int M,int L,float *output);
void reducemean_0_grad(unsigned long int eid, float *input,float* grad,int N,int M,int L,float *output);
void moments(unsigned long int eid, float* input, int N,int M, int C, int L, float* mean, float* var);
void moments_grad(unsigned long int eid, float* input, float* grad1,float* grad2,int N,int M, int C, int L, float* output);
void batchnorm(unsigned long int eid, float* input,float* scale,float* offset,float* mean_x,float* variance, int N,int M,int H,int W, float* output,float epsilon);
void batchnorm_grad(unsigned long int eid, float* input,float* grad,float* scale,float* offset,float* mean_x,float* variance, int N,int M,int H,int W, float* grad_in,float* grad_sc,float* grad_off,float* grad_mean,float* grad_var,float epsilon);
void dot(unsigned long int eid, float* input1,float* input2, int N1,int N2,int* shape1, int* shape2 ,int ndims,float* output);
void dot_grad(unsigned long int eid, float* input1,float* input2, float* grad,int N1,int N2,int* shape1, int* shape2,int ndims ,float* output1,float* output2);
void divi(unsigned long int eid, float* input1,float* input2, int N1,int N2,int* shape1, int* shape2 ,float* output);
void divi_grad(unsigned long int eid, float* input1,float* input2, float* grad,int N1,int N2,int* shape1, int* shape2 ,float* output1,float* output2);
void math(unsigned long int eid,float* input,float num,int N,float* output);
void matadd(unsigned long int eid, float* input1,float* input2, int N,int C,float* output);
void matadd_grad(unsigned long int eid, float* input1,float* input2, float* grad,int N,int C,float* output1,float* output2);
void reducemin_one(unsigned long int eid, float *input,int N,int M,int C,int L, float *output);
void reducemin_one_grad(unsigned long int eid, float *input, float *grad,int N,int M,int C,int L, float *output);
void reducemin_zero(unsigned long int eid, float *input,int N,int M,int C,int L, float *output);
void reducemin_zero_grad(unsigned long int eid, float *input, float *grad,int N,int M,int C,int L, float *output);
void reducemax_zero(unsigned long int eid, float *input,int N,int M,int C,int L, float *output);
void reducemax_zero_grad(unsigned long int eid, float *input, float *grad,int N,int M,int C,int L, float *output);
void sort_de(unsigned long int eid,float *input,int N,int M ,int C,int L,float *output);
void sort_de_grad(unsigned long int eid,float *input,float* grad,int N,int M ,int C,int L,float *output);
void where_equal(unsigned long int eid,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output);
void where_equal_grad(unsigned long int eid,float* grad,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output1,float* output2);
void where_gequal(unsigned long int eid,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output);
void where_gequal_grad(unsigned long int eid,float* grad,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output1,float* output2);
void where_greater(unsigned long int eid,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output);
void where_greater_grad(unsigned long int eid,float* grad,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output1,float* output2);
void softmax_cross_entropy(unsigned long int eid,float* pred,int N,int M,int C,float* real,float* output);
void softmax_cross_entropy_grad(unsigned long int eid,float* pred,float* grad, int N, int M,int C,float* real,float* output,float* grad_r);
void matmul2d(unsigned long int eid,float *input,float* weight,int N,int M,int C,float *output);
void matmul2d_grad(unsigned long int eid,float *input,float* grad,float* weight,int N,int M,int C,float *output1,float* output2);
//void emb_init(unsigned long int eid,float *output,int M,int C);
//void emb_transfer(unsigned long int eid,float* input,float* output,int M,int C);
//void emb_transfer_grad(unsigned long int eid,float* grad,float* output,int M,int C);

#if defined(__cplusplus)
}
#endif

#endif /* !_APP_H_ */

