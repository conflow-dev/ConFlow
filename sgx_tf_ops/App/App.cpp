#include <string.h>
#include <assert.h>
#include <fstream>
#include <thread>
#include <iostream>
# include <unistd.h>
# include <pwd.h>
# define MAX_PATH FILENAME_MAX

#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"

/* Global EID shared by multiple threads */
extern "C"{

void ocall_print_string(const char *str)
{
    printf("%s", str);
}

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }
    
    if (idx == ttl)
    	printf("Error code is 0x%X. Please refer to the \"Intel SGX SDK Developer Reference\" for more details.\n", ret);
}


unsigned long int initialize_enclave()
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    unsigned long int eid;
    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave("/home/zhangyan/jhrsgx/privacy_test/privacy_tf/sgx_tf_ops/enclave.signed.so", SGX_DEBUG_FLAG, NULL, NULL, &eid, NULL);
    if (ret != SGX_SUCCESS) {
        printf("Failed to create enclave, ret code: %d\n", ret);
        print_error_message(ret);
        throw ret;
    }
    printf("initialize %lu finish.\n",eid);
    return eid;
}


void destroy_enclave(unsigned long int eid)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    ret = sgx_destroy_enclave(eid);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
    }
    printf("destroy %lu finish.\n",eid);
}

void ceshi(unsigned long int eid,float *input,float *output,int M,int C){
    printf("ceshisgx/n");
    sgx_status_t ret = ecall_ceshi(eid,input,output,M,C); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	}
}

void ceshi_grad(unsigned long int eid,float *input,float *output,int M,int C){
    sgx_status_t ret = ecall_ceshi_grad(eid,input,output,M,C); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	}
}

void encrypt1(unsigned long int eid,float *input, int N, float *output){
    //printf("ceshi_encrypt\n");
    sgx_status_t ret = ecall_encrypt(eid, input,N,output);
    //print_error_message(ret);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
	}
}

void decrypt(unsigned long int eid,float *input, int N, float *output){
    sgx_status_t ret = ecall_decrypt(eid, input,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void relu(unsigned long int eid,float *input, int N, float *output){
    
    
    sgx_status_t ret = ecall_relu(eid, input,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void sigmoid(unsigned long int eid,float *input, int N, float *output){
    
    
    sgx_status_t ret = ecall_sigmoid(eid, input,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void logloss(unsigned long int eid,float *input,int N,int M,int C,float *label,float *weight,float *output){
    
    sgx_status_t ret = ecall_logloss(eid, input,N,M,C,label,weight,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	}  
}

void relu_grad(unsigned long int eid,float *input,float *grad,int N,float *output){
    
    sgx_status_t ret = ecall_relu_grad(eid, input,grad,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void sigmoid_grad(unsigned long int eid,float *input,float *grad,int N,float *output){
    sgx_status_t ret = ecall_sigmoid_grad(eid, input,grad,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
    
}

void logloss_grad(unsigned long int eid,float *input,float grad,int N,int M,int C,float *label,float *weight,float *grad_x,float *grad_w,float *grad_l){
    sgx_status_t ret = ecall_logloss_grad(eid,input,grad,N,M,C,label,weight,grad_x,grad_w,grad_l); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
    
}

void embedding_ss(unsigned long int eid,float* input,float* output,int M,int C,int num,int dim){
    sgx_status_t ret = emb_ss(eid,input,output,M,C,num,dim); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void embedding_ss_grad(unsigned long int eid,float* grad,float* output,int M,int C,int num,int dim){
    sgx_status_t ret = emb_ss_grad(eid,grad,output,M,C,num,dim); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void emb_encrypt(unsigned long int eid,uint32_t dim, float* values,float range){
    sgx_status_t ret = emb_en(eid,dim,values,range); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	}
}

void tanh(unsigned long int eid, float *input,int N,float *output){
    sgx_status_t ret = ecall_tanh(eid,input,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void tanh_grad(unsigned long int eid, float *input, float *grad,int N, float *output){
    sgx_status_t ret = ecall_tanh_grad(eid,input,grad,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void softplus(unsigned long int eid,float *input,int N,float *output){
	sgx_status_t ret = ecall_softplus(eid,input,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void softplus_grad(unsigned long int eid,float *input, float *grad,int N, float *output){
    sgx_status_t ret = ecall_softplus_grad(eid,input,grad,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void leakyrelu(unsigned long int eid, float *input,int N, float *output,float alpha){
    sgx_status_t ret = ecall_leakyrelu(eid,input,N,output,alpha); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void leakyrelu_grad(unsigned long int eid, float *input, float *grad,int N, float *output,float alpha){
    sgx_status_t ret = ecall_leakyrelu_grad(eid,input,grad,N,output,alpha); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void abs1(unsigned long int eid, float *input,int N, float *output){
    sgx_status_t ret = ecall_abs(eid,input,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void abs_grad(unsigned long int eid, float *input, float *grad,int N, float *output){
    sgx_status_t ret = ecall_abs_grad(eid,input,grad,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void log1(unsigned long int eid, float *input,int N, float *output){
    sgx_status_t ret = ecall_log(eid,input,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void log_grad(unsigned long int eid, float *input, float *grad,int N, float *output){
    sgx_status_t ret = ecall_log_grad(eid,input,grad,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void exp1(unsigned long int eid, float *input,int N, float *output){
    sgx_status_t ret = ecall_exp(eid,input,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void exp_grad(unsigned long int eid, float *input, float *grad,int N, float *output){
    sgx_status_t ret = ecall_exp_grad(eid,input,grad,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void sign(unsigned long int eid, float *input,int N, float *output){
    sgx_status_t ret = ecall_sign(eid,input,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void sign_grad(unsigned long int eid, float *input, float *grad,int N, float *output){
    sgx_status_t ret = ecall_sign_grad(eid,input,grad,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void square(unsigned long int eid, float *input,int N, float *output){
    sgx_status_t ret = ecall_square(eid,input,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void square_grad(unsigned long int eid, float *input, float *grad,int N, float *output){
    sgx_status_t ret = ecall_square_grad(eid,input,grad,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void lessequal(unsigned long int eid, float *input,int N, float *output,float alpha){
    sgx_status_t ret = ecall_lessequal(eid,input,N,output,alpha); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void greater(unsigned long int eid, float *input,int N, float *output,float alpha){
    sgx_status_t ret = ecall_greater(eid,input,N,output,alpha); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void reducesum(unsigned long int eid, float *input,int N, float *output){
    sgx_status_t ret = ecall_reducesum(eid,input,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void reducesum_grad(unsigned long int eid, float *input,float grad,int N, float *output){
    sgx_status_t ret = ecall_reducesum_grad(eid,input,grad,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void reducemean(unsigned long int eid, float *input,int N, float *output){
    sgx_status_t ret = ecall_reducemean(eid,input,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void reducemean_grad(unsigned long int eid, float *input,float grad,int N, float *output){
    sgx_status_t ret = ecall_reducemean_grad(eid,input,grad,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void maximum(unsigned long int eid, float *input1, float *input2,int N, float *output){
    sgx_status_t ret = ecall_maximum(eid,input1,input2,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void maximum_grad(unsigned long int eid, float *input1, float *input2, float *grad,int N, float *output1, float *output2){
    sgx_status_t ret = ecall_maximum_grad(eid,input1,input2,grad,N,output1,output2); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void minimum(unsigned long int eid, float *input1, float *input2,int N, float *output){
    sgx_status_t ret = ecall_minimum(eid,input1,input2,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void minimum_grad(unsigned long int eid, float *input1, float *input2, float *grad,int N, float *output1, float *output2){
    sgx_status_t ret = ecall_minimum_grad(eid,input1,input2,grad,N,output1,output2); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void softmax(unsigned long int eid, float *input,int N,int M,int C, float *output){
    sgx_status_t ret = ecall_softmax(eid,input,N,M,C,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void softmax_grad(unsigned long int eid, float *input, float *grad,int N,int M,int C, float *output){
    sgx_status_t ret = ecall_softmax_grad(eid,input,grad,N,M,C,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void clip(unsigned long int eid, float *input,  float* down, float* up,int n1,int n2,int N, float *output){
    sgx_status_t ret = ecall_clip(eid,input,down,up,n1,n2,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void clip_grad(unsigned long int eid, float *input, float *grad,  float* down, float* up,int n1,int n2,int N, float *output){
    sgx_status_t ret = ecall_clip_grad(eid,input,grad,down,up,n1,n2,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void reducemax_one(unsigned long int eid, float *input,int N,int M,int C,int L, float *output){
    sgx_status_t ret = ecall_reducemax_one(eid,input,N,M,C,L,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void reducemax_one_grad(unsigned long int eid, float *input, float *grad,int N,int M,int C,int L, float *output){
    sgx_status_t ret = ecall_reducemax_one_grad(eid,input,grad,N,M,C,L,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void huberloss(unsigned long int eid, float* pred, float* real,float* delta, int N,int C,int B, float* output){
    sgx_status_t ret = ecall_huberloss(eid,pred,real,delta,N,C,B,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void huberloss_grad(unsigned long int eid, float* pred, float* real,float* delta, float* grad, int N,int C,int B, float* output, float* grad_r){
    sgx_status_t ret = ecall_huberloss_grad(eid,pred,real,delta,grad,N,C,B,output,grad_r); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}


void onehot(unsigned long int eid, float* input, int N, int M,int C,int L, float* output,int alpha){
    sgx_status_t ret = ecall_onehot(eid,input,N,M,C,L,output,alpha); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}


void sample(unsigned long int eid, float* input1, float* input2, int N,int nums,  float* output){
    sgx_status_t ret = ecall_sample(eid,input1,input2,N,nums,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void logprob(unsigned long int eid, float* input, float* mu, float* sigma, int N, float* output){
    sgx_status_t ret = ecall_logprob(eid,input,mu,sigma,N,output); 
    if (ret != SGX_SUCCESS) {
		print_error_message(ret);
        throw ret;
	} 
}

void logprob_grad(unsigned long int eid, float* input,float* grad, float* mu, float* sigma, int N, float* output){
    sgx_status_t ret = ecall_logprob_grad(eid,input,grad,mu,sigma,N,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void logloss_none(unsigned long int eid, float *input,int N,float *label,float *weight,float *output){
    sgx_status_t ret = ecall_logloss_none(eid,input,N,label,weight,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void logloss_none_grad(unsigned long int eid, float *input,float *grad,int N,float *label,float *weight,float *grad_x,float *grad_w,float *grad_l){
    sgx_status_t ret = ecall_logloss_none_grad(eid,input,grad,N,label,weight,grad_x,grad_w,grad_l);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void reducemean_0(unsigned long int eid, float *input,int N,int M,int L,float *output){
    sgx_status_t ret = ecall_reducemean_0(eid,input,N,M,L,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void reducemean_0_grad(unsigned long int eid, float *input,float* grad,int N,int M,int L,float *output){
    sgx_status_t ret = ecall_reducemean_0_grad(eid,input,grad,N,M,L,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void moments(unsigned long int eid, float* input, int N,int M, int C, int L, float* mean, float* var){
    sgx_status_t ret = ecall_moments(eid,input,N,M,C,L,mean,var);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void moments_grad(unsigned long int eid, float* input, float* grad1,float* grad2,int N,int M, int C, int L, float* output){
    sgx_status_t ret = ecall_moments_grad(eid,input,grad1,grad2,N,M,C,L,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void batchnorm(unsigned long int eid, float* input,float* scale,float* offset,float* mean_x,float* variance, int N,int M,int H,int W, float* output,float epsilon){
    sgx_status_t ret = ecall_batchnorm(eid,input,scale,offset,mean_x,variance,N,M,H,W,output,epsilon);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void batchnorm_grad(unsigned long int eid, float* input,float* grad,float* scale,float* offset,float* mean_x,float* variance, int N,int M,int H,int W, float* grad_in,float* grad_sc,float* grad_off,float* grad_mean,float* grad_var,float epsilon){
    sgx_status_t ret = ecall_batchnorm_grad(eid,input,grad,scale,offset,mean_x,variance,N,M,H,W,grad_in,grad_sc,grad_off,grad_mean,grad_var,epsilon);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void dot(unsigned long int eid, float* input1,float* input2, int N1, int N2,int* shape1, int* shape2,int ndims ,float* output){
    sgx_status_t ret = ecall_dot(eid,input1,input2,N1,N2,shape1,shape2,ndims,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void dot_grad(unsigned long int eid, float* input1,float* input2, float* grad,int N1,int N2,int* shape1, int* shape2,int ndims ,float* output1,float* output2){
    sgx_status_t ret = ecall_dot_grad(eid,input1,input2,grad,N1,N2,shape1,shape2,ndims,output1,output2);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}


void divi(unsigned long int eid, float* input1,float* input2, int N1, int N2,int* shape1, int* shape2 ,float* output){
    sgx_status_t ret = ecall_divi(eid,input1,input2,N1,N2,shape1,shape2,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void divi_grad(unsigned long int eid, float* input1,float* input2, float* grad,int N1,int N2,int* shape1, int* shape2 ,float* output1,float* output2){
    sgx_status_t ret = ecall_divi_grad(eid,input1,input2,grad,N1,N2,shape1,shape2,output1,output2);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void math(unsigned long int eid, float* input,float num,int N,float* output){
    sgx_status_t ret = ecall_math(eid,input,num,N,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }

}

void matadd(unsigned long int eid, float* input1,float* input2, int N,int C,float* output){
	sgx_status_t ret = ecall_matadd(eid,input1,input2,N,C,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}
void matadd_grad(unsigned long int eid, float* input1,float* input2, float* grad,int N,int C,float* output1,float* output2){
sgx_status_t ret = ecall_matadd_grad(eid,input1,input2,grad,N,C,output1,output2);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void reducemax_zero(unsigned long int eid, float *input,int N,int M,int C,int L, float *output){
    sgx_status_t ret = ecall_reducemax_zero(eid,input,N,M,C,L,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void reducemax_zero_grad(unsigned long int eid, float *input, float *grad,int N,int M,int C,int L, float *output){
    sgx_status_t ret = ecall_reducemax_zero_grad(eid,input,grad,N,M,C,L,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void reducemin_one(unsigned long int eid, float *input,int N,int M,int C,int L, float *output){
    sgx_status_t ret = ecall_reducemin_one(eid,input,N,M,C,L,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void reducemin_one_grad(unsigned long int eid, float *input, float *grad,int N,int M,int C,int L, float *output){
    sgx_status_t ret = ecall_reducemin_one_grad(eid,input,grad,N,M,C,L,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void reducemin_zero(unsigned long int eid, float *input,int N,int M,int C,int L, float *output){
    sgx_status_t ret = ecall_reducemin_zero(eid,input,N,M,C,L,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void reducemin_zero_grad(unsigned long int eid, float *input, float *grad,int N,int M,int C,int L, float *output){
    sgx_status_t ret = ecall_reducemin_zero_grad(eid,input,grad,N,M,C,L,output);
    if (ret != SGX_SUCCESS) {
                print_error_message(ret);
        throw ret;
        }
}

void sort_de(unsigned long int eid,float *input,int N,int M ,int C,int L,float *output){
    sgx_status_t ret = ecall_sort_de(eid,input,N,M,C,L,output);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }
}
void sort_de_grad(unsigned long int eid,float *input,float* grad,int N,int M ,int C,int L,float *output){
	sgx_status_t ret = ecall_sort_de_grad(eid,input,grad,N,M,C,L,output);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }
}

void where_equal(unsigned long int eid,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output){
	sgx_status_t ret = ecall_where_equal(eid,input1,input2,cond1,cond2,N,n1,n2,output);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }
}

void where_equal_grad(unsigned long int eid,float* grad,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output1,float* output2){
	sgx_status_t ret = ecall_where_equal_grad(eid,grad,input1,input2,cond1,cond2,N,n1,n2,output1,output2);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }
}

void where_gequal(unsigned long int eid,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output){
	sgx_status_t ret = ecall_where_gequal(eid,input1,input2,cond1,cond2,N,n1,n2,output);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }
}

void where_gequal_grad(unsigned long int eid,float* grad,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output1,float* output2){
	sgx_status_t ret = ecall_where_gequal_grad(eid,grad,input1,input2,cond1,cond2,N,n1,n2,output1,output2);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }
}

void where_greater(unsigned long int eid,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output){
	sgx_status_t ret = ecall_where_greater(eid,input1,input2,cond1,cond2,N,n1,n2,output);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }
}

void where_greater_grad(unsigned long int eid,float* grad,float *input1,float *input2,float *cond1,float *cond2,int N,int n1,int n2,float *output1,float* output2){
	sgx_status_t ret = ecall_where_greater_grad(eid,grad,input1,input2,cond1,cond2,N,n1,n2,output1,output2);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }
}	

void softmax_cross_entropy(unsigned long int eid,float* pred,int N,int M,int C,float* real,float* output){
	sgx_status_t ret = ecall_softmax_cross_entropy(eid,pred,N,M,C,real,output);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }

}
void softmax_cross_entropy_grad(unsigned long int eid,float* pred,float* grad, int N, int M,int C,float* real,float* output,float* grad_r){
	 sgx_status_t ret = ecall_softmax_cross_entropy_grad(eid,pred,grad,N,M,C,real,output,grad_r);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }
}

void matmul2d(unsigned long int eid,float *input,float* weight,int N,int M,int C,float *output){
	sgx_status_t ret = ecall_matmul(eid,input,weight,N,M,C,output);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }	
}
void matmul2d_grad(unsigned long int eid,float *input,float* grad,float* weight,int N,int M,int C,float *output1,float* output2){
	sgx_status_t ret = ecall_matmul_grad(eid,input,grad,weight,N,M,C,output1,output2);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }
}
/*
void emb_init(unsigned long int eid,float* output,int M,int C){
	sgx_status_t ret = ecall_emb_init(eid,output,M,C);
	 if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }

}

void emb_transfer(unsigned long int eid,float* input,float* output,int M,int C){
	sgx_status_t ret = ecall_emb_transfer(eid,input,output,M,C);
         if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }
}

void emb_transfer_grad(unsigned long int eid,float* grad,float* output,int M,int C){
	sgx_status_t ret = ecall_emb_transfer_grad(eid,grad,output,M,C);
         if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        throw ret;
        }
}
*/
}
