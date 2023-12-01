import tensorflow.compat.v1 as tf
import os
from ctypes import *
from tensorflow.python.framework import ops
current_path = os.path.dirname(os.path.abspath(__file__))
parts = current_path.split(os.sep)
new_path = os.sep.join(parts[:-2])
trainer_op_path = os.path.realpath(os.path.join(str(new_path) + "/en_ops/", "ops.so"))
trainer_ops = tf.load_op_library(trainer_op_path)

__all__ = [
    "e_sigmoid",
    "e_relu",
    "en_crypt",
    "de_crypt",
    "e_logloss",
    "c_relu",
    "emb_convert",
    "ceshi",
    "e_abs",
    "e_batch_norm",
    "e_clip",
    "e_dot",
    "e_exp",
    "e_greater",
    "e_huber_loss",
    "e_leakyrelu",
    "e_less_equal",
    "e_log",
    "e_logprob",
    "e_maximum",
    "e_minimum",
    "e_moments",
    "e_one_hot",
    "e_reduce_max_one",
    "e_reduce_mean",
    "e_reduce_sum",
    "e_sample",
    "e_sign",
    "e_softmax",
    "e_softplus",
    "e_square",
    "e_tanh",
    "e_logloss_none",
    "e_reduce_mean_zero",
    "e_matmul",
    "e_div",
    "e_math",
    "create_enclave",
    "e_matadd",
    "add_ss",
    "e_reduce_max_zero",
    "e_reduce_min_one",
    "e_reduce_min_zero",
    "e_sort_de",
    "e_where_equal",
    "e_where_gequal",
    "e_where_greater",
    "e_softmax_cross_entropy",
    "e_matmul2d",
    "c_matmul2d",
    "destroy",
]

eid = 2
times = 5

@tf.custom_gradient
def e_matmul(x, y):
    z = tf.matmul(x, y)
    def grad(dz):
        index = [1,2,0,3,1,4]
        dx = tf.matmul(dz, tf.transpose(y))
        real_batch = 32
        dy = tf.zeros_like(y)
        for i in index:
            for j in index:
                dy += tf.matmul(tf.transpose(x[i * real_batch : (i + 1) * real_batch]),dz[j * real_batch : (j + 1) * real_batch])
        return dx, dy
    return z, grad


def create_enclave():
    sgx_lib = cdll.LoadLibrary(os.path.join(new_path + "/sgx_tf_ops/", "sgx.so"))
    sgx_lib.initialize_enclave.restype = c_ulong
    global eid
    eid = sgx_lib.initialize_enclave()
    return eid,times

def destroy():
    global eid
    sgx_lib = cdll.LoadLibrary(os.path.join(new_path + "/sgx_tf_ops/", "sgx.so"))
    sgx_lib.destroy_enclave.argtypes = [c_ulong]
    sgx_lib.destroy_enclave(eid)
    return 0

def e_emb_init(input_dim,output_dim):
    global eid
    return trainer_ops.emb_init(input_dim,output_dim,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32))

def e_matmul2d(inputs,weight):
    global eid,times
    return trainer_ops.e_matmul2d(inputs,weight,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EMatmul2d")
def _e_matmul2d_grad(op, grad):
    with tf.name_scope("EMatmul2dGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_matmul2d_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_matmul2d_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1]

def e_softmax_cross_entropy(inputs,real):
    global eid,times
    return trainer_ops.e_softmax_cross_entropy(inputs,real,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ESoftmaxCrossEntropy")
def _e_softmax_cross_entropy_grad(op,grad):
    with tf.name_scope("ESoftmaxCrossEntropyGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_softmax_cross_entropy_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_softmax_cross_entropy_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1]

def e_where_equal(input1,input2,cond1,cond2):
    global eid,times
    return trainer_ops.e_where_equal(input1,input2,cond1,cond2,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EWhereEqual")
def _e_where_equal_grad(op,grad):
    with tf.name_scope("EWhereEqualGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_where_equal_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_where_equal_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1],None,None

def e_where_gequal(input1,input2,cond1,cond2):
    global eid,times
    return trainer_ops.e_where_gequal(input1,input2,cond1,cond2,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EWhereGequal")
def _e_where_gequal_grad(op,grad):
    with tf.name_scope("EWhereGequalGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_where_gequal_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_where_gequal_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1],None,None

def e_where_greater(input1,input2,cond1,cond2):
    global eid,times
    return trainer_ops.e_where_greater(input1,input2,cond1,cond2,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EWhereGreater")
def _e_where_greater_grad(op,grad):
    with tf.name_scope("EWhereGreaterGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_where_greater_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_where_greater_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1],None,None


def e_reduce_max_zero(inputs):
    global eid,times
    return trainer_ops.e_reduce_max_zero(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EReduceMaxZero")
def _e_reduce_max_zero_grad(op, grad):
    with tf.name_scope("EReduceMaxZeroGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_reduce_max_zero_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_reduce_min_one(inputs):
    global eid,times
    return trainer_ops.e_reduce_min_one(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EReduceMinOne")
def _e_reduce_min_one_grad(op, grad):
    with tf.name_scope("EReduceMinOneGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_reduce_min_one_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_reduce_min_zero(inputs):
    global eid,times
    return trainer_ops.e_reduce_min_zero(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EReduceMinZero")
def _e_reduce_min_zero_grad(op, grad):
    with tf.name_scope("EReduceMinZeroGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_reduce_min_zero_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))


def e_sort_de(inputs):
    global eid,times
    return trainer_ops.e_sort_de(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ESortDe")
def _e_sort_de_grad(op, grad):
    with tf.name_scope("ESortDeGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_sort_de_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))


def e_math(inputs,num):
    global eid,times
    return trainer_ops.e_math(inputs,num,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EMath")
def _e_math_grad(op, grad):
    with tf.name_scope("EMathGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return grad

def e_matadd(input1,input2):
    global eid,times
    return trainer_ops.e_matadd(input1,input2,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EMatadd")
def _e_matadd_grad(op, grad):
    with tf.name_scope("EMataddGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_matadd_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_matadd_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1]


def e_div(input1,input2):
    global eid,times
    return trainer_ops.e_div(input1,input2,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EDiv")
def _e_div_grad(op, grad):
    with tf.name_scope("EDivGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_div_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_div_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1]

def e_reduce_mean_zero(inputs):
    global eid,times
    return trainer_ops.e_reduce_mean_zero(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EReduceMeanZero")
def _e_reduce_mean_zero_grad(op, grad):
    with tf.name_scope("EReduceMeanZeroGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_reduce_mean_zero_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_logloss_none(pred,real,weights):
    global eid,times
    return trainer_ops.e_logloss_none(pred,real,weights,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ELoglossNone")
def _e_logloss_none_grad(op, grad):
    with tf.name_scope("ELoglossNoneGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_logloss_none_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_logloss_none_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[2],trainer_ops.e_logloss_none_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1]

def e_square(inputs):
    global eid,times
    return trainer_ops.e_square(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ESquare")
def _e_square_grad(op, grad):
    with tf.name_scope("ESquareGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_square_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_reduce_max_one(inputs):
    global eid,times
    return trainer_ops.e_reduce_max_one(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EReduceMaxOne")
def _e_reduce_max_one_grad(op, grad):
    with tf.name_scope("EReduceMaxOneGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_reduce_max_one_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_reduce_mean(inputs):
    global eid,times
    return trainer_ops.e_reduce_mean(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EReduceMean")
def _e_reduce_mean_grad(op, grad):
    with tf.name_scope("EReduceMeanGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_reduce_mean_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_reduce_sum(inputs):
    global eid,times
    return trainer_ops.e_reduce_sum(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EReduceSum")
def _e_reduce_sum_grad(op, grad):
    with tf.name_scope("EReduceSumGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_reduce_sum_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_sample(mu,sigma,nums=1):
    global eid,times
    return trainer_ops.e_sample(mu,sigma,nums,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ESample")

def e_sign(inputs):
    global eid,times
    return trainer_ops.e_sign(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ESign")
def _e_sign_grad(op, grad):
    with tf.name_scope("ESignGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_sign_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_softmax(inputs):
    global eid,times
    return trainer_ops.e_softmax(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ESoftmax")
def _e_softmax_grad(op, grad):
    with tf.name_scope("ESoftmaxGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_softmax_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_softplus(inputs):
    global eid,times
    return trainer_ops.e_softplus(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ESoftplus")
def _e_softplus_grad(op, grad):
    with tf.name_scope("ESoftplusGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_softplus_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_tanh(inputs):
    global eid,times
    return trainer_ops.e_tanh(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ETanh")
def _e_tanh_grad(op, grad):
    with tf.name_scope("ETanhGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_tanh_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_one_hot(inputs,alpha):
    global eid,times
    return trainer_ops.e_one_hot(inputs,alpha = alpha,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EOneHot")

def e_moments(inputs):
    global eid,times
    return trainer_ops.e_moments(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)[0],trainer_ops.e_moments(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)[1]
@ops.RegisterGradient("EMoments")
def _e_moments_grad(op, grad1,grad2):
    with tf.name_scope("EMomentsGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_moments_grad(grad1,grad2,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_minimum(input1,input2):
    global eid,times
    return trainer_ops.e_minimum(input1,input2,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EMinimum")
def _e_maximum_grad(op, grad):
    with tf.name_scope("EMinimumGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_minimum_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_minimum_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1]

def e_maximum(input1,input2):
    global eid,times
    return trainer_ops.e_maximum(input1,input2,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EMaximum")
def _e_maximum_grad(op, grad):
    with tf.name_scope("EMaximumGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_maximum_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_maximum_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1]

def e_logprob(inputs,mu,sigma):
    global eid,times
    return trainer_ops.e_logprob(inputs,mu,sigma,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ELogprob")
def _e_logprob_grad(op, grad):
    with tf.name_scope("ELogprobGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_logprob_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times")),None,None

def e_log(inputs):
    global eid,times
    return trainer_ops.e_log(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ELog")
def _e_log_grad(op, grad):
    with tf.name_scope("ELogGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_log_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_less_equal(inputs,alpha):
    global eid,times
    return trainer_ops.e_less_equal(inputs,alpha = alpha,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ELessEqual")

def e_leakyrelu(inputs,alpha):
    global eid,times
    return trainer_ops.e_leakyrelu(inputs,alpha=alpha,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ELeakyrelu")
def _e_leakyrelu_grad(op, grad):
    with tf.name_scope("ELeakyreluGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_leakyrelu_grad(grad,op.inputs[0],alpha = op.get_attr("alpha"),eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_huber_loss(pred,real,delta):
    global eid,times
    return trainer_ops.e_huber_loss(pred,real,delta,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EHuberLoss")
def _e_huber_loss_grad(op, grad):
    with tf.name_scope("EHuberLossGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_huber_loss_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_huber_loss_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1],None

def e_greater(inputs,alpha):
    global eid,times
    return trainer_ops.e_greater(inputs,alpha = alpha,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EGreater")

def e_exp(inputs):
    global eid,times
    return trainer_ops.e_exp(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EExp")
def _e_exp_grad(op, grad):
    with tf.name_scope("EExpGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_exp_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_dot(input1,input2):
    global eid,times
    return trainer_ops.e_dot(input1,input2,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EDot")
def _e_dot_grad(op, grad):
    with tf.name_scope("EDotGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_dot_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_dot_grad(grad,op.inputs[0],op.inputs[1],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1]


def e_clip(inputs,down,up):
    global eid,times
    return trainer_ops.e_clip(inputs,down,up,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EClip")
def _e_clip_grad(op, grad):
    with tf.name_scope("EClipGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_clip_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times")),None,None


def e_batch_norm(inputs,scale,offset,mean,variance,epsilon):
    global eid,times
    return trainer_ops.e_batch_norm(inputs,scale,offset,mean,variance,epsilon=epsilon,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EBatchNorm")
def _e_batch_norm_grad(op, grad):
    with tf.name_scope("EBatchNormGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_batch_norm_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],op.inputs[4],op.get_attr('epsilon'),eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_batch_norm_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],op.inputs[4],op.get_attr('epsilon'),eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1],trainer_ops.e_batch_norm_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],op.inputs[4],op.get_attr('epsilon'),eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[2],trainer_ops.e_batch_norm_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],op.inputs[4],op.get_attr('epsilon'),eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[3],trainer_ops.e_batch_norm_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],op.inputs[4],op.get_attr('epsilon'),eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[4]

def e_abs(inputs):
    global eid,times
    return trainer_ops.e_abs(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EAbs")
def _e_abs_grad(op, grad):
    with tf.name_scope("EAbsGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_abs_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))


def ceshi(inputs):
    global eid,times
    return trainer_ops.ceshi(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32))
@ops.RegisterGradient("Ceshi")
def _ceshi_grad(op, grad):
    with tf.name_scope("CeshiGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.ceshi_grad(grad,eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"))

def emb_convert(inputs,num,dim):
    global eid,times
    return trainer_ops.emb_convert(inputs,num=num,dim=dim,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EmbConvert")
def _emb_convert_grad(op, grad):
    with tf.name_scope("EmbConvertGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.emb_convert_grad(grad,num=op.get_attr("num"),dim=op.get_attr("dim"),eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))


def e_relu(inputs):
    global eid,times
    return trainer_ops.e_relu(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ERelu")
def _e_relu_grad(op, grad):
    with tf.name_scope("EReluGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_relu_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def en_crypt(inputs):
    global eid,times
    return trainer_ops.en_crypt(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("EnCrypt")
def _en_crypt_grad(op, grad):
    with tf.name_scope("EnCryptGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.en_crypt_grad(grad,eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def de_crypt(inputs):
    global eid,times
    return trainer_ops.de_crypt(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("DeCrypt")
def _de_crypt_grad(op, grad):
    with tf.name_scope("DeCryptGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.de_crypt_grad(grad,eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_sigmoid(inputs):
    global eid,times
    return trainer_ops.e_sigmoid(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ESigmoid")
def _e_sigmoid_grad(op, grad):
    with tf.name_scope("ESigmoidGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_sigmoid_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))

def e_logloss(pred,real,weights):
    global eid,times
    return trainer_ops.e_log_loss(pred,real,weights,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
@ops.RegisterGradient("ELogLoss")
def _e_logloss_grad(op, grad):
    with tf.name_scope("ELogLossGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
        return trainer_ops.e_log_loss_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[0],trainer_ops.e_log_loss_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[2],trainer_ops.e_log_loss_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))[1]
