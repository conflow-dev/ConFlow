

## **环境**
需要预先安装好tensorflow(version>2.0)与Intel SGX相关软硬件

## **代码结构**

- **en_ops** :

- - Makefile
  - *.cc : tensorflow自定义算子的C++文件。

- **sgx_tf_ops** :

- - App ：加密算子与外界的接口
  - Enclave：加密算子的真正实现文件
  -  Makefile

- **model_demo** :

- - DeepCrossing：DeepCrossing模型，将输入进行加密，并将非线性算子替换为加密算子。

- **operators.py** : 加密算子的python包装层，模型可直接使用import operators进行使用。

- **run.sh** : 配置及编译脚本



## **使用指南**

### **配置及编译**



```bash
sh run.sh #运行脚本
请输入扩大倍数: #输入一个数字，该数字为数据加密后扩大的倍数
请输入抽取次数: #输入一个数字，该数字为数据加密时随机抽取的次数，例如5，为从随机生成的数据中抽取5次。
请输入密钥：#输入一串数字，用‘,’分隔开，数字数目应与抽取次数相同
#等待编译完成即可。en_ops文件夹下会出现ops.so,sgx_tf_ops文件夹下会出现sgx.so文件。
```



### **简单使用及测试**



```python
import tensorflow.compat.v1 as tf #导入tensorflow包
import operators as sgx #导入加密算子包

sgx.create_enclave() #进行可信执行环境的初始化

a = tf.constant([[2,3,4],[-1,-2,0]],tf.float32) #创建一个tensor
b = sgx.en_crypt(a) #对tensor进行加密处理
c = sgx.e_relu(b) #对加密的tensor进行加密的relu运算
d = sgx.de_crypt(c) #对加密计算的结果进行解密
```



### **运行模型**



```bash
cd ./model_demo/Deepcrossing #进入模型目录
python train.py #运行加密模型
```







## **开发自定义算子**

### **自定义SGX算子**

1. 在./sgx_tf_ops/Enclave/Enclave.cpp中实现具体计算逻辑：
 

   ```c++
     void ecall_test(float *input,int N,float *output){
   
     }
     void ecall_test_grad(float *input,float *grad,int N,float *output){
   
     }
   ```

   

3. 在./sgx_tf_ops/Enclave/Enclave.edl中的trusted 中注册：
 

   ```
     public void ecall_test([user_check]float* input,int N,[user_check]float* output);
     public void ecall_test_grad([user_check]float* input,[user_check]float* grad,int N,[user_check]float* output);
   ```

   

5. 在./sgx_tf_ops/App/App.h中写接口的头文件：


   ```c++
     void test(unsigned long int eid,float *input, int N, float *output);
     void test_grad(unsigned long int eid,float *input,float* grad, int N, float *output);
   ```

   

7. 在./sgx_tf_ops/App/App.cpp中写接口函数：
 

   ```c++
     void test(unsigned long int eid,float *input, int N, float *output){
         sgx_status_t ret = ecall_test(eid, input,N,output);
         if (ret != SGX_SUCCESS) {
             print_error_message(ret);
             throw ret;
             }
     }
     void test_grad(unsigned long int eid,float *input,float *grad, int N, float *output){
         sgx_status_t ret = ecall_test_grad(eid, input,grad,N,output);
         if (ret != SGX_SUCCESS) {
             print_error_message(ret);
             throw ret;
             }
     }
   ```

   

9. 最后重新make编译即可。

### **自定义TF算子:**

1. 在./en_ops文件夹下新建test.cc文件

2. 导入头文件以及设置命名空间：


   ```c++
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
   ```

   

4. 注册ops


   ```c++
     REGISTER_OP("ETest")
         .Input("input: float")
         .Attr("eid_low: int")
         .Attr("eid_high: int")
         .Attr("times:int")
         .Output("output: float")
         .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
           c->set_output(0, c->input(0)); //形状设定
           return Status::OK();
         });
   
     REGISTER_OP("ETestGrad")
         .Input("grad: float")
         .Input("input: float")
         .Attr("eid_low: int")
         .Attr("eid_high: int")
         .Attr("times:int")
         .Output("output: float");
   ```

   

6. 注册kernels


   ```c++
     class ETestOp : public OpKernel {
     public:
         explicit ETestOp(OpKernelConstruction* context) : OpKernel(context) {
             OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
             OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
             OP_REQUIRES_OK(context, context->GetAttr("times", &times_));
             lib = dlopen("(所在路径)/sgx.so",RTLD_LAZY);
             OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load sgx.so!"));
         }
         void Compute(OpKernelContext* context) override {
         const Tensor& input = context->input(0); //获取输入
         auto input_flat = input.flat<float>(); 
         const TensorShape& input_shape = input.shape(); //获取输入形状
         Tensor* output = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output)); //初始化输出
         auto output_flat = output->flat<float>();
         int N = input_flat.size()/times_;
   
         unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
         typedef void (*function)(unsigned long int eid,float* input, int N, float* output);
         dlerror();
         function test_kernel = (function) dlsym(lib, "test"); //调用SGX算子
         const char *dlsym_error = dlerror();
         OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of test failed: ", dlsym_error)); //失败处理
         test_kernel(eid_,(float*)input_flat.data(),N,(float*)output_flat.data());//调用SGX进行计算
     };
     private:
         void* lib;
         int64 eid_low_;
         int64 eid_high_;
         int64 times_;
     };
     REGISTER_KERNEL_BUILDER(Name("ETest").Device(DEVICE_CPU), ETestOp);
   
     class ETestGradOp : public OpKernel {
     public:
         explicit ETestGradOp(OpKernelConstruction* context) : OpKernel(context) {
             OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
             OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
             OP_REQUIRES_OK(context, context->GetAttr("times", &times_));
             lib = dlopen("(所在路径)/sgx.so",RTLD_LAZY);
             OP_REQUIRES(context, lib != NULL, errors::Unknown("Unable to load sgx.so!"));
         }
         void Compute(OpKernelContext* context) override {
         const Tensor& grad = context->input(0);
         const Tensor& input = context->input(1);
   
         auto grad_flat = grad.flat<float>();
         auto input_flat = input.flat<float>();
         // check shapes of input 
         const TensorShape& input_shape = input.shape();
         // create output tensor
         Tensor* output = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
         auto output_flat = output->flat<float>();
   
         const int N = input_flat.size()/times_;
   
         unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
         typedef void (*function)(unsigned long int eid,float* input,float *grad, int N, float* output);
         dlerror();
         function test_grad_kernel = (function) dlsym(lib, "test_grad");
         const char *dlsym_error = dlerror();
         OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of relu_grad failed: ", dlsym_error));
         test_grad_kernel(eid_,(float*)input_flat.data(),(float*)grad_flat.data(),N,(float*)output_flat.data());
     };
     private:
         void* lib;
         int64 eid_low_;
         int64 eid_high_;
         int64 times_;
     };
     REGISTER_KERNEL_BUILDER(Name("ETestGrad").Device(DEVICE_CPU), ETestGradOp);
   ```

   

8. 使用make重新编译

9. python层包装及梯度注册

10. 在operators.py中进行函数包装和梯度注册：



```python
   def e_test(inputs):
       global eid,times
       return trainer_ops.e_test(inputs,eid_low=(eid&0xFFFFFFFF),eid_high=(eid>>32),times=times)
   @ops.RegisterGradient("ETest")
   def _e_test_grad(op, grad):
       with tf.name_scope("ETestGrad"), tf.xla.experimental.jit_scope(compile_ops=False):
           return trainer_ops.e_test_grad(grad,op.inputs[0],eid_low=op.get_attr("eid_low"), eid_high=op.get_attr("eid_high"),times = op.get_attr("times"))
```



### **使用**



```python
   import tensorflow as tf
   import operators
   operators.create_enclave() #初始化enclave
   operators.e_test() #使用自定义算子
```
