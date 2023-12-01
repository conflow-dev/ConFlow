#!/bin/bash
echo "请输入扩大倍数:"
read user_times
echo "请输入随机抽取次数:"
read user_num
echo "请输入密钥:"
read user_key

insert=$((user_times - 1))

cur_path=$(pwd)

sgx_dir="$cur_path/sgx_tf_ops"
cd "$sgx_dir" || exit 1 # 检查切换目录是否成功
sgx_path="$sgx_dir/enclave.signed.so"
app_file_path="$sgx_dir/App/App.cpp"
enclave_path="$sgx_dir/Enclave/Enclave.cpp"

sed -i "s|ret = sgx_create_enclave(\".*\", SGX_DEBUG_FLAG, NULL, NULL, \&eid, NULL);|ret = sgx_create_enclave(\"$sgx_path\", SGX_DEBUG_FLAG, NULL, NULL, \&eid, NULL);|g" "$app_file_path"

sed -i "s|int indexs\[.*\]={[0-9,]*};|int indexs\[${user_times}+5\]=\{${user_key}\};|g" "$enclave_path"
sed -i "s/int Nt = [0-9]\+;/int Nt = ${user_num};/g" "$enclave_path"
sed -i "s/int Ne = [0-9]\+;/int Ne = ${user_times};/g" "$enclave_path"
sed -i "s/int insert = [0-9]\+;/int insert = ${insert};/g" "$enclave_path"

op_path="$cur_path/operators.py"
sed -i "s|times = [0-9]\+|times = ${user_times}|g" "$op_path"
sed -i "s|index = \[.*\]|index = \[${user_key}\]|g" "$op_path"
sed -i "s|times = [0-9]\+|times = ${user_times}|g" "$cur_path/model_demo/DeepCrossing/operators.py"
sed -i "s|index = \[.*\]|index = \[${user_key}\]|g" "$cur_path/model_demo/DeepCrossing/operators.py"

make clean && make

ops_dir="$cur_path/en_ops"
cd "$ops_dir" || exit 1 # 检查切换目录是否成功
tf_ops_path="$sgx_dir/sgx.so"
find . -type f -exec sed -i "s|lib = dlopen(\".*\", RTLD_LAZY);|lib = dlopen(\"$tf_ops_path\", RTLD_LAZY);|g" {} +

make clean && make

cd "$cur_path" || exit 1 # 检查切换目录是否成功

