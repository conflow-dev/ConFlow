TF_CFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

objects = $(wildcard *.cc)

ops.so: $(objects)
	g++ -std=c++11 -shared $(objects) -o ops.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -DUSE_SGX 

clean:
	rm -rf ops.so

