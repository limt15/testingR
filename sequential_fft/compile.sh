TF_INC=$(tiapy27 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(tiapt27 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

USE_CXX11_ABI=0
nvcc -std=c++11 -c -o sequential_batch_fft_kernel.cu.o \
  sequential_batch_fft_kernel.cu.cc \
  -D_GLIBCXX_USE_CXX11_ABI=0 -DNDEBUG \
  -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o ./build/sequential_batch_fft.so \
  sequential_batch_fft_kernel.cu.o \
  sequential_batch_fft.cc \
  -D_GLIBCXX_USE_CXX11_ABI=0 -DNDEBUG \
  -I $TF_INC -fPIC \
  -lcudart -lcufft -L/usr/local/cuda/lib64

rm -rf sequential_batch_fft_kernel.cu.o
