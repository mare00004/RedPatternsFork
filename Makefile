run-tayl:
	./build/red-patterns --use-taylor --T=300.0 --DT=5e-04 --NO=30000 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=3.0e-10 --delta=1.0e-11 --kappa=1.0e-12 --NU=-6.0e-30 --MU=-1.0e-36 --out-dir=./data/tayl_sigmoid

run-conv:
	./build/red-patterns --use-convolution --T=200.0 --DT=5e-04 --NO=3000 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --out-dir=./data/conv_sigmoid

KERNEL_FILE=./data/kernel.h5

run-kernel-notebook-conv:
	./build/red-patterns --use-convolution --T=1000.0 --DT=5e-04 --NO=10000 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --out-dir=./data/conv_sigmoid_kernel_notebook --kernel-file=/home/max/src/tries/2026-05-18-RedPatterns-fixed/intKernelNotebook.h5 --phi-file=/home/max/projects/RedPatternsFork/data/phi.h5

run-kernel-notebook-tayl-linear:
	./build/red-patterns --use-taylor --T=1000.0 --DT=5e-03 --NO=200 --gradient=linear --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --NU=-2.832638e-30 --MU=-4.468455e-37 --out-dir=./data/tayl_sigmoid_kernel_notebook_linear  --phi-file=/home/max/projects/RedPatternsFork/config/phi.h5

run-kernel-notebook-tayl:
	./build/red-patterns --use-taylor --T=1000.0 --DT=1e-02 --NO=500 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --NU=-2.832638e-30 --MU=-4.468455e-37 --out-dir=./data/tayl_sigmoid_kernel_notebook  --phi-file=/home/max/projects/RedPatternsFork/config/phi.h5

	
##########
# TAYLOR #
##########

# Gauss
run-tayl-gauss-linear:
	./build/red-patterns --use-taylor --T=1000.0 --DT=1e-02 --NO=500 --gradient=linear --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --NU=-2.832638e-30 --MU=-4.468455e-37 --phi-file=/home/max/projects/RedPatternsFork/data/phi_gauss.h5 --out-dir=./data/tayl_gauss_linear

run-tayl-gauss-sigmoid:
	./build/red-patterns --use-taylor --T=1000.0 --DT=1e-02 --NO=500 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --NU=-2.832638e-30 --MU=-4.468455e-37 --phi-file=/home/max/projects/RedPatternsFork/data/phi_gauss.h5 --out-dir=./data/tayl_gauss_sigmoid

# Constant
run-tayl-const-linear:
	./build/red-patterns --use-taylor --T=1000.0 --DT=1e-02 --NO=500 --gradient=linear --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --NU=-2.832638e-30 --MU=-4.468455e-37 --phi-file=/home/max/projects/RedPatternsFork/data/phi_const.h5 --out-dir=./data/tayl_const_linear

run-tayl-const-sigmoid:
	./build/red-patterns --use-taylor --T=1000.0 --DT=1e-02 --NO=500 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --NU=-2.832638e-30 --MU=-4.468455e-37 --phi-file=/home/max/projects/RedPatternsFork/data/phi_const.h5 --out-dir=./data/tayl_const_sigmoid

###########
# SIGMOID #
###########

# Gauss
run-conv-gauss-linear:
	./build/red-patterns --use-convolution --T=1000.0 --DT=1e-02 --NO=500 --gradient=linear --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --kernel-file=./data/kernel.h5 --phi-file=/home/max/projects/RedPatternsFork/data/phi_gauss.h5 --out-dir=./data/conv_gauss_linear

run-conv-gauss-sigmoid:
	./build/red-patterns --use-convolution --T=1000.0 --DT=1e-02 --NO=500 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --kernel-file=./data/kernel.h5 --phi-file=/home/max/projects/RedPatternsFork/data/phi_gauss.h5 --out-dir=./data/conv_gauss_sigmoid

# Constant
run-conv-const-linear:
	./build/red-patterns --use-convolution --T=1000.0 --DT=1e-02 --NO=500 --gradient=linear --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --kernel-file=./data/kernel.h5 --phi-file=/home/max/projects/RedPatternsFork/data/phi_const.h5 --out-dir=./data/conv_const_linear

run-conv-const-sigmoid:
	./build/red-patterns --use-convolution --T=1000.0 --DT=1e-02 --NO=500 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --kernel-file=./data/kernel.h5 --phi-file=/home/max/projects/RedPatternsFork/data/phi_const.h5 --out-dir=./data/conv_const_sigmoid


test:
	./build/red-patterns --use-taylor --T=1000.0 --DT=1e-03 --NO=5000 --gradient=linear --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=-2.8e-30 --NU=-2.8e-30 --MU=-4.4e-37 --phi-file=/home/max/projects/RedPatternsFork/data/phi_const.h5 --out-dir=./data/test

# With hihger EQ_DIST (kernel and nu, mu calculated on fine grid with U = 100e-18)
# With kernel_d and kernel_old and the constant phi and linear gradient the simulations diverge (for the convolution not the taylor)
test-kernel-d:
	./build/red-patterns --use-convolution --T=300.0 --DT=5e-04 --NO=5000 --gradient=sigmoid --U=1.0e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=-2.8e-30 --kernel-file=/home/max/projects/RedPatternsFork/data/kernel_d.h5 --phi-file=/home/max/projects/RedPatternsFork/data/phi_gauss.h5 --out-dir=./data/test_kernel_d

test-kernel-old:
	./build/red-patterns --use-convolution --T=300.0 --DT=5e-04 --NO=5000 --gradient=sigmoid --U=1.0e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=-2.8e-30 --kernel-file=/home/max/projects/RedPatternsFork/data/kernel1.h5 --phi-file=/home/max/projects/RedPatternsFork/data/phi_gauss.h5 --out-dir=./data/test_kernel_old

test-taylor-d:
	./build/red-patterns --use-taylor --T=800.0 --DT=1e-03 --NO=5000 --gradient=sigmoid --U=1.0e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=-2.8e-30 --NU=-2.275e-30 --MU=-4.397e-37 --phi-file=/home/max/projects/RedPatternsFork/data/phi_gauss.h5 --out-dir=./data/test_taylor_d

.PHONY: run-tayl run-conv test run-kernel-notebook-conv run-kernel-notebook-tayl run-kernel-notebook-tayl-linear
