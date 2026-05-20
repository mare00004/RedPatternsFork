run-tayl:
	./build/red-patterns --use-taylor --T=300.0 --DT=5e-04 --NO=30000 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=3.0e-10 --delta=1.0e-11 --kappa=1.0e-12 --NU=-6.0e-30 --MU=-1.0e-36 --out-dir=./data/tayl_sigmoid

run-conv:
	./build/red-patterns --use-convolution --T=200.0 --DT=5e-04 --NO=3000 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --out-dir=./data/conv_sigmoid

test:
	./build/red-patterns --use-taylor --T=1.0 --DT=5e-04 --NO=1 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --NU=-1.6049962938777745e-29 --MU=-7.052525226362305e-36 --out-dir=./data/tayl_sigmoid

run-kernel-notebook-conv:
	./build/red-patterns --use-convolution --T=1000.0 --DT=5e-04 --NO=10000 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --out-dir=./data/conv_sigmoid_kernel_notebook --kernel-file=/home/max/src/tries/2026-05-18-RedPatterns-fixed/intKernelNotebook.h5

run-kernel-notebook-tayl:
	./build/red-patterns --use-taylor --T=1000.0 --DT=5e-04 --NO=10000 --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 --NU=-2.832638e-30 --MU=-4.468455e-37 --out-dir=./data/tayl_sigmoid_kernel_notebook 
	

.PHONY: run-tayl run-conv test run-kernel-notebook-conv run-kernel-notebook-tayl
