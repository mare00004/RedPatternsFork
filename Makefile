BIN       := ./build/red-patterns
PHI_INIT  := uv run analysis/phi_init.py export
KERNEL_CLI := uv run analysis/kernel.py export

# phi defaults
PHI_N      := 256
PHI_WING   := 30
PHI_DZ     := 0.000267651
PHI_RHO_C  := 1100.0
PHI_RHO_S  := 30.0
GAUSS_MU   := 1100.0
GAUSS_SIG  := 4.0

# kernel defaults
K_CLOSURE  := force
K_PAIRDIST := nearest-neighbor
K_SIGMA    := 5.6e-6
K_N        := 31
K_DZ       := 2.66614256e-4
K_SUBDIV   := 256
K_G0       := 4.0e7
K_NN_D     := 6.585467201064237e-6
K_NN_SIG   := 0.5e-6

# taylor coefficients
NU  := -2.832638e-30
MU  := -4.468455e-37

# shared generated files
PHI_GAUSS  := ./data/phi_gauss.h5
PHI_CONST  := ./data/phi_const.h5
KERNEL     := ./data/kernel.h5

.PHONY: help \
       run-tayl-gauss-linear run-tayl-gauss-sigmoid \
       run-tayl-const-linear run-tayl-const-sigmoid \
       run-conv-gauss-linear run-conv-gauss-sigmoid \
       run-conv-const-linear run-conv-const-sigmoid

help:
	@echo "RedPatterns — quick-test targets"
	@echo ""
	@echo "This Makefile is for quick testing and serves as a reference for"
	@echo "how to use the phi_init.py, kernel.py, and red-patterns CLIs together."
	@echo ""
	@echo "Each target creates its output directory, generates the required phi"
	@echo "(and kernel for convolution) files, then runs the simulation."
	@echo "Output is written to data/<model>-<phi>-<gradient>/."
	@echo ""
	@echo "Targets:"
	@echo "  run-tayl-gauss-linear    Taylor + Gaussian phi + linear gradient"
	@echo "  run-tayl-gauss-sigmoid   Taylor + Gaussian phi + sigmoid gradient"
	@echo "  run-tayl-const-linear    Taylor + Homogeneous phi + linear gradient"
	@echo "  run-tayl-const-sigmoid   Taylor + Homogeneous phi + sigmoid gradient"
	@echo "  run-conv-gauss-linear    Convolution + Gaussian phi + linear gradient"
	@echo "  run-conv-gauss-sigmoid   Convolution + Gaussian phi + sigmoid gradient"
	@echo "  run-conv-const-linear    Convolution + Homogeneous phi + linear gradient"
	@echo "  run-conv-const-sigmoid   Convolution + Homogeneous phi + sigmoid gradient"
	@echo ""
	@echo "Analyze a completed run:"
	@echo "  uv run marimo run --sandbox analysis/analyze_single_run.py"
	@echo "  (select the run.h5 inside the output directory)"
	@echo ""
	@echo "Explore interactively (build kernel + phi, run, inspect):"
	@echo "  uv run marimo run --sandbox analysis/workbench.py"

$(PHI_GAUSS):
	mkdir -p ./data
	$(PHI_INIT) \
		--output=$(PHI_GAUSS) \
		--phi-type=gaussian \
		--psi-avg=$(PSI) \
		--N=$(PHI_N) \
		--wing=$(PHI_WING) \
		--rho-center=$(PHI_RHO_C) \
		--rho-span=$(PHI_RHO_S) \
		--dz=$(PHI_DZ) \
		--gaussian-mu=$(GAUSS_MU) \
		--gaussian-sigma=$(GAUSS_SIG)

$(PHI_CONST):
	mkdir -p ./data
	$(PHI_INIT) \
		--output=$(PHI_CONST) \
		--phi-type=homogeneous \
		--psi-avg=$(PSI) \
		--N=$(PHI_N) \
		--wing=$(PHI_WING) \
		--rho-center=$(PHI_RHO_C) \
		--rho-span=$(PHI_RHO_S) \
		--dz=$(PHI_DZ)

$(KERNEL):
	mkdir -p ./data
	$(KERNEL_CLI) \
		--output=$(KERNEL) \
		--closure=$(K_CLOSURE) \
		--pair-distribution=$(K_PAIRDIST) \
		--sigma=$(K_SIGMA) \
		--kernel-n=$(K_N) \
		--dz=$(K_DZ) \
		--subdiv=$(K_SUBDIV) \
		--g0=$(K_G0) \
		--nn-d=$(K_NN_D) \
		--nn-sigma=$(K_NN_SIG)

# ---------- TAYLOR ----------

run-tayl-gauss-linear: $(PHI_GAUSS)
	mkdir -p ./data/tayl_gauss_linear
	$(BIN) --use-taylor \
		--T=1000.0 --DT=1e-02 --NO=500 \
		--gradient=linear \
		--NU=$(NU) --MU=$(MU) \
		--phi-file=$(PHI_GAUSS) \
		--out-dir=./data/tayl_gauss_linear \
		--store=phi \
		--store=psi \
		--store=percoll

run-tayl-gauss-sigmoid: $(PHI_GAUSS)
	mkdir -p ./data/tayl_gauss_sigmoid
	$(BIN) --use-taylor \
		--T=1000.0 --DT=1e-02 --NO=500 \
		--gradient=sigmoid \
		--NU=$(NU) --MU=$(MU) \
		--phi-file=$(PHI_GAUSS) \
		--out-dir=./data/tayl_gauss_sigmoid

run-tayl-const-linear: $(PHI_CONST)
	mkdir -p ./data/tayl_const_linear
	$(BIN) --use-taylor \
		--T=2000.0 --DT=1e-02 --NO=500 \
		--gradient=linear \
		--NU=$(NU) --MU=$(MU) \
		--phi-file=$(PHI_CONST) \
		--out-dir=./data/tayl_const_linear

run-tayl-const-sigmoid: $(PHI_CONST)
	mkdir -p ./data/tayl_const_sigmoid
	$(BIN) --use-taylor \
		--T=1000.0 --DT=1e-02 --NO=500 \
		--gradient=sigmoid \
		--NU=$(NU) --MU=$(MU) \
		--phi-file=$(PHI_CONST) \
		--out-dir=./data/tayl_const_sigmoid

# ---------- CONVOLUTION ----------

run-conv-gauss-linear: $(PHI_GAUSS) $(KERNEL)
	mkdir -p ./data/conv_gauss_linear
	$(BIN) --use-convolution \
		--T=300.0 --DT=1e-03 --NO=500 \
		--gradient=linear \
		--kernel-file=$(KERNEL) \
		--phi-file=$(PHI_GAUSS) \
		--out-dir=./data/conv_gauss_linear

run-conv-gauss-sigmoid: $(PHI_GAUSS) $(KERNEL)
	mkdir -p ./data/conv_gauss_sigmoid
	$(BIN) --use-convolution \
		--T=300.0 --DT=1e-03 --NO=500 \
		--gradient=sigmoid \
		--kernel-file=$(KERNEL) \
		--phi-file=$(PHI_GAUSS) \
		--out-dir=./data/conv_gauss_sigmoid

run-conv-const-linear: $(PHI_CONST) $(KERNEL)
	mkdir -p ./data/conv_const_linear
	$(BIN) --use-convolution \
		--T=300.0 --DT=1e-03 --NO=500 \
		--gradient=linear \
		--kernel-file=$(KERNEL) \
		--phi-file=$(PHI_CONST) \
		--out-dir=./data/conv_const_linear

run-conv-const-sigmoid: $(PHI_CONST) $(KERNEL)
	mkdir -p ./data/conv_const_sigmoid
	$(BIN) --use-convolution \
		--T=300.0 --DT=1e-03 --NO=500 \
		--gradient=sigmoid \
		--kernel-file=$(KERNEL) \
		--phi-file=$(PHI_CONST) \
		--out-dir=./data/conv_const_sigmoid
