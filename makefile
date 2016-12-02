

# Location of the CUDA toolkit
CUDA_DIR = /usr/local/cuda-8.0
LEGACY_CC_PATH = /bin/g++-5
# Compute capability of the target GPU
GPU_ARCH = compute_30
GPU_CODE = sm_30,sm_32,sm_35,sm_37,sm_50,sm_52,sm_53,sm_60,sm_61,sm_62


# Compilers to use
NVCC = $(CUDA_DIR)/bin/nvcc
LEGACY_CC_PATH = /usr/bin/g++-5
# Flags for the host compiler
CCFLAGS = -O3 -std=c++11 -c
WIGNORE = -Wno-return-stack-address

# Flags for nvcc
# ptxas-options=-dlcm=cg (vs. default of ca) is about a 2% performance gain
NVCC_FLAGS = -ccbin $(LEGACY_CC_PATH) -std=c++11 -arch=$(GPU_ARCH) -code=$(GPU_CODE)
TEST = test
TEST_CPU = testcpu
SRC = test128.cu
SRC_CPU = test128cpu.cu
INCLUDE = cuda_uint128.h
INCLUDE_PATHS = -I /home/curtis/CUDASieve/include
CUDASIEVE_LIB = /home/curtis/CUDASieve/libcudasieve.a
LIBS = -lcurand

$(TEST): $(SRC) $(INCLUDE)
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATHS) $(CUDASIEVE_LIB) $(LIBS) $< -o $@
	@echo "     CUDA     " $@

$(TEST_CPU) : $(SRC_CPU) $(INCLUDE)
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATHS) -Xcompiler=-fopenmp $< -o $@
