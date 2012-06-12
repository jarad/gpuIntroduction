# USER DEFINITIONS

# THIS IS THE LOCATION OF CUDA
CUDA = /usr/local/cuda

# EMULATION MODE OFF (0) OR ON (1)
EMU = 0
#EMU = 1

# PICK YOUR COMPILERS: gnu or intel

# GCC
#CC = gcc
#CPPC = g++

# INTEL
CC = icc
CPPC = icpc

# IF YOU HAVE THE ICC COMPILER, EVEN IF YOU'RE USING GCC, PLEASE PUT THE DIRECTORY HERE SINCE
# USE OF THE INTEL MATH FUNCTIONS MAKES HOST CODE CONSIDERABLY FASTER

ICCDIR = /opt/intel/cce/10.1.015/

DEFINES = 

# COMMENT OUT IF YOU DON'T WANT TO USE CUDPP
USE_CUDPP = 1
DEFINES += -DUSE_CUDPP

# UNCOMMENT WHICHEVER VERSION YOU HAVE
#CUDPP_TYPE = 32
CUDPP_TYPE = 64

# CHOOSE YOUR RNG
DEFINES += -DRNG_XORSHIFT # PARALLEL XORSHIFT
#DEFINES += -DRNG_XORSHIFT_REF # XORSHIFT RUN ON THE HOST
#DEFINES += -DRNG_MRG # PARALLEL MRG

# END OF USER DEFINITIONS

NVCC = nvcc

NVCCFLAGS = -O3
NVCCFLAGS += --compiler-options -Wall
NVCCFLAGS += --compiler-options -Wno-unused-function 
NVCCFLAGS += --compiler-options -ffast-math
NVCCFLAGS += --compiler-options -fno-strict-aliasing
NVCCFLAGS += --compiler-options -funroll-loops
NVCCFLAGS += -use_fast_math
ifeq ($(EMU),1)
NVCCFLAGS += -deviceemu
endif

ifeq ($(CC),gcc)

GCCFLAGS = -O3
GCCFLAGS += -Wall
GCCFLAGS += -Wno-unused-function
GCCFLAGS += -ffast-math
GCCFLAGS += -funroll-loops

ifdef ICCDIR
LIBRARIES = -L$(ICCDIR)/lib -limf
else
LIBRARIES =
endif

CFLAGS = $(GCCFLAGS)
CPPFLAGS = $(GCCFLAGS)
LFLAGS =

endif 

ifeq ($(CC),icc)
ICCFLAGS = -Wall -wd981 -wd869 -wd1572 -wd1418 -O3 -ipo -fast -axT -vec-report0

LIBRARIES = -limf -lguide -lpthread

CFLAGS = $(ICCFLAGS)
CPPFLAGS = $(ICCFLAGS)
LFLAGS = -ipo -fPIC -vec-report0
endif

LIBRARIES += -L$(CUDA)/lib -lcuda -lcudart -Llib -lcutil

ifeq ($(USE_CUDPP),1)    
    ifeq ($(CUDPP_TYPE),32)
        ifeq ($(EMU),0)
            LIBRARIES += -lcudpp
        else
            LIBRARIES += -lcudpp_emu
        endif
    endif
    
    ifeq ($(CUDPP_TYPE),64)
        ifeq ($(EMU),0)
            LIBRARIES += -lcudpp64
        else
            LIBRARIES += -lcudpp64_emu
        endif
    endif
endif

SRC = src

OBJ = obj

BIN = bin

DEPEND = depend

UTIL = $(SRC)/util
RNG = $(UTIL)/rng
DIST = $(SRC)/dist
MC = $(SRC)/mc
MCMC = $(SRC)/mcmc
SMC = $(SRC)/smc
SMCS = $(SRC)/smc_sampler
BASIC = $(SRC)/basic

VPATH = $(SRC) $(UTIL) $(RNG) $(DIST) $(MC) $(MCMC) $(SMC) $(SMCS) $(BASIC)

INCLUDES = -I$(CUDA)/include -Iinclude -I$(UTIL) -I$(RNG) -I$(MC) \
           -I$(DIST) -I$(MCMC) -I$(SMC) -I$(SMCS)
           
UTIL_OBJ = $(OBJ)/reduce.o $(OBJ)/square.o $(OBJ)/matrix.o $(OBJ)/output.o \
           $(OBJ)/scan.o $(OBJ)/scan_ref.o $(OBJ)/order.o $(OBJ)/test_functions.o

RNG_OBJ = $(OBJ)/xorshift.o $(OBJ)/kiss.o $(OBJ)/rng.o $(OBJ)/rng_shared.o $(OBJ)/MRG.o \
          $(OBJ)/xorshift_ref.o $(OBJ)/MRG_ref.o $(OBJ)/rng_shared_ref.o

MC_OBJ = $(OBJ)/mc_mix_gauss.o $(OBJ)/mc_gauss.o $(OBJ)/mc_gauss_mv.o $(OBJ)/test_mc.o \
         $(OBJ)/mc_mix_gauss_mu.o $(OBJ)/mc_mix_gauss_ref.o $(OBJ)/mc_gauss_ref.o \
         $(OBJ)/mc_gauss_mv_ref.o $(OBJ)/mc_mix_gauss_mu_ref.o

MCMC_OBJ = $(OBJ)/mcmc_gauss.o $(OBJ)/mcmc_mix_gauss.o $(OBJ)/mcmc_gauss_mv.o $(OBJ)/test_mcmc.o \
           $(OBJ)/mcmc_mix_gauss_mu.o $(OBJ)/mcmc_gauss_mv_ref.o $(OBJ)/mcmc_mix_gauss_mu_ref.o \
           $(OBJ)/mcmc_gauss_ref.o $(OBJ)/mcmc_mix_gauss_ref.o

SMC_OBJ = $(OBJ)/test_smc.o $(OBJ)/kalman.o $(OBJ)/smc_shared.o \
          $(OBJ)/smc_fsv.o $(OBJ)/smc_lg.o $(OBJ)/smc_mvlg.o \
          $(OBJ)/smc_fsv_ref.o $(OBJ)/smc_lg_ref.o $(OBJ)/smc_mvlg_ref.o \
          $(OBJ)/smc_usv_ref.o $(OBJ)/smc_usv.o

SMCS_OBJ = $(OBJ)/test_smcs.o $(OBJ)/smcs_mix_gauss_mu.o $(OBJ)/smc_shared.o $(OBJ)/smc_shared_ref.o \
           $(OBJ)/smcs_gauss_gauss.o $(OBJ)/smcs_mix_gauss_mu_ref.o $(OBJ)/smcs_gauss_gauss_ref.o

ALL_OBJS = $(UTIL_OBJ) $(RNG_OBJ)

all: $(BIN)/mc $(BIN)/mcmc $(BIN)/smc $(BIN)/smcs $(BIN)/util $(BIN)/is_example

$(BIN)/mc: $(ALL_OBJS) $(MC_OBJ)
	$(CPPC) $(LFLAGS) -o $@ $^ $(LIBRARIES)

$(BIN)/mcmc: $(ALL_OBJS) $(MCMC_OBJ)
	$(CPPC) $(LFLAGS) -o $@ $^ $(LIBRARIES)

$(BIN)/smc: $(ALL_OBJS) $(SMC_OBJ)
	$(CPPC) $(LFLAGS) -o $@ $^ $(LIBRARIES)

$(BIN)/smcs: $(ALL_OBJS) $(SMCS_OBJ)
	$(CPPC) $(LFLAGS) -o $@ $^ $(LIBRARIES)

$(BIN)/util: $(ALL_OBJS) $(OBJ)/test_util.o
	$(CPPC) $(LFLAGS) -o $@ $^ $(LIBRARIES)

$(BIN)/is_example: $(OBJ)/is_example.o $(ALL_OBJS)
	$(CPPC) $(LFLAGS) -o $@ $^ $(LIBRARIES)

ifeq ($(EMU),0)
$(OBJ)/%.o: %.cu
	$(NVCC) $(DEFINES) $(INCLUDES) $(NVCCFLAGS) -odir $(OBJ) -M $< > $(DEPEND)/$(<F).depend
	$(NVCC) $(DEFINES) $(INCLUDES) $(NVCCFLAGS) -o $(OBJ)/$(<F)bin -cubin $<
	$(NVCC) $(DEFINES) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<
else
$(OBJ)/%.o: %.cu
	$(NVCC) $(DEFINES) $(INCLUDES) $(NVCCFLAGS) -odir $(OBJ) -M $< > $(DEPEND)/$(<F).depend
	$(NVCC) $(DEFINES) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<
endif
	
	
$(OBJ)/%.o: %.c
	$(CC) $(DEFINES) $(INCLUDES) $(CFLAGS) -o $@ -c $<

$(OBJ)/%.o: %.cpp
	$(CPPC) $(DEFINES) $(INCLUDES) $(CPPFLAGS) -MT$@ -M $< > $(DEPEND)/$(<F).depend
	$(CPPC) $(DEFINES) $(INCLUDES) $(CPPFLAGS) -o $@ -c $<
	
clean:
	rm -f $(OBJ)/*.o $(OBJ)/*.cubin $(BIN)/mc $(BIN)/mcmc $(BIN)/smc $(BIN)/smcs \
	    $(BIN)/util *.linkinfo $(DEPEND)/*.depend

FORCE:

-include $(DEPEND)/*.depend
