# Set ARCH for your GPU:
#   sm_75 Turing (RTX 20xx)   sm_86 Ampere (RTX 30xx)   sm_89 Ada (RTX 40xx)
ARCH  ?= sm_86
NVCC  ?= nvcc
FLAGS := -O3 -arch=$(ARCH)

all: vramcheck vrampill

vramcheck: src/vramcheck.cu
	$(NVCC) $(FLAGS) -o $@ $<

vrampill: src/vrampill.cu
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f vramcheck vrampill

.PHONY: all clean
