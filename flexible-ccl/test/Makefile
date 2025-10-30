#
# Test Makefile for NCCL Two-Phase Activation
#

# Auto-detect CUDA installation
ifeq ($(CUDA_HOME),)
    ifneq ($(wildcard /usr/local/cuda/bin/nvcc),)
        CUDA_HOME = /usr/local/cuda
    else ifneq ($(wildcard /usr/local/cuda-11.8/bin/nvcc),)
        CUDA_HOME = /usr/local/cuda-11.8
    else ifneq ($(wildcard /usr/local/cuda-11.7/bin/nvcc),)
        CUDA_HOME = /usr/local/cuda-11.7
    else
        CUDA_HOME = /usr/local/cuda
    endif
endif

NCCL_HOME ?= ../build

CXX = g++
NVCC = $(CUDA_HOME)/bin/nvcc

# Compiler flags
CXXFLAGS = -std=c++11 -O2 -g
NVCCFLAGS = -std=c++11 -O2 -g

# Include paths
INCLUDES = -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include

# Library paths and libraries
LDFLAGS = -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib
LIBS = -lcudart -lnccl

# Source files
SOURCES = two_phase_activation_test.cc addition.cc removal.cc concurrent_addition_test.cc serial_addition_test.cc
TARGETS = two_phase_activation_test addition removal concurrent_addition_test serial_addition_test

.PHONY: all clean test

all: $(TARGETS)

# Build two-phase activation test
two_phase_activation_test: two_phase_activation_test.cc
	@echo "Building two-phase activation test..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)

# Build existing addition test
addition: addition.cc
	@echo "Building addition test..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)

# Build existing removal test
removal: removal.cc
	@echo "Building removal test..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)

# Build concurrent addition test
concurrent_addition_test: concurrent_addition_test.cc
	@echo "Building concurrent addition test..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS) -lpthread

# Build serial addition test
serial_addition_test: serial_addition_test.cc
	@echo "Building serial addition test..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)

# Run tests
test: two_phase_activation_test
	@echo "Running two-phase activation test..."
	./two_phase_activation_test

# Clean build artifacts
clean:
	rm -f $(TARGETS)

# Help target
help:
	@echo "Available targets:"
	@echo "  all                    - Build all tests"
	@echo "  two_phase_activation_test - Build two-phase activation test"
	@echo "  addition              - Build addition test"
	@echo "  removal               - Build removal test"
	@echo "  concurrent_addition_test - Build concurrent addition test"
	@echo "  serial_addition_test  - Build serial addition test"
	@echo "  test                  - Run two-phase activation test"
	@echo "  clean                 - Clean build artifacts"
	@echo "  help                  - Show this help message"