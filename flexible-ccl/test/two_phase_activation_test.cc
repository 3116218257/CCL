#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

#define CUDACHECK(cmd)                                                    \
    do {                                                                  \
        cudaError_t err = cmd;                                            \
        if (err != cudaSuccess) {                                         \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define NCCLCHECK(cmd)                                                    \
    do {                                                                  \
        ncclResult_t res = cmd;                                           \
        if (res != ncclSuccess) {                                         \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
                   ncclGetErrorString(res));                              \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

void print_test_header(const char* test_name) {
    printf("\n=== %s ===\n", test_name);
}

void print_test_result(const char* test_name, bool passed) {
    printf("[%s] %s: %s\n", passed ? "PASS" : "FAIL", test_name, 
           passed ? "SUCCESS" : "FAILED");
}

// Test 1: Basic two-phase activation functionality
bool test_basic_two_phase_activation() {
    print_test_header("Basic Two-Phase Activation Test");
    
    // Check available GPU count first
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    printf("Available GPUs: %d\n", deviceCount);
    
    const int nranks = (deviceCount >= 4) ? 4 : 2;
    const int size = 1024;
    ncclComm_t comms[nranks];
    float **sendbuff = (float **)malloc(nranks * sizeof(float *));
    float **recvbuff = (float **)malloc(nranks * sizeof(float *));
    cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nranks);
    
    // Initialize devices and buffers
    for (int i = 0; i < nranks; i++) {
        if (i >= deviceCount) {
            printf("Error: Trying to use device %d but only %d devices available\n", i, deviceCount);
            return false;
        }
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc((void **)&sendbuff[i], size * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)&recvbuff[i], size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], i + 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Phase 1: Initialize communicators (connectors should be inactive)
    printf("Phase 1: Initializing communicators with inactive connectors...\n");
    ncclUniqueId uniqueId;
    NCCLCHECK(ncclGetUniqueId(&uniqueId));
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nranks; i++) {
        if (i >= deviceCount) {
            printf("Error: Trying to use device %d but only %d devices available\n", i, deviceCount);
            return false;
        }
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommInitRank(&comms[i], nranks, uniqueId, i));
    }
    NCCLCHECK(ncclGroupEnd());
    printf("Phase 1 completed: All communicators initialized\n");
    
    // Phase 2: Selective activation of connectors
    int active_ranks[2];
    int num_active_ranks;
    if (nranks >= 4) {
        printf("Phase 2: Activating connectors for ranks 0 and 2...\n");
        active_ranks[0] = 0;
        active_ranks[1] = 2;
        num_active_ranks = 2;
    } else {
        printf("Phase 2: Activating connectors for rank 0...\n");
        active_ranks[0] = 0;
        num_active_ranks = 1;
    }
    
    for (int i = 0; i < nranks; i++) {
        NCCLCHECK(ncclActivateConnectors(comms[i], active_ranks, num_active_ranks));
    }
    printf("Phase 2 completed: Connectors activated for specified ranks\n");
    
    // Test communication with all ranks (AllReduce requires all ranks to participate)
    printf("Testing communication with all ranks...\n");
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nranks; i++) {
        if (i >= deviceCount) {
            printf("Error: Trying to use device %d but only %d devices available\n", i, deviceCount);
            return false;
        }
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], 
                               size, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    
    // Synchronize and verify results for all ranks
    for (int i = 0; i < nranks; i++) {
        if (i >= deviceCount) {
            printf("Error: Trying to use device %d but only %d devices available\n", i, deviceCount);
            return false;
        }
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // Cleanup
    for (int i = 0; i < nranks; i++) {
        if (i >= deviceCount) {
            printf("Error: Trying to use device %d but only %d devices available\n", i, deviceCount);
            return false;
        }
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
    }
    free(sendbuff);
    free(recvbuff);
    free(streams);
    
    printf("Basic two-phase activation test completed successfully\n");
    return true;
}

// Test 2: Progressive activation test
bool test_progressive_activation() {
    print_test_header("Progressive Activation Test");
    
    // Check available GPU count and adjust nranks accordingly
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    printf("Available GPUs: %d\n", deviceCount);
    
    int nranks = (deviceCount >= 4) ? 4 : deviceCount;
    if (nranks < 2) {
        printf("Progressive activation test requires at least 2 GPUs, but only %d available\n", deviceCount);
        return false;
    }
    
    const int size = 512;
    ncclComm_t *comms = (ncclComm_t *)malloc(nranks * sizeof(ncclComm_t));
    float **sendbuff = (float **)malloc(nranks * sizeof(float *));
    float **recvbuff = (float **)malloc(nranks * sizeof(float *));
    cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nranks);
    
    // Initialize devices and buffers
    for (int i = 0; i < nranks; i++) {
        if (i >= deviceCount) {
            printf("Error: Trying to use device %d but only %d devices available\n", i, deviceCount);
            return false;
        }
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc((void **)&sendbuff[i], size * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)&recvbuff[i], size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], i + 1, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Initialize communicators
    printf("Initializing communicators...\n");
    ncclUniqueId uniqueId;
    NCCLCHECK(ncclGetUniqueId(&uniqueId));
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nranks; i++) {
        if (i >= deviceCount) {
            printf("Error: Trying to use device %d but only %d devices available\n", i, deviceCount);
            return false;
        }
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommInitRank(&comms[i], nranks, uniqueId, i));
    }
    NCCLCHECK(ncclGroupEnd());
    
    // Progressive activation: start with 2 ranks, then add more
    printf("Step 1: Activating ranks 0 and 1...\n");
    int step1_ranks[] = {0, 1};
    for (int i = 0; i < nranks; i++) {
        NCCLCHECK(ncclActivateConnectors(comms[i], step1_ranks, 2));
    }
    
    if (nranks >= 3) {
        printf("Step 2: Adding rank 2...\n");
        int step2_ranks[] = {0, 1, 2};
        for (int i = 0; i < nranks; i++) {
            NCCLCHECK(ncclActivateConnectors(comms[i], step2_ranks, 3));
        }
    }
    
    if (nranks >= 4) {
        printf("Step 3: Adding rank 3 (all ranks active)...\n");
        int step3_ranks[] = {0, 1, 2, 3};
        for (int i = 0; i < nranks; i++) {
            NCCLCHECK(ncclActivateConnectors(comms[i], step3_ranks, 4));
        }
    } else {
        printf("Final step: All %d ranks active...\n", nranks);
        int *all_ranks = (int *)malloc(nranks * sizeof(int));
        for (int j = 0; j < nranks; j++) {
            all_ranks[j] = j;
        }
        for (int i = 0; i < nranks; i++) {
            NCCLCHECK(ncclActivateConnectors(comms[i], all_ranks, nranks));
        }
        free(all_ranks);
    }
    
    // Test final communication with all ranks
    printf("Testing communication with all ranks...\n");
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nranks; i++) {
        if (i >= deviceCount) {
            printf("Error: Trying to use device %d but only %d devices available\n", i, deviceCount);
            return false;
        }
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], 
                               size, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    
    // Synchronize
    for (int i = 0; i < nranks; i++) {
        if (i >= deviceCount) {
            printf("Error: Trying to use device %d but only %d devices available\n", i, deviceCount);
            return false;
        }
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // Cleanup
    for (int i = 0; i < nranks; i++) {
        if (i >= deviceCount) {
            printf("Error: Trying to use device %d but only %d devices available\n", i, deviceCount);
            return false;
        }
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }
    free(comms);
    free(sendbuff);
    free(recvbuff);
    free(streams);
    
    printf("Progressive activation test completed successfully\n");
    return true;
}

// Test 3: Error handling test
bool test_error_handling() {
    print_test_header("Error Handling Test");
    
    // Check available GPU count
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    printf("Available GPUs: %d\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("Error handling test requires at least 2 GPUs, but only %d available\n", deviceCount);
        return false;
    }
    
    const int nranks = 2;
    ncclComm_t comms[nranks];
    
    // Initialize communicators
    ncclUniqueId uniqueId;
    NCCLCHECK(ncclGetUniqueId(&uniqueId));
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nranks; i++) {
        if (i >= deviceCount) {
            printf("Error: Trying to use device %d but only %d devices available\n", i, deviceCount);
            return false;
        }
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommInitRank(&comms[i], nranks, uniqueId, i));
    }
    NCCLCHECK(ncclGroupEnd());
    
    // Test invalid rank
    printf("Testing invalid rank handling...\n");
    int invalid_ranks[] = {0, 5}; // rank 5 doesn't exist
    ncclResult_t result = ncclActivateConnectors(comms[0], invalid_ranks, 2);
    if (result == ncclInvalidArgument) {
        printf("✓ Invalid rank correctly rejected\n");
    } else {
        printf("✗ Invalid rank should have been rejected\n");
        return false;
    }
    
    // Test negative rank count
    printf("Testing negative rank count...\n");
    int valid_ranks[] = {0, 1};
    result = ncclActivateConnectors(comms[0], valid_ranks, -1);
    if (result == ncclInvalidArgument) {
        printf("✓ Negative rank count correctly rejected\n");
    } else {
        printf("✗ Negative rank count should have been rejected\n");
        return false;
    }
    
    // Test NULL ranks array
    printf("Testing NULL ranks array...\n");
    result = ncclActivateConnectors(comms[0], NULL, 1);
    if (result == ncclInvalidArgument) {
        printf("✓ NULL ranks array correctly rejected\n");
    } else {
        printf("✗ NULL ranks array should have been rejected\n");
        return false;
    }
    
    // Cleanup
    for (int i = 0; i < nranks; i++) {
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }
    
    printf("Error handling test completed successfully\n");
    return true;
}

int main(int argc, char *argv[]) {
    printf("=== Two-Phase Activation Test Suite ===\n");
    printf("Testing NCCL two-phase activation functionality\n\n");
    
    bool all_tests_passed = true;
    
    // Run all tests
    all_tests_passed &= test_basic_two_phase_activation();
    all_tests_passed &= test_progressive_activation();
    all_tests_passed &= test_error_handling();
    
    // Print final results
    printf("\n=== Test Suite Results ===\n");
    if (all_tests_passed) {
        printf("🎉 ALL TESTS PASSED! Two-phase activation is working correctly.\n");
        return 0;
    } else {
        printf("❌ SOME TESTS FAILED! Please check the implementation.\n");
        return 1;
    }
}