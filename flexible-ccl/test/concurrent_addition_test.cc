#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
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

// Global variables for test
ncclComm_t comms[10];
float **sendbuff;
float **recvbuff;
cudaStream_t *streams;
ncclUniqueId global_uniqueId;  // Store the original uniqueId for reuse
int nFirst = 2;
int nTotal = 3;
int rank_new = 2;
int size = 1024;
volatile int communication_running = 0;
volatile int addition_completed = 0;
volatile int test_failed = 0;

// Timing variables
struct timespec test_start_time, test_end_time;
struct timespec comm_start_time, comm_end_time;
long total_communication_time = 0;
int communication_iterations = 0;

// Thread function for continuous communication
void* communication_thread(void* arg) {
    printf("[COMM THREAD] Starting continuous communication...\n");
    
    int iteration = 0;
    communication_running = 1;
    clock_gettime(CLOCK_MONOTONIC, &comm_start_time);
    
    while (!addition_completed && !test_failed) {
        iteration++;
        printf("[COMM THREAD] Communication iteration %d\n", iteration);
        
        struct timespec iter_start, iter_end;
        clock_gettime(CLOCK_MONOTONIC, &iter_start);
        
        // Perform AllReduce on existing ranks with proper error handling
        ncclResult_t ncclRes = ncclGroupStart();
        if (ncclRes != ncclSuccess) {
            printf("[COMM THREAD] ERROR: ncclGroupStart failed: %s\n", ncclGetErrorString(ncclRes));
            test_failed = 1;
            break;
        }
        
        for (int i = 0; i < nFirst; ++i) {
            // Ensure proper device context
            CUDACHECK(cudaSetDevice(i));
            ncclRes = ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], 
                                   size, ncclFloat, ncclSum, comms[i], streams[i]);
            if (ncclRes != ncclSuccess) {
                printf("[COMM THREAD] ERROR: ncclAllReduce failed for rank %d: %s\n", i, ncclGetErrorString(ncclRes));
                test_failed = 1;
                break;
            }
        }
        
        if (test_failed) break;
        
        ncclRes = ncclGroupEnd();
        if (ncclRes != ncclSuccess) {
            printf("[COMM THREAD] ERROR: ncclGroupEnd failed: %s\n", ncclGetErrorString(ncclRes));
            test_failed = 1;
            break;
        }
        
        // Wait for completion
        for (int i = 0; i < nFirst; ++i) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaStreamSynchronize(streams[i]));
        }
        
        // Verify results (only occasionally to reduce output)
        if (iteration % 5 == 1) {
            float value;
            for (int i = 0; i < nFirst; ++i) {
                CUDACHECK(cudaSetDevice(i));
                CUDACHECK(cudaMemcpy(&value, &recvbuff[i][0], sizeof(float), cudaMemcpyDeviceToHost));
                printf("[COMM THREAD] Iteration %d, Rank %d result: %f\n", iteration, i, value);
            }
        }
        
        // Calculate iteration time
        clock_gettime(CLOCK_MONOTONIC, &iter_end);
        long iter_seconds = iter_end.tv_sec - iter_start.tv_sec;
        long iter_nanoseconds = iter_end.tv_nsec - iter_start.tv_nsec;
        long iter_elapsed = (iter_seconds * 1000000) + (iter_nanoseconds / 1000);
        total_communication_time += iter_elapsed;
        communication_iterations = iteration;
        
        printf("[COMM THREAD] Iteration %d completed in %ld us\n", iteration, iter_elapsed);
        
        // Small delay to allow addition thread to work
        usleep(100000); // 100ms
    }
    
    clock_gettime(CLOCK_MONOTONIC, &comm_end_time);
    communication_running = 0;
    printf("[COMM THREAD] Communication thread finished after %d iterations\n", iteration);
    
    long comm_seconds = comm_end_time.tv_sec - comm_start_time.tv_sec;
    long comm_nanoseconds = comm_end_time.tv_nsec - comm_start_time.tv_nsec;
    long total_comm_elapsed = (comm_seconds * 1000000) + (comm_nanoseconds / 1000);
    printf("[COMM THREAD] Total communication time: %ld us (%.3f ms)\n", total_comm_elapsed, total_comm_elapsed / 1000.0);
    printf("[COMM THREAD] Average per iteration: %ld us\n", iteration > 0 ? total_communication_time / iteration : 0);
    
    return NULL;
}

// Thread function for adding new rank
void* addition_thread(void* arg) {
    printf("[ADD THREAD] Starting rank addition process...\n");
    
    // Wait a bit to ensure communication is running
    usleep(200000); // 200ms
    
    if (!communication_running) {
        printf("[ADD THREAD] ERROR: Communication not running when addition started\n");
        test_failed = 1;
        return NULL;
    }
    
    printf("[ADD THREAD] Communication is running, proceeding with rank addition...\n");
    
    struct timespec start, end;
    ncclCommInfo *exportedInfo = (ncclCommInfo *)malloc(sizeof(ncclCommInfo));
    ncclNewRankInfo *newRankInfo = (ncclNewRankInfo *)malloc(sizeof(ncclNewRankInfo));
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Step 1: Export communicator info from any healthy rank
    printf("[ADD THREAD] Step 1: Exporting communicator info...\n");
    // Set device context for rank 0 before exporting
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaDeviceSynchronize());
    
    // Use the same uniqueId that was used for initial communicator setup
    NCCLCHECK(ncclCommExportInfo(comms[0], &global_uniqueId, exportedInfo));
    
    // Step 2: Initialize new rank with exported communicator info
    printf("[ADD THREAD] Step 2: Initializing new rank...\n");
    // Ensure proper device context and memory allocation for new rank
    CUDACHECK(cudaSetDevice(rank_new));
    CUDACHECK(cudaDeviceSynchronize());
    
    // Check if device is available and has sufficient memory
    size_t free_mem, total_mem;
    CUDACHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("[ADD THREAD] Device %d memory: %zu MB free / %zu MB total\n", 
           rank_new, free_mem / (1024*1024), total_mem / (1024*1024));
    
    NCCLCHECK(ncclCommInitNewRank(comms + rank_new, exportedInfo, newRankInfo));
    
    // Step 3: Update metadata with new rank's info for previous ranks
    printf("[ADD THREAD] Step 3: Updating metadata for old ranks...\n");
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nFirst; i++) {
        NCCLCHECK(ncclCommAddNewRank(comms[i], newRankInfo));
    }
    NCCLCHECK(ncclGroupEnd());
    
    // Step 4: Setup connections for each rank
    printf("[ADD THREAD] Step 4: Setting up connections...\n");
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nTotal; i++) {
        NCCLCHECK(ncclCommSetupNewRank(comms[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    long elapsed = (seconds * 1000000) + (nanoseconds / 1000);
    printf("[ADD THREAD] Rank addition completed in %ld us (%.3f ms)\n", elapsed, elapsed / 1000.0);
    
    // Verify communication is still running
    if (communication_running) {
        printf("[ADD THREAD] SUCCESS: Communication was still running during rank addition!\n");
    } else {
        printf("[ADD THREAD] WARNING: Communication stopped during rank addition\n");
    }
    
    addition_completed = 1;
    
    free(exportedInfo);
    free(newRankInfo);
    
    printf("[ADD THREAD] Addition thread finished\n");
    return NULL;
}

int main(int argc, char *argv[]) {
    printf("=== Concurrent Addition Test ===\n");
    printf("Testing concurrent communication and rank addition...\n");
    
    // Initialize parameters
    if (argc > 1) {
        sscanf(argv[argc - 1], "%d", &nTotal);
        nFirst = nTotal - 1;
        rank_new = nFirst;
    }
    printf("Initial ranks: %d, Total after addition: %d, New rank: %d\n", nFirst, nTotal, rank_new);
    
    // Check available CUDA devices
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    printf("Available CUDA devices: %d\n", deviceCount);
    
    if (nTotal > deviceCount) {
        printf("ERROR: Requested %d ranks but only %d CUDA devices available\n", nTotal, deviceCount);
        exit(EXIT_FAILURE);
    }
    
    // Allocate and initialize device buffers
    sendbuff = (float **)malloc(nTotal * sizeof(float *));
    recvbuff = (float **)malloc(nTotal * sizeof(float *));
    streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nTotal);
    
    // Initialize all devices first (including the new rank device)
    for (int i = 0; i < nTotal; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaDeviceSynchronize());
        
        // Check device properties
        cudaDeviceProp prop;
        CUDACHECK(cudaGetDeviceProperties(&prop, i));
        printf("Device %d: %s, Memory: %.1f MB\n", i, prop.name, prop.totalGlobalMem / (1024.0 * 1024.0));
    }
    
    float value = 0.0;
    for (int i = 0; i < nTotal; ++i) {
        value += 1.0;
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc((void **)sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(streams + i));
        CUDACHECK(cudaMemcpy(&sendbuff[i][0], &value, sizeof(float), cudaMemcpyHostToDevice));
        CUDACHECK(cudaDeviceSynchronize());
    }
    
    // Initialize NCCL for initial ranks
    printf("Initializing NCCL for initial %d ranks...\n", nFirst);
    NCCLCHECK(ncclGetUniqueId(&global_uniqueId));
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nFirst; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommInitRank(comms + i, nFirst, global_uniqueId, i));
    }
    NCCLCHECK(ncclGroupEnd());
    
    // Perform initial communication to verify setup
    printf("Performing initial communication test...\n");
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nFirst; ++i) {
        NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], 
                               size, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    
    for (int i = 0; i < nFirst; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    printf("Initial communication test completed successfully\n");
    
    // Create threads for concurrent test
    pthread_t comm_thread, add_thread;
    
    printf("\n=== Starting Concurrent Test ===\n");
    clock_gettime(CLOCK_MONOTONIC, &test_start_time);
    
    // Start communication thread
    if (pthread_create(&comm_thread, NULL, communication_thread, NULL) != 0) {
        printf("Failed to create communication thread\n");
        exit(EXIT_FAILURE);
    }
    
    // Start addition thread
    if (pthread_create(&add_thread, NULL, addition_thread, NULL) != 0) {
        printf("Failed to create addition thread\n");
        exit(EXIT_FAILURE);
    }
    
    // Wait for both threads to complete
    pthread_join(add_thread, NULL);
    pthread_join(comm_thread, NULL);
    
    clock_gettime(CLOCK_MONOTONIC, &test_end_time);
    
    printf("\n=== Concurrent Test Results ===\n");
    
    // Calculate total test time
    long test_seconds = test_end_time.tv_sec - test_start_time.tv_sec;
    long test_nanoseconds = test_end_time.tv_nsec - test_start_time.tv_nsec;
    long total_test_time = (test_seconds * 1000000) + (test_nanoseconds / 1000);
    
    printf("=== PERFORMANCE METRICS (CONCURRENT) ===\n");
    printf("Total test time: %ld us (%.3f ms)\n", total_test_time, total_test_time / 1000.0);
    printf("Communication iterations: %d\n", communication_iterations);
    printf("Total communication time: %ld us (%.3f ms)\n", total_communication_time, total_communication_time / 1000.0);
    printf("Average communication per iteration: %ld us\n", communication_iterations > 0 ? total_communication_time / communication_iterations : 0);
    printf("Communication efficiency: %.2f%% (communication time / total time)\n", 
           total_test_time > 0 ? (double)total_communication_time / total_test_time * 100.0 : 0.0);
    
    if (test_failed) {
        printf("TEST FAILED: Error occurred during concurrent execution\n");
        return EXIT_FAILURE;
    }
    
    if (addition_completed) {
        printf("SUCCESS: Rank addition completed while communication was running!\n");
        
        // Test communication with all ranks after addition
        printf("Testing communication with all %d ranks...\n", nTotal);
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < nTotal; ++i) {
            NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], 
                                   size, ncclFloat, ncclSum, comms[i], streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());
        
        for (int i = 0; i < nTotal; ++i) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaStreamSynchronize(streams[i]));
        }
        
        // Verify final results
        for (int i = 0; i < nTotal; ++i) {
            CUDACHECK(cudaMemcpy(&value, &recvbuff[i][0], sizeof(float), cudaMemcpyDeviceToHost));
            printf("Final AllReduce result for rank %d: %f\n", i, value);
        }
        
        printf("CONCURRENT TEST PASSED: Communication and rank addition can run concurrently!\n");
    } else {
        printf("TEST INCOMPLETE: Addition did not complete\n");
        return EXIT_FAILURE;
    }
    
    // Cleanup
    for (int i = 0; i < nTotal; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
    }
    
    free(sendbuff);
    free(recvbuff);
    free(streams);
    
    printf("Test completed successfully!\n");
    return 0;
}