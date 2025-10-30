#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",      \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",      \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Global variables
ncclComm_t* comms;
float** sendbuff;
float** recvbuff;
cudaStream_t* s;
ncclUniqueId global_uniqueId;

// Timing variables
struct timespec test_start_time, test_end_time;
struct timespec comm_start_time, comm_end_time;
long total_communication_time = 0;
int communication_iterations = 0;

// Function to perform communication rounds
void perform_communication_rounds(int nRanks, int rounds) {
    printf("[SERIAL] Starting %d communication rounds...\n", rounds);
    clock_gettime(CLOCK_MONOTONIC, &comm_start_time);
    
    for (int round = 0; round < rounds; round++) {
        printf("[SERIAL] Communication round %d\n", round + 1);
        
        struct timespec iter_start, iter_end;
        clock_gettime(CLOCK_MONOTONIC, &iter_start);
        
        // Perform AllReduce on all ranks
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < nRanks; ++i) {
            CUDACHECK(cudaSetDevice(i));
            NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], 1, ncclFloat, ncclSum, comms[i], s[i]));
        }
        NCCLCHECK(ncclGroupEnd());
        
        // Synchronize all streams
        for (int i = 0; i < nRanks; ++i) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaStreamSynchronize(s[i]));
        }
        
        // Calculate iteration time
        clock_gettime(CLOCK_MONOTONIC, &iter_end);
        long iter_seconds = iter_end.tv_sec - iter_start.tv_sec;
        long iter_nanoseconds = iter_end.tv_nsec - iter_start.tv_nsec;
        long iter_elapsed = (iter_seconds * 1000000) + (iter_nanoseconds / 1000);
        total_communication_time += iter_elapsed;
        communication_iterations = round + 1;
        
        printf("[SERIAL] Round %d completed in %ld us\n", round + 1, iter_elapsed);
        
        // Verify results (only occasionally to reduce output)
        if ((round + 1) % 5 == 1) {
            float value;
            for (int i = 0; i < nRanks; ++i) {
                CUDACHECK(cudaSetDevice(i));
                CUDACHECK(cudaMemcpy(&value, &recvbuff[i][0], sizeof(float), cudaMemcpyDeviceToHost));
                printf("[SERIAL] Round %d, Rank %d result: %f\n", round + 1, i, value);
            }
        }
        
        // Small delay to simulate realistic workload
        usleep(100000); // 100ms
    }
    
    clock_gettime(CLOCK_MONOTONIC, &comm_end_time);
    
    long comm_seconds = comm_end_time.tv_sec - comm_start_time.tv_sec;
    long comm_nanoseconds = comm_end_time.tv_nsec - comm_start_time.tv_nsec;
    long total_comm_elapsed = (comm_seconds * 1000000) + (comm_nanoseconds / 1000);
    printf("[SERIAL] Total communication time: %ld us (%.3f ms)\n", total_comm_elapsed, total_comm_elapsed / 1000.0);
    printf("[SERIAL] Average per round: %ld us\n", rounds > 0 ? total_communication_time / rounds : 0);
}

// Function to add a new rank (blocking)
void add_new_rank_blocking(int nFirst, int rank_new) {
    printf("[SERIAL] Starting rank addition (BLOCKING)...\n");
    
    struct timespec add_start, add_end;
    clock_gettime(CLOCK_MONOTONIC, &add_start);
    
    // Step 1: Export communicator info from rank 0
    printf("[SERIAL] Step 1: Exporting communicator info from rank 0\n");
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaDeviceSynchronize());
    
    ncclCommInfo *commInfo = (ncclCommInfo *)malloc(sizeof(ncclCommInfo));
    ncclNewRankInfo *newRankInfo = (ncclNewRankInfo *)malloc(sizeof(ncclNewRankInfo));
    
    NCCLCHECK(ncclCommExportInfo(comms[0], &global_uniqueId, commInfo));
    
    // Step 2: Initialize new rank
    printf("[SERIAL] Step 2: Initializing new rank %d\n", rank_new);
    CUDACHECK(cudaSetDevice(rank_new));
    CUDACHECK(cudaDeviceSynchronize());
    
    // Check available memory
    size_t free_mem, total_mem;
    CUDACHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("[SERIAL] Device %d memory: %zu MB free / %zu MB total\n", 
           rank_new, free_mem / (1024*1024), total_mem / (1024*1024));
    
    NCCLCHECK(ncclCommInitNewRank(&comms[rank_new], commInfo, newRankInfo));
    
    // Step 3: Update metadata for old ranks
    printf("[SERIAL] Step 3: Updating metadata for old ranks\n");
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nFirst; ++i) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommAddNewRank(comms[i], newRankInfo));
    }
    NCCLCHECK(ncclGroupEnd());
    
    // Step 4: Setup connections for all ranks
    printf("[SERIAL] Step 4: Setting up connections for all ranks\n");
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i <= rank_new; ++i) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommSetupNewRank(comms[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    
    clock_gettime(CLOCK_MONOTONIC, &add_end);
    
    long add_seconds = add_end.tv_sec - add_start.tv_sec;
    long add_nanoseconds = add_end.tv_nsec - add_start.tv_nsec;
    long add_elapsed = (add_seconds * 1000000) + (add_nanoseconds / 1000);
    
    printf("[SERIAL] Rank addition completed in %ld us (%.3f ms)\n", add_elapsed, add_elapsed / 1000.0);
    
    // Free allocated memory
    free(commInfo);
    free(newRankInfo);
}

int main(int argc, char* argv[]) {
    int nFirst = 2;  // Initial number of ranks
    int rank_new = 2; // New rank to add
    int nTotal = nFirst + 1; // Total ranks after addition
    
    printf("=== Serial Addition Test ===\n");
    printf("Initial ranks: %d, New rank: %d, Total after addition: %d\n", nFirst, rank_new, nTotal);
    
    // Check available CUDA devices
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    printf("Available CUDA devices: %d\n", deviceCount);
    
    if (nTotal > deviceCount) {
        printf("Error: Requested %d ranks but only %d CUDA devices available\n", nTotal, deviceCount);
        exit(EXIT_FAILURE);
    }
    
    // Initialize all devices
    for (int i = 0; i < nTotal; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaDeviceSynchronize());
    }
    
    // Allocate arrays
    comms = (ncclComm_t*)malloc(nTotal * sizeof(ncclComm_t));
    sendbuff = (float**)malloc(nTotal * sizeof(float*));
    recvbuff = (float**)malloc(nTotal * sizeof(float*));
    s = (cudaStream_t*)malloc(nTotal * sizeof(cudaStream_t));
    
    // Allocate buffers and streams for all potential ranks
    for (int i = 0; i < nTotal; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(sendbuff + i, sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + i));
        
        // Initialize send buffer with rank value
        float value = (float)i;
        CUDACHECK(cudaMemcpy(sendbuff[i], &value, sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // Initialize NCCL for initial ranks
    printf("\n=== Initializing NCCL for initial ranks ===\n");
    NCCLCHECK(ncclGetUniqueId(&global_uniqueId));
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nFirst; ++i) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommInitRank(&comms[i], nFirst, global_uniqueId, i));
    }
    NCCLCHECK(ncclGroupEnd());
    
    // Test initial communication
    printf("\n=== Testing initial communication ===\n");
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nFirst; ++i) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], 1, ncclFloat, ncclSum, comms[i], s[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    
    // Synchronize and verify
    for (int i = 0; i < nFirst; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
        
        float value;
        CUDACHECK(cudaMemcpy(&value, &recvbuff[i][0], sizeof(float), cudaMemcpyDeviceToHost));
        printf("Initial test - Rank %d result: %f\n", i, value);
    }
    
    printf("\n=== Starting Serial Test ===\n");
    clock_gettime(CLOCK_MONOTONIC, &test_start_time);
    
    // Phase 1: Communication before rank addition
    printf("\n--- Phase 1: Communication before rank addition ---\n");
    perform_communication_rounds(nFirst, 4);
    
    // Phase 2: Add new rank (BLOCKING - this stops all communication)
    printf("\n--- Phase 2: Adding new rank (BLOCKING) ---\n");
    add_new_rank_blocking(nFirst, rank_new);
    
    // Phase 3: Communication after rank addition
    printf("\n--- Phase 3: Communication after rank addition ---\n");
    perform_communication_rounds(nTotal, 1);
    
    clock_gettime(CLOCK_MONOTONIC, &test_end_time);
    
    printf("\n=== Serial Test Results ===\n");
    
    // Calculate total test time
    long test_seconds = test_end_time.tv_sec - test_start_time.tv_sec;
    long test_nanoseconds = test_end_time.tv_nsec - test_start_time.tv_nsec;
    long total_test_time = (test_seconds * 1000000) + (test_nanoseconds / 1000);
    
    printf("=== PERFORMANCE METRICS (SERIAL) ===\n");
    printf("Total test time: %ld us (%.3f ms)\n", total_test_time, total_test_time / 1000.0);
    printf("Communication rounds: %d\n", communication_iterations);
    printf("Total communication time: %ld us (%.3f ms)\n", total_communication_time, total_communication_time / 1000.0);
    printf("Average communication per round: %ld us\n", communication_iterations > 0 ? total_communication_time / communication_iterations : 0);
    printf("Communication efficiency: %.2f%% (communication time / total time)\n", 
           total_test_time > 0 ? (double)total_communication_time / total_test_time * 100.0 : 0.0);
    
    // Final verification
    printf("\n=== Final Verification ===\n");
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nTotal; ++i) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], 1, ncclFloat, ncclSum, comms[i], s[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    
    for (int i = 0; i < nTotal; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
        
        float value;
        CUDACHECK(cudaMemcpy(&value, &recvbuff[i][0], sizeof(float), cudaMemcpyDeviceToHost));
        printf("Final verification - Rank %d result: %f\n", i, value);
    }
    
    printf("SUCCESS: Serial test completed successfully!\n");
    
    // Cleanup
    for (int i = 0; i < nTotal; ++i) {
        if (i < nFirst || i == rank_new) {
            ncclCommDestroy(comms[i]);
        }
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
        CUDACHECK(cudaStreamDestroy(s[i]));
    }
    
    free(comms);
    free(sendbuff);
    free(recvbuff);
    free(s);
    
    return 0;
}