#include "alloc.h"
#include "bootstrap.h"
#include "checks.h"
#include "graph.h"
#include "nccl.h"
#include "group.h"
#include "enqueue.h"
#include "nccl_common.h"
#include "serialize.h"
#include <cassert>
#include <scale.h>
#include <cstdlib>
#include <unistd.h>
#include "tuner.h"
#include "transport.h"
#include "coll_net.h"

struct ncclCommAddRankAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
  // for ncclCommAddNewRank & ncclCommInitNewRank
  ncclNewRankInfo *newRankInfo;
  // for ncclCommInitNewRank
  struct ncclCommInfoInternal *peerInfo;
};

static ncclResult_t ncclCommInitNewRankFunc(struct ncclAsyncJob *job_) {
  ncclResult_t result = ncclSuccess;
  struct ncclCommAddRankAsyncJob *job = (struct ncclCommAddRankAsyncJob *)job_;
  ncclComm_t comm = job->comm;
  int cudaDev = comm->cudaDev;
  ncclNewRankInfo *newRankInfo = job->newRankInfo;
  ncclComm_t peerCommInfo = job->peerInfo->comm;
  ncclUniqueId *commId = job->peerInfo->uniqueId;
  int nRanks = peerCommInfo->nRanks + 1;
  int myRank = nRanks - 1;
  size_t maxLocalSizeBytes = 0;
  int cudaArch;
  int archMajor, archMinor;
  unsigned long long commIdHash;
  struct ncclNewRankInfoInternal info;
  struct bootstrapState *state = (struct bootstrapState *)peerCommInfo->bootstrap;

  CUDACHECKGOTO(cudaSetDevice(cudaDev), result, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev), result, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev), result, fail);
  cudaArch = 100 * archMajor + 10 * archMinor;

  NCCLCHECK(ncclInitKernelsForDevice(cudaArch, &maxLocalSizeBytes));
  // Set the maximum kernel stack size of all kernels to avoid
  // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zu", maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));
  }

  NCCLCHECKGOTO(commAlloc(comm, NULL, nRanks, myRank), result, fail);
  // obtain a unique hash using the first commId
  comm->commHash = getHash(commId->internal, NCCL_UNIQUE_ID_BYTES);
  commIdHash = hashUniqueId(*commId);
  INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init START", __func__,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
  NCCLCHECKGOTO(bootstrapInitNew(comm, state), result, fail);
  comm->cudaArch = cudaArch;

  NCCLCHECKGOTO(initTransportsNewRank(comm, peerCommInfo), result, fail);
  NCCLCHECKGOTO(ncclTunerPluginLoad(comm), result, fail);
  if (comm->tuner) {
    NCCLCHECK(comm->tuner->init(comm->nRanks, comm->nNodes, ncclDebugLog, &comm->tunerContext));
  }
  comm->initState = ncclSuccess;

  info.comm = comm;
  ncclInfoSerialize((char *)newRankInfo, &info);

exit:
  return result;
fail:
  comm->initState = result;
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommInitNewRank, ncclComm_t* comm, ncclCommInfo* commInfo, ncclNewRankInfo* newRankInfo);
ncclResult_t ncclCommInitNewRank(ncclComm_t* newcomm, ncclCommInfo* commInfo, ncclNewRankInfo* newRankInfo) {
  ncclResult_t result = ncclSuccess;
  ncclCommInfoInternal *peerInfo = (ncclCommInfoInternal *)commInfo->internal;
  int cudaDev = -1;
  ncclComm_t comm = NULL;

  ncclInfoDeserialize(peerInfo);
  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  CUDACHECKGOTO(cudaGetDevice(&cudaDev), result, fail);
  // first call ncclInit, this will setup the environment
  NCCLCHECKGOTO(ncclInit(), result, fail);

  // Make sure the CUDA runtime is initialized.
  CUDACHECKGOTO(cudaFree(NULL), result, fail);

  NCCLCHECKGOTO(ncclCalloc(&comm, 1), result, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlag, 1), result, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->abortFlagDev, 1), result, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlagRefCount, 1), result, fail);
  comm->startMagic = comm->endMagic = NCCL_MAGIC; // Used to detect comm corruption.
  *comm->abortFlagRefCount = 1;
  comm->cudaDev = cudaDev;
  NCCLCHECKGOTO(parseCommConfig(comm, &config), result, fail);
  /* start with ncclInternalError and will be changed to ncclSuccess if init succeeds. */
  comm->initState = ncclInternalError;
  *newcomm = comm;

  struct ncclCommAddRankAsyncJob *job;
  NCCLCHECKGOTO(ncclCalloc(&job, 1), result, fail);
  job->comm = comm;
  job->newRankInfo = newRankInfo;
  job->peerInfo = peerInfo;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommInitNewRankFunc, NULL, free, comm), result, fail);

exit:
  return ncclGroupErrCheck(result);
fail:
  if (comm) {
    free(comm->abortFlag);
    if (comm->abortFlagDev) (void)ncclCudaHostFree((void*)comm->abortFlagDev);
    free(comm->abortFlagRefCount);
    free(comm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

static ncclResult_t ncclCommAddNewRankFunc(struct ncclAsyncJob *job_) {
  ncclResult_t result = ncclSuccess;
  struct ncclCommAddRankAsyncJob *job = (struct ncclCommAddRankAsyncJob *)job_;
  ncclComm_t comm = job->base.comm;
  int cudaDev = comm->cudaDev;
  struct ncclNewRankInfoInternal *newRankInfo = (ncclNewRankInfoInternal *)job->newRankInfo->internal;
  ncclComm_t newRankComm = newRankInfo->comm;
  struct bootstrapState *state = (struct bootstrapState *)comm->bootstrap;
  struct bootstrapState *newRankState = (struct bootstrapState *)newRankComm->bootstrap;
  struct bootstrapState *pendingState;

  CUDACHECKGOTO(cudaSetDevice(cudaDev), result, fail);

  // Stage new rank data instead of directly modifying comm structures
  pthread_mutex_lock(&comm->commStateLock);
  
  if (comm->staging.hasPendingRank) {
    pthread_mutex_unlock(&comm->commStateLock);
    WARN("Another rank addition is already in progress");
    result = ncclInvalidUsage;
    goto fail;
  }

  // Set staging flag
  comm->staging.hasPendingRank = true;
  comm->staging.pendingNRanks = comm->nRanks + 1;
  
  // Stage bootstrap state
  NCCLCHECKGOTO(ncclCalloc(&comm->staging.pendingBootstrapState, 1), result, fail_unlock);
  pendingState = (struct bootstrapState *)comm->staging.pendingBootstrapState;
  pendingState->nranks = comm->staging.pendingNRanks;
  
  // Copy existing bootstrap addresses and add new rank
  NCCLCHECKGOTO(ncclCalloc(&pendingState->peerP2pAddresses, comm->staging.pendingNRanks * sizeof(union ncclSocketAddress)), result, fail_unlock);
  NCCLCHECKGOTO(ncclCalloc(&pendingState->peerProxyAddresses, comm->staging.pendingNRanks * sizeof(union ncclSocketAddress)), result, fail_unlock);
  NCCLCHECKGOTO(ncclCalloc(&pendingState->peerProxyAddressesUDS, comm->staging.pendingNRanks * sizeof(uint64_t)), result, fail_unlock);
  
  memcpy(pendingState->peerP2pAddresses, state->peerP2pAddresses, comm->nRanks * sizeof(*state->peerP2pAddresses));
  memcpy(pendingState->peerProxyAddresses, state->peerProxyAddresses, comm->nRanks * sizeof(*state->peerProxyAddresses));
  memcpy(pendingState->peerProxyAddressesUDS, state->peerProxyAddressesUDS, comm->nRanks * sizeof(*state->peerProxyAddressesUDS));
  
  pendingState->peerP2pAddresses[comm->nRanks] = newRankState->peerP2pAddresses[comm->nRanks];
  pendingState->peerProxyAddresses[comm->nRanks] = newRankState->peerProxyAddresses[comm->nRanks];
  pendingState->peerProxyAddressesUDS[comm->nRanks] = newRankState->peerProxyAddressesUDS[comm->nRanks];

  // Stage peerInfo
  NCCLCHECKGOTO(ncclCalloc(&comm->staging.pendingPeerInfo, comm->staging.pendingNRanks + 1), result, fail_unlock);
  memcpy(comm->staging.pendingPeerInfo, comm->peerInfo, (comm->nRanks + 1) * sizeof(*comm->peerInfo));
  comm->staging.pendingPeerInfo[comm->staging.pendingNRanks] = comm->staging.pendingPeerInfo[comm->nRanks];
  comm->staging.pendingPeerInfo[comm->nRanks] = newRankComm->peerInfo[comm->nRanks];

  // Stage graphs
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    comm->staging.pendingGraphs[i] = newRankComm->graphs[i];
  }

  // Mark staging as ready for activation
  comm->staging.activationReady = true;
  
  pthread_mutex_unlock(&comm->commStateLock);

exit:
  return result;
fail_unlock:
  pthread_mutex_unlock(&comm->commStateLock);
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommAddNewRank, ncclComm_t comm, ncclNewRankInfo* newRankInfo);
ncclResult_t ncclCommAddNewRank(ncclComm_t comm, ncclNewRankInfo* newRankInfo) {
  ncclResult_t result = ncclSuccess;
  ncclNewRankInfoInternal *info = (ncclNewRankInfoInternal *)newRankInfo->internal;
  ncclInfoDeserialize(info);

  struct ncclCommAddRankAsyncJob *job;
  NCCLCHECKGOTO(ncclCalloc(&job, 1), result, fail);
  job->comm = comm;
  job->newRankInfo = newRankInfo;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommAddNewRankFunc, NULL, free, comm), result, fail);

ret:
  return result;
fail:
  goto ret;
}

NCCL_API(ncclResult_t, ncclCommExportInfo, ncclComm_t comm, ncclUniqueId* commId, ncclCommInfo* commInfo);
ncclResult_t ncclCommExportInfo(ncclComm_t comm, ncclUniqueId* commId, ncclCommInfo* commInfo) {
  struct ncclCommInfoInternal info{comm, commId};
  int offset = ncclInfoSerialize(commInfo->internal, &info);
  assert(offset <= sizeof(ncclCommInfo));
  return ncclSuccess;
}

static ncclResult_t ncclCommSetupNewRankFunc(struct ncclAsyncJob *job_) {
  ncclResult_t res = ncclSuccess;
  struct ncclCommAddRankAsyncJob *job = (struct ncclCommAddRankAsyncJob *)job_;
  ncclComm_t comm = job->base.comm;
  int cudaDev = comm->cudaDev;
  int rank = comm->rank;
  int nRanks = comm->nRanks;
  bool isNewRank;

  CUDACHECKGOTO(cudaSetDevice(cudaDev), res, fail);
  
  // Simple decoupling: set rank addition in progress flag
  comm->rankAddInProgress = true;
  
  // Wait for active communication operations to complete
  while (comm->activeCommOps > 0) {
    usleep(1000); // 1ms wait
  }

  // Check if this is a new rank (doesn't have staged data) or existing rank (has staged data)
  pthread_mutex_lock(&comm->commStateLock);
  
  isNewRank = !(comm->staging.hasPendingRank && comm->staging.activationReady);
  
  if (!isNewRank) {
    // This is an existing rank with staged data - activate it
    printf("DEBUG: Activating staged data for existing rank\n");
    
    // Declare variables before any potential goto
    struct bootstrapState *state;
    struct bootstrapState *pendingState;
    int oldNRanks;
    
    // Atomically activate staged data
    printf("DEBUG: Getting bootstrap states\n");
    state = (struct bootstrapState *)comm->bootstrap;
    pendingState = (struct bootstrapState *)comm->staging.pendingBootstrapState;
    printf("DEBUG: Bootstrap states obtained\n");
    
    // Update nRanks and bootstrap state
    oldNRanks = comm->nRanks;
    comm->nRanks = comm->staging.pendingNRanks;
    nRanks = comm->nRanks;
    
    // Replace bootstrap addresses
    free(state->peerP2pAddresses);
    free(state->peerProxyAddresses);
    free(state->peerProxyAddressesUDS);
    state->nranks = pendingState->nranks;
    state->peerP2pAddresses = pendingState->peerP2pAddresses;
    state->peerProxyAddresses = pendingState->peerProxyAddresses;
    state->peerProxyAddressesUDS = pendingState->peerProxyAddressesUDS;
    
    // Nullify transferred pointers to prevent double-free
    pendingState->peerP2pAddresses = NULL;
    pendingState->peerProxyAddresses = NULL;
    pendingState->peerProxyAddressesUDS = NULL;
    
    // Replace peerInfo
    free(comm->peerInfo);
    comm->peerInfo = comm->staging.pendingPeerInfo;
    
    // Replace graphs
    for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
      comm->graphs[i] = comm->staging.pendingGraphs[i];
    }
    
    // Update channels for new rank
    printf("DEBUG: Updating channels for new rank\n");
    NCCLCHECKGOTO(ncclRealloc(&comm->connectSend, oldNRanks, nRanks), res, fail_unlock);
    NCCLCHECKGOTO(ncclRealloc(&comm->connectRecv, oldNRanks, nRanks), res, fail_unlock);
    comm->connectSend[nRanks - 1] = comm->connectRecv[nRanks - 1] = 0;
    printf("DEBUG: Starting channel loop\n");
    
    for (int c = 0; c < comm->nChannels; c++) {
      printf("DEBUG: Processing channel %d\n", c);
      struct ncclChannel *channel = comm->channels + c;
      printf("DEBUG: Got channel pointer\n");
      int &prev = channel->ring.prev;
      int &next = channel->ring.next;
      printf("DEBUG: Got prev=%d, next=%d\n", prev, next);
      if (prev == nRanks - 2) {
        printf("DEBUG: Updating prev connection\n");
        prev = nRanks - 1;
        NCCLCHECKGOTO(ncclTransportP2pForcedConnect(comm, c, 1, &prev, 0, nullptr, 0), res, fail_unlock);
      } else if (next == 0) {
        printf("DEBUG: Updating next connection\n");
        next = nRanks - 1;
        NCCLCHECKGOTO(ncclTransportP2pForcedConnect(comm, c, 0, nullptr, 1, &next, 0), res, fail_unlock);
      }
      printf("DEBUG: Channel %d processed\n", c);
    }
    
    // Clear staging area
    // Note: Don't free pendingBootstrapState as its pointers have been transferred to state
    // Just free the structure itself, not the pointers it contained
    free(comm->staging.pendingBootstrapState);
    comm->staging.hasPendingRank = false;
    comm->staging.pendingNRanks = 0;
    // Note: pendingPeerInfo ownership has been transferred to comm->peerInfo
    comm->staging.pendingPeerInfo = NULL;
    comm->staging.pendingChannels = NULL;
    memset(comm->staging.pendingGraphs, 0, sizeof(comm->staging.pendingGraphs));
    comm->staging.pendingBootstrapState = NULL;
    comm->staging.activationReady = false;
  }
  
  pthread_mutex_unlock(&comm->commStateLock);

  printf("DEBUG: About to setup transport, rank=%d, nRanks=%d\n", rank, nRanks);
  if (rank == nRanks - 1) {
    // New rank (last rank) needs full setup
    printf("DEBUG: New rank - calling ncclTransportP2pSetup\n");
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), res, fail);
    printf("DEBUG: New rank - calling devCommSetup\n");
    NCCLCHECKGOTO(devCommSetup(comm), res, fail);
    printf("DEBUG: New rank setup completed\n");
  } else {
    // Existing ranks need resetup to accommodate new rank
    printf("DEBUG: Existing rank - calling ncclTransportP2pSetup\n");
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), res, fail);
    printf("DEBUG: Existing rank - calling devCommResetup\n");
    NCCLCHECKGOTO(devCommResetup(comm), res, fail);
    printf("DEBUG: Existing rank resetup completed\n");
  }

exit:
  // Clear rank addition in progress flag
  comm->rankAddInProgress = false;
  return res;
fail_unlock:
  pthread_mutex_unlock(&comm->commStateLock);
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommSetupNewRank, ncclComm_t comm);
ncclResult_t ncclCommSetupNewRank(ncclComm_t comm) {
  ncclResult_t res = ncclSuccess;
  struct ncclCommAddRankAsyncJob *job;
  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommSetupNewRankFunc, NULL, free, comm), res, fail);

exit:
  return res;
fail:
  goto exit;
}