# NCCL Two-Phase Activation Test Plan

## Overview

This document outlines the comprehensive testing strategy for the NCCL two-phase activation feature, which enables parallel scaling by allowing connectors to be initialized in an inactive state and then selectively activated.

## Test Objectives

1. **Functional Verification**: Ensure the two-phase activation mechanism works correctly
2. **Performance Validation**: Verify that the feature doesn't introduce performance regressions
3. **Error Handling**: Test robustness against invalid inputs and edge cases
4. **Backward Compatibility**: Ensure existing NCCL functionality remains intact

## Test Architecture

### Phase 1: Connector Initialization
- Connectors are created but remain inactive (`isActive = 0`)
- Traditional initialization sets all connectors to active (`isActive = 1`)
- Two-phase initialization leaves connectors inactive until explicit activation

### Phase 2: Selective Activation
- Use `ncclActivateConnectors()` API to activate specific ranks
- Only activated connectors participate in communication
- Progressive activation allows dynamic scaling

## Test Suite Components

### 1. Basic Functionality Tests

#### Test 1.1: Basic Two-Phase Activation
**Objective**: Verify basic two-phase activation workflow
**Steps**:
1. Initialize 4 communicators with inactive connectors
2. Activate connectors for ranks 0 and 2
3. Perform AllReduce operation with activated ranks
4. Verify communication succeeds only for activated ranks

**Expected Result**: Communication works for ranks 0 and 2, others remain inactive

#### Test 1.2: Progressive Activation
**Objective**: Test incremental activation of connectors
**Steps**:
1. Initialize 4 communicators
2. Activate ranks 0-1, test communication
3. Activate ranks 0-2, test communication
4. Activate all ranks 0-3, test communication

**Expected Result**: Each step should work with the specified active ranks

#### Test 1.3: Full Activation
**Objective**: Verify all connectors can be activated
**Steps**:
1. Initialize N communicators
2. Activate all ranks at once
3. Test full collective operations

**Expected Result**: Behavior identical to traditional initialization

### 2. Error Handling Tests

#### Test 2.1: Invalid Rank Handling
**Objective**: Test rejection of invalid rank numbers
**Test Cases**:
- Negative rank numbers
- Rank numbers >= nRanks
- Duplicate ranks in activation list

**Expected Result**: `ncclInvalidArgument` error returned

#### Test 2.2: Invalid Parameters
**Objective**: Test parameter validation
**Test Cases**:
- NULL communicator
- NULL ranks array
- Zero or negative nranks

**Expected Result**: Appropriate error codes returned

#### Test 2.3: Uninitialized Communicator
**Objective**: Test activation on uninitialized communicator
**Expected Result**: `ncclInvalidArgument` error

### 3. Performance Tests

#### Test 3.1: Initialization Overhead
**Objective**: Measure initialization time difference
**Metrics**:
- Traditional initialization time
- Two-phase initialization time
- Activation time per rank

**Expected Result**: Two-phase initialization should be faster initially

#### Test 3.2: Communication Performance
**Objective**: Verify no performance regression in communication
**Metrics**:
- Latency comparison (traditional vs two-phase)
- Bandwidth comparison
- Scaling efficiency

**Expected Result**: No significant performance difference after activation

#### Test 3.3: Memory Usage
**Objective**: Compare memory consumption
**Metrics**:
- Memory usage before activation
- Memory usage after activation
- Memory overhead per inactive connector

### 4. Integration Tests

#### Test 4.1: Mixed Mode Operation
**Objective**: Test interaction between traditional and two-phase initialization
**Steps**:
1. Initialize some communicators traditionally
2. Initialize others with two-phase
3. Test inter-communicator operations

#### Test 4.2: Dynamic Scaling Simulation
**Objective**: Simulate real-world dynamic scaling scenario
**Steps**:
1. Start with subset of ranks
2. Progressively add more ranks
3. Perform collective operations at each step
4. Measure scaling efficiency

### 5. Stress Tests

#### Test 5.1: Large Scale Test
**Objective**: Test with many ranks (if hardware allows)
**Configuration**: 8+ GPUs if available
**Metrics**: Activation time scaling, memory usage

#### Test 5.2: Repeated Activation
**Objective**: Test multiple activation cycles
**Steps**:
1. Activate subset of ranks
2. Deactivate (if supported) or reinitialize
3. Activate different subset
4. Repeat multiple times

## Test Environment Requirements

### Hardware Requirements
- Multi-GPU system (minimum 4 GPUs recommended)
- CUDA-capable GPUs
- Sufficient GPU memory for test data

### Software Requirements
- CUDA Toolkit
- Modified NCCL with two-phase activation support
- C++ compiler with C++11 support

### Environment Variables
```bash
export CUDA_HOME=/usr/local/cuda
export NCCL_HOME=/path/to/flexible-ccl/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Running the Tests

### Build Tests
```bash
cd test/
make all
```

### Run Individual Tests
```bash
# Basic functionality test
./two_phase_activation_test

# Existing addition test (for comparison)
./addition

# Performance comparison
./addition 4  # Traditional approach
```

### Automated Test Suite
```bash
make test
```

## Test Data Collection

### Metrics to Collect
1. **Timing Data**:
   - Initialization time
   - Activation time
   - Communication latency
   - Throughput

2. **Resource Usage**:
   - Memory consumption
   - GPU utilization
   - CPU usage

3. **Correctness Data**:
   - Communication results verification
   - Error code validation
   - State consistency checks

### Data Analysis
- Compare performance metrics between traditional and two-phase approaches
- Analyze scaling characteristics
- Identify performance bottlenecks
- Validate correctness across all test scenarios

## Success Criteria

### Functional Requirements
- ✅ All basic functionality tests pass
- ✅ Error handling works correctly
- ✅ No crashes or memory leaks
- ✅ Backward compatibility maintained

### Performance Requirements
- ✅ No significant performance regression (< 5% overhead)
- ✅ Initialization time improvement for partial activation
- ✅ Linear scaling with number of activated ranks

### Quality Requirements
- ✅ Code coverage > 90% for new functionality
- ✅ All edge cases handled gracefully
- ✅ Comprehensive error reporting

## Test Execution Schedule

1. **Phase 1** (Basic Tests): 1-2 days
   - Implement and run basic functionality tests
   - Verify core two-phase activation works

2. **Phase 2** (Error Handling): 1 day
   - Implement error handling tests
   - Verify robustness

3. **Phase 3** (Performance): 2-3 days
   - Implement performance benchmarks
   - Collect and analyze performance data

4. **Phase 4** (Integration): 1-2 days
   - Run integration and stress tests
   - Validate in realistic scenarios

5. **Phase 5** (Documentation): 1 day
   - Document results
   - Create performance reports

## Risk Mitigation

### Potential Issues
1. **Hardware Limitations**: Limited GPU availability
   - Mitigation: Use smaller scale tests, simulate larger configurations

2. **Performance Regressions**: Unexpected overhead
   - Mitigation: Profile code, optimize critical paths

3. **Compatibility Issues**: Breaking existing functionality
   - Mitigation: Extensive regression testing

### Contingency Plans
- Fallback to smaller test configurations if hardware unavailable
- Performance optimization phase if regressions detected
- Rollback plan if critical issues discovered

## Reporting

### Test Reports
- Daily progress reports during testing phase
- Detailed test results with pass/fail status
- Performance analysis with graphs and metrics
- Final validation report with recommendations

### Documentation Updates
- Update API documentation with test results
- Create user guide with best practices
- Document known limitations and workarounds