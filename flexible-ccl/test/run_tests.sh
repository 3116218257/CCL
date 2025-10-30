#!/bin/bash

# NCCL Two-Phase Activation Test Runner
# This script builds and runs all tests for the two-phase activation feature

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="../build"
LOG_DIR="$TEST_DIR/logs"
RESULTS_FILE="$LOG_DIR/test_results.txt"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo
    print_status $BLUE "=================================================="
    print_status $BLUE "$1"
    print_status $BLUE "=================================================="
    echo
}

print_success() {
    print_status $GREEN "✅ $1"
}

print_error() {
    print_status $RED "❌ $1"
}

print_warning() {
    print_status $YELLOW "⚠️  $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check CUDA
    NVCC_PATH=""
    if command -v nvcc &> /dev/null; then
        NVCC_PATH=$(which nvcc)
    elif [ -f "/usr/local/cuda/bin/nvcc" ]; then
        NVCC_PATH="/usr/local/cuda/bin/nvcc"
        export PATH="/usr/local/cuda/bin:$PATH"
    elif [ -f "/usr/local/cuda-11.8/bin/nvcc" ]; then
        NVCC_PATH="/usr/local/cuda-11.8/bin/nvcc"
        export PATH="/usr/local/cuda-11.8/bin:$PATH"
    elif [ -f "/usr/local/cuda-11.7/bin/nvcc" ]; then
        NVCC_PATH="/usr/local/cuda-11.7/bin/nvcc"
        export PATH="/usr/local/cuda-11.7/bin:$PATH"
    fi
    
    if [ -n "$NVCC_PATH" ]; then
        CUDA_VERSION=$($NVCC_PATH --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        print_success "CUDA found: version $CUDA_VERSION at $NVCC_PATH"
    else
        print_error "CUDA not found. Please install CUDA toolkit."
        exit 1
    fi
    
    # Check if NCCL is built
    if [ -f "$BUILD_DIR/lib/libnccl.so" ]; then
        print_success "NCCL library found"
    else
        print_warning "NCCL library not found. Building NCCL first..."
        cd ..
        make -j$(nproc)
        cd "$TEST_DIR"
        if [ -f "$BUILD_DIR/lib/libnccl.so" ]; then
            print_success "NCCL built successfully"
        else
            print_error "Failed to build NCCL"
            exit 1
        fi
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        print_success "Found $GPU_COUNT GPU(s)"
        if [ $GPU_COUNT -lt 2 ]; then
            print_warning "At least 2 GPUs recommended for comprehensive testing"
        fi
    else
        print_error "nvidia-smi not found. GPU testing may not work."
    fi
}

# Function to build tests
build_tests() {
    print_header "Building Tests"
    
    cd "$TEST_DIR"
    
    # Clean previous builds
    make clean &> /dev/null || true
    
    # Build all tests
    if make all 2>&1 | tee "$LOG_DIR/build.log"; then
        print_success "All tests built successfully"
    else
        print_error "Failed to build tests. Check $LOG_DIR/build.log for details."
        exit 1
    fi
}

# Function to run a single test
run_test() {
    local test_name=$1
    local test_executable=$2
    local timeout=${3:-60}  # Default 60 seconds timeout
    
    print_status $BLUE "Running $test_name..."
    
    local log_file="$LOG_DIR/${test_name}.log"
    local start_time=$(date +%s)
    
    if timeout $timeout ./$test_executable > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "$test_name completed in ${duration}s"
        echo "PASS: $test_name (${duration}s)" >> "$RESULTS_FILE"
        return 0
    else
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        if [ $exit_code -eq 124 ]; then
            print_error "$test_name timed out after ${timeout}s"
            echo "TIMEOUT: $test_name (${timeout}s)" >> "$RESULTS_FILE"
        else
            print_error "$test_name failed with exit code $exit_code"
            echo "FAIL: $test_name (exit code: $exit_code)" >> "$RESULTS_FILE"
        fi
        
        # Show last few lines of log for debugging
        print_warning "Last 10 lines of $test_name log:"
        tail -n 10 "$log_file" | sed 's/^/  /'
        return 1
    fi
}

# Function to run all tests
run_tests() {
    print_header "Running Tests"
    
    cd "$TEST_DIR"
    
    # Initialize results file
    echo "NCCL Two-Phase Activation Test Results" > "$RESULTS_FILE"
    echo "Date: $(date)" >> "$RESULTS_FILE"
    echo "=======================================" >> "$RESULTS_FILE"
    
    local total_tests=0
    local passed_tests=0
    
    # Test 1: Two-Phase Activation Test
    if [ -f "two_phase_activation_test" ]; then
        total_tests=$((total_tests + 1))
        if run_test "Two-Phase Activation Test" "two_phase_activation_test" 120; then
            passed_tests=$((passed_tests + 1))
        fi
    else
        print_warning "two_phase_activation_test not found, skipping"
    fi
    
    # Test 2: Addition Test (for comparison)
    if [ -f "addition" ]; then
        total_tests=$((total_tests + 1))
        if run_test "Addition Test" "addition" 60; then
            passed_tests=$((passed_tests + 1))
        fi
    else
        print_warning "addition test not found, skipping"
    fi
    
    # Test 3: Removal Test (if exists)
    if [ -f "removal" ]; then
        total_tests=$((total_tests + 1))
        if run_test "Removal Test" "removal" 60; then
            passed_tests=$((passed_tests + 1))
        fi
    else
        print_warning "removal test not found, skipping"
    fi
    
    # Summary
    echo >> "$RESULTS_FILE"
    echo "Summary: $passed_tests/$total_tests tests passed" >> "$RESULTS_FILE"
    
    print_header "Test Summary"
    if [ $passed_tests -eq $total_tests ]; then
        print_success "All tests passed! ($passed_tests/$total_tests)"
        return 0
    else
        print_error "Some tests failed. ($passed_tests/$total_tests passed)"
        return 1
    fi
}

# Function to generate test report
generate_report() {
    print_header "Generating Test Report"
    
    local report_file="$LOG_DIR/test_report.md"
    
    cat > "$report_file" << EOF
# NCCL Two-Phase Activation Test Report

**Date**: $(date)
**Test Directory**: $TEST_DIR
**Build Directory**: $BUILD_DIR

## Environment Information

- **CUDA Version**: $(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/' || echo "Not found")
- **GPU Count**: $(nvidia-smi -L | wc -l 2>/dev/null || echo "Unknown")
- **System**: $(uname -a)

## Test Results

EOF
    
    # Add test results
    if [ -f "$RESULTS_FILE" ]; then
        echo '```' >> "$report_file"
        cat "$RESULTS_FILE" >> "$report_file"
        echo '```' >> "$report_file"
    fi
    
    # Add log file information
    echo >> "$report_file"
    echo "## Log Files" >> "$report_file"
    echo >> "$report_file"
    for log_file in "$LOG_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            local filename=$(basename "$log_file")
            echo "- [$filename](./$filename)" >> "$report_file"
        fi
    done
    
    print_success "Test report generated: $report_file"
}

# Function to cleanup
cleanup() {
    print_header "Cleanup"
    
    # Kill any remaining processes
    pkill -f "two_phase_activation_test" 2>/dev/null || true
    pkill -f "addition" 2>/dev/null || true
    pkill -f "removal" 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Main execution
main() {
    print_header "NCCL Two-Phase Activation Test Suite"
    
    # Set up trap for cleanup
    trap cleanup EXIT
    
    # Check if help is requested
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  -h, --help     Show this help message"
        echo "  --build-only   Only build tests, don't run them"
        echo "  --run-only     Only run tests, don't build them"
        echo "  --no-report    Don't generate test report"
        exit 0
    fi
    
    local build_only=false
    local run_only=false
    local generate_report_flag=true
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build-only)
                build_only=true
                shift
                ;;
            --run-only)
                run_only=true
                shift
                ;;
            --no-report)
                generate_report_flag=false
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute based on options
    if [ "$run_only" = false ]; then
        check_prerequisites
        build_tests
    fi
    
    if [ "$build_only" = false ]; then
        if ! run_tests; then
            exit 1
        fi
        
        if [ "$generate_report_flag" = true ]; then
            generate_report
        fi
    fi
    
    print_header "Test Suite Completed Successfully"
}

# Run main function with all arguments
main "$@"