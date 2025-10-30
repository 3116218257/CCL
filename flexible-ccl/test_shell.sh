#!/bin/bash

# Flexible NCCL 编译和测试脚本
# 用于编译NCCL库、测试程序并运行动态rank添加测试

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# 配置变量
BUILD_DIR="$PROJECT_ROOT/build"
LIB_DIR="$BUILD_DIR/lib"
TEST_DIR="$BUILD_DIR/test"
NPROC=$(nproc)

export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# 寻找并设置CUDA PATH
find_cuda_path() {
    local cuda_paths=(
        "/usr/local/cuda"
        "/usr/local/cuda-12.0"
        "/usr/local/cuda-11.8"
        "/usr/local/cuda-11.7"
        "/usr/local/cuda-11.6"
        "/usr/local/cuda-11.5"
        "/usr/local/cuda-11.4"
        "/usr/local/cuda-11.3"
        "/usr/local/cuda-11.2"
        "/usr/local/cuda-11.1"
        "/usr/local/cuda-11.0"
        "/opt/cuda"
        "$HOME/cuda"
    )
    
    for cuda_path in "${cuda_paths[@]}"; do
        if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ]; then
            echo "$cuda_path"
            return 0
        fi
    done
    
    return 1
}

# 设置CUDA环境
setup_cuda_env() {
    log_info "设置CUDA环境..."
    
    # 如果nvcc已经在PATH中，直接返回
    if command -v nvcc &> /dev/null; then
        local nvcc_path=$(which nvcc)
        local cuda_home=$(dirname $(dirname "$nvcc_path"))
        log_success "CUDA已在PATH中，CUDA_HOME: $cuda_home"
        export CUDA_HOME="$cuda_home"
        return 0
    fi
    
    # 尝试寻找CUDA安装路径
    local cuda_path=$(find_cuda_path)
    if [ $? -eq 0 ]; then
        log_success "找到CUDA安装路径: $cuda_path"
        export CUDA_HOME="$cuda_path"
        export PATH="$cuda_path/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_path/lib64:$LD_LIBRARY_PATH"
        log_success "CUDA环境已设置"
        return 0
    else
        log_warning "未找到CUDA安装路径"
        return 1
    fi
}

# 检查CUDA环境
check_cuda() {
    log_info "检查CUDA环境..."
    
    # 首先尝试设置CUDA环境
    if ! setup_cuda_env; then
        log_warning "CUDA编译器(nvcc)未找到，将跳过CUDA相关检查"
        log_warning "如果需要编译NCCL，请确保CUDA已正确安装并添加到PATH"
        return 0
    fi
    
    # 验证nvcc是否可用
    if ! command -v nvcc &> /dev/null; then
        log_warning "设置CUDA环境后仍无法找到nvcc"
        return 0
    fi
    
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    log_success "CUDA版本: $CUDA_VERSION"
    log_success "CUDA_HOME: ${CUDA_HOME:-未设置}"
}

# 清理构建目录（已禁用）
clean_build() {
    log_warning "清理功能已禁用，保留现有构建目录"
}

# 编译NCCL库
build_nccl() {
    log_info "开始编译NCCL库..."
    cd "$PROJECT_ROOT"
    
    # 使用多核编译
    make -j$NPROC
    
    if [ $? -eq 0 ]; then
        log_success "NCCL库编译完成"
    else
        log_error "NCCL库编译失败"
        exit 1
    fi
}

# 创建符号链接
create_symlinks() {
    log_info "创建库文件符号链接..."
    cd "$LIB_DIR"
    
    # 创建libnccl.so.2符号链接
    if [ -f "libnccl.so.2.23.4" ]; then
        ln -sf libnccl.so.2.23.4 libnccl.so.2
        ln -sf libnccl.so.2.23.4 libnccl.so
        log_success "符号链接创建完成"
    else
        log_error "找不到libnccl.so.2.23.4文件"
        exit 1
    fi
}

# 编译测试程序
build_tests() {
    log_info "编译测试程序..."
    cd "$PROJECT_ROOT"
    
    # 检查nvcc是否可用
    if ! command -v nvcc &> /dev/null; then
        log_warning "nvcc未找到，跳过测试程序编译"
        log_warning "如果需要编译测试程序，请确保CUDA已正确安装并添加到PATH"
        return 0
    fi
    
    # 编译addition测试
    if [ -f "test/addition.cc" ]; then
        mkdir -p "$TEST_DIR"
        nvcc -I./src -I./build/include -L./build/lib -lnccl -lcuda -lcudart \
             test/addition.cc -o "$TEST_DIR/addition"
        
        if [ $? -eq 0 ]; then
            log_success "addition测试程序编译完成"
        else
            log_error "addition测试程序编译失败"
            return 1
        fi
    else
        log_warning "未找到addition.cc测试文件"
    fi
    
    # 编译removal测试（如果存在）
    if [ -f "test/removal.cc" ]; then
        nvcc -I./src -I./build/include -L./build/lib -lnccl -lcuda -lcudart \
             test/removal.cc -o "$TEST_DIR/removal"
        
        if [ $? -eq 0 ]; then
            log_success "removal测试程序编译完成"
        else
            log_warning "removal测试程序编译失败"
        fi
    fi
}

# 运行测试
run_tests() {
    log_info "开始运行测试..."
    cd "$BUILD_DIR"
    
    # 设置库路径
    export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"
    
    # 测试不同规模的动态rank添加
    local test_cases=(3)
    
    for total_ranks in "${test_cases[@]}"; do
        log_info "测试 $((total_ranks-1))+1=$total_ranks ranks 场景..."
        
        # 运行测试
        if timeout 30s ./test/addition $total_ranks; then
            log_success "$total_ranks ranks 测试通过"
        else
            local exit_code=$?
            if [ $exit_code -eq 124 ]; then
                log_warning "$total_ranks ranks 测试超时（可能卡住）"
            else
                log_error "$total_ranks ranks 测试失败 (退出码: $exit_code)"
            fi
        fi
        echo "----------------------------------------"
    done
}

# 运行调试测试
run_debug_test() {
    local ranks=${1:-4}
    log_info "运行调试测试 (${ranks} ranks)..."
    cd "$BUILD_DIR"
    
    export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"
    export NCCL_DEBUG=INFO
    export NCCL_RUNTIME_CONNECT=1  # lazy 连接
    export NCCL_DEBUG_SUBSYS=INIT,GRAPH
    export NCCL_PARALLEL_JOIN=1
    
    ./test/addition $ranks
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -c, --clean         清理构建目录 (已禁用)"
    echo "  -b, --build         仅编译NCCL库和测试程序"
    echo "  -t, --test          仅运行测试"
    echo "  -d, --debug [N]     运行调试测试 (N为ranks数量，默认3)"
    echo "  -a, --all           完整流程：编译、测试 (默认)"
    echo ""
    echo "示例:"
    echo "  $0                  # 运行完整流程"
    echo "  $0 --clean          # 仅清理"
    echo "  $0 --build          # 仅编译"
    echo "  $0 --test           # 仅测试"
    echo "  $0 --debug 3        # 调试3个ranks的测试"
}

# 主函数
main() {
    log_info "Flexible NCCL 测试脚本启动"
    log_info "项目路径: $PROJECT_ROOT"
    
    # 解析命令行参数
    case "${1:-all}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            clean_build
            ;;
        -b|--build)
            check_cuda
            build_nccl
            create_symlinks
            build_tests
            ;;
        -t|--test)
            run_tests
            ;;
        -d|--debug)
            run_debug_test ${2:-4}
            ;;
        -a|--all|*)
            check_cuda
            build_nccl
            create_symlinks
            build_tests
            run_tests
            ;;
    esac
    
    log_success "脚本执行完成"
}

# 执行主函数
main "$@"