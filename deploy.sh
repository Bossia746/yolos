#!/bin/bash
# YOLOS 部署启动脚本 (Linux/macOS)
# 用法: ./deploy.sh <environment> <version> [options]

set -e

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_SCRIPT="$SCRIPT_DIR/deployment/scripts/deploy.py"

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

# 检查依赖
check_dependencies() {
    # 检查Python
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        log_error "Python未安装或不在PATH中"
        exit 1
    fi
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_warning "Docker未安装，部署可能失败"
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_warning "Docker Compose未安装，部署可能失败"
    fi
    
    # 检查部署脚本
    if [[ ! -f "$DEPLOY_SCRIPT" ]]; then
        log_error "部署脚本不存在: $DEPLOY_SCRIPT"
        exit 1
    fi
}

# 显示帮助信息
show_help() {
    cat << EOF
${BLUE}YOLOS 自动化部署工具${NC}

${YELLOW}用法:${NC}
  ./deploy.sh <environment> <version> [options]

${YELLOW}环境:${NC}
  development  - 开发环境
  staging      - 预发布环境
  production   - 生产环境

${YELLOW}选项:${NC}
  --force      - 强制部署（忽略检查）
  --verbose    - 详细输出
  --config     - 指定配置文件路径
  --help       - 显示帮助信息

${YELLOW}示例:${NC}
  ./deploy.sh development v1.0.0
  ./deploy.sh production v1.0.0 --force
  ./deploy.sh staging v1.0.0 --verbose

${YELLOW}其他操作:${NC}
  ./deploy.sh list development     - 查看部署历史
  ./deploy.sh status production   - 查看环境状态
  ./deploy.sh rollback production - 回滚到上一个版本

${YELLOW}快捷命令:${NC}
  ./deploy.sh dev v1.0.0          - 部署到开发环境
  ./deploy.sh prod v1.0.0         - 部署到生产环境
  ./deploy.sh stage v1.0.0        - 部署到预发布环境
EOF
}

# 环境别名映射
map_environment() {
    case "$1" in
        "dev"|"development")
            echo "development"
            ;;
        "stage"|"staging")
            echo "staging"
            ;;
        "prod"|"production")
            echo "production"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

# 获取Python命令
get_python_cmd() {
    if command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        log_error "找不到Python解释器"
        exit 1
    fi
}

# 主函数
main() {
    # 检查依赖
    check_dependencies
    
    # 设置环境变量
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    
    # 获取Python命令
    PYTHON_CMD=$(get_python_cmd)
    
    # 如果没有参数或请求帮助，显示帮助信息
    if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    # 特殊操作处理
    case "$1" in
        "list")
            if [[ -z "$2" ]]; then
                log_error "list操作需要指定环境"
                exit 1
            fi
            ENV=$(map_environment "$2")
            log_info "查看 $ENV 环境的部署历史..."
            $PYTHON_CMD "$DEPLOY_SCRIPT" list --environment "$ENV" "${@:3}"
            exit $?
            ;;
        "status")
            if [[ -z "$2" ]]; then
                log_error "status操作需要指定环境"
                exit 1
            fi
            ENV=$(map_environment "$2")
            log_info "查看 $ENV 环境状态..."
            $PYTHON_CMD "$DEPLOY_SCRIPT" status --environment "$ENV" "${@:3}"
            exit $?
            ;;
        "rollback")
            if [[ -z "$2" ]]; then
                log_error "rollback操作需要指定环境"
                exit 1
            fi
            ENV=$(map_environment "$2")
            log_info "回滚 $ENV 环境..."
            # 这里可以调用回滚脚本
            log_warning "回滚功能正在开发中"
            exit 0
            ;;
    esac
    
    # 验证参数
    if [[ $# -lt 2 ]]; then
        log_error "缺少必要参数"
        show_help
        exit 1
    fi
    
    # 映射环境名称
    ENVIRONMENT=$(map_environment "$1")
    VERSION="$2"
    
    # 验证环境
    case "$ENVIRONMENT" in
        "development"|"staging"|"production")
            ;;
        *)
            log_error "无效的环境: $ENVIRONMENT"
            log_info "支持的环境: development, staging, production"
            exit 1
            ;;
    esac
    
    # 验证版本格式
    if [[ ! "$VERSION" =~ ^v?[0-9]+\.[0-9]+\.[0-9]+.*$ ]]; then
        log_warning "版本格式可能不正确: $VERSION"
        log_info "建议使用语义化版本，如: v1.0.0"
    fi
    
    # 确认生产环境部署
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo
        log_warning "您即将部署到生产环境！"
        log_info "环境: $ENVIRONMENT"
        log_info "版本: $VERSION"
        echo
        read -p "确认继续部署？(y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "部署已取消"
            exit 0
        fi
    fi
    
    # 执行部署
    echo
    log_info "开始部署 YOLOS $VERSION 到 $ENVIRONMENT 环境..."
    echo
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 执行部署命令
    if $PYTHON_CMD "$DEPLOY_SCRIPT" deploy --environment "$ENVIRONMENT" --version "$VERSION" "${@:3}"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo
        log_success "部署成功完成！耗时: ${DURATION}秒"
        
        # 显示部署后信息
        echo
        log_info "部署后检查:"
        echo "  - 查看服务状态: ./deploy.sh status $ENVIRONMENT"
        echo "  - 查看应用日志: docker-compose logs -f yolos-app"
        echo "  - 访问应用: http://localhost:8000"
        echo "  - 访问Web界面: http://localhost:8080"
        
        exit 0
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo
        log_error "部署失败！耗时: ${DURATION}秒"
        
        # 显示故障排除信息
        echo
        log_info "故障排除:"
        echo "  - 查看部署日志: ls -la deployment/logs/"
        echo "  - 查看容器状态: docker-compose ps"
        echo "  - 查看容器日志: docker-compose logs"
        echo "  - 检查系统资源: docker system df"
        
        exit 1
    fi
}

# 捕获中断信号
trap 'log_warning "部署被中断"; exit 130' INT TERM

# 执行主函数
main "$@"