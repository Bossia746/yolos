#!/usr/bin/env python3
"""
YOLOS 自动化部署脚本
支持多环境部署、回滚、监控和通知
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml
import requests
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """部署配置"""
    environment: str
    version: str
    docker_compose_file: str
    health_check_timeout: int
    rollback_on_failure: bool
    notification_enabled: bool
    backup_before_deploy: bool
    run_tests: bool
    parallel_deployment: bool
    max_deployment_time: int

@dataclass
class DeploymentResult:
    """部署结果"""
    success: bool
    environment: str
    version: str
    start_time: datetime
    end_time: datetime
    duration: float
    steps_completed: List[str]
    errors: List[str]
    warnings: List[str]
    rollback_performed: bool = False
    snapshot_id: Optional[str] = None

class DeploymentManager:
    """部署管理器"""
    
    def __init__(self, config_path: str = "deployment/config/deploy_config.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.project_root = Path.cwd()
        
        # 导入回滚管理器
        sys.path.append(str(self.project_root / "deployment" / "scripts"))
        try:
            from rollback_recovery_manager import RollbackRecoveryManager
            self.rollback_manager = RollbackRecoveryManager()
        except ImportError:
            logger.warning("回滚管理器不可用")
            self.rollback_manager = None
    
    def _load_config(self) -> Dict[str, Any]:
        """加载部署配置"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 创建默认配置
            default_config = {
                'environments': {
                    'development': {
                        'docker_compose_file': 'deployment/docker/docker-compose.optimized.yml',
                        'health_check_timeout': 120,
                        'rollback_on_failure': True,
                        'notification_enabled': False,
                        'backup_before_deploy': False,
                        'run_tests': True,
                        'parallel_deployment': False,
                        'max_deployment_time': 600
                    },
                    'staging': {
                        'docker_compose_file': 'deployment/docker/docker-compose.optimized.yml',
                        'health_check_timeout': 180,
                        'rollback_on_failure': True,
                        'notification_enabled': True,
                        'backup_before_deploy': True,
                        'run_tests': True,
                        'parallel_deployment': False,
                        'max_deployment_time': 900
                    },
                    'production': {
                        'docker_compose_file': 'deployment/docker/docker-compose.optimized.yml',
                        'health_check_timeout': 300,
                        'rollback_on_failure': True,
                        'notification_enabled': True,
                        'backup_before_deploy': True,
                        'run_tests': True,
                        'parallel_deployment': True,
                        'max_deployment_time': 1200
                    }
                },
                'services': {
                    'yolos-app': {
                        'health_endpoint': 'http://localhost:8000/health',
                        'startup_time': 60,
                        'critical': True
                    },
                    'yolos-web': {
                        'health_endpoint': 'http://localhost:8080/',
                        'startup_time': 30,
                        'critical': False
                    },
                    'postgres': {
                        'health_command': 'pg_isready -U yolos',
                        'startup_time': 30,
                        'critical': True
                    },
                    'redis': {
                        'health_command': 'redis-cli ping',
                        'startup_time': 10,
                        'critical': True
                    }
                },
                'deployment_strategies': {
                    'blue_green': {
                        'enabled': False,
                        'switch_timeout': 60
                    },
                    'rolling_update': {
                        'enabled': True,
                        'batch_size': 1,
                        'batch_interval': 30
                    },
                    'recreate': {
                        'enabled': True,
                        'downtime_acceptable': True
                    }
                },
                'notifications': {
                    'webhook_url': None,
                    'slack_webhook': None,
                    'email_recipients': []
                },
                'testing': {
                    'smoke_tests': True,
                    'integration_tests': False,
                    'performance_tests': False
                }
            }
            
            # 保存默认配置
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            return default_config
    
    def deploy(self, environment: str, version: str, force: bool = False) -> DeploymentResult:
        """执行部署"""
        start_time = datetime.now()
        steps_completed = []
        errors = []
        warnings = []
        rollback_performed = False
        snapshot_id = None
        
        logger.info(f"开始部署 YOLOS {version} 到 {environment} 环境")
        
        try:
            # 获取环境配置
            env_config = self.config['environments'].get(environment)
            if not env_config:
                raise ValueError(f"未找到环境配置: {environment}")
            
            deploy_config = DeploymentConfig(
                environment=environment,
                version=version,
                **env_config
            )
            
            # 步骤1: 预部署检查
            logger.info("执行预部署检查...")
            self._pre_deployment_checks(deploy_config, force)
            steps_completed.append("预部署检查")
            
            # 步骤2: 创建部署快照
            if deploy_config.backup_before_deploy and self.rollback_manager:
                logger.info("创建部署快照...")
                snapshot = self.rollback_manager.create_snapshot(version, environment)
                snapshot_id = snapshot.id
                steps_completed.append("创建部署快照")
            
            # 步骤3: 构建和推送镜像
            logger.info("构建和推送镜像...")
            self._build_and_push_images(version)
            steps_completed.append("构建和推送镜像")
            
            # 步骤4: 更新配置文件
            logger.info("更新配置文件...")
            self._update_configuration(deploy_config)
            steps_completed.append("更新配置文件")
            
            # 步骤5: 执行部署策略
            logger.info("执行部署策略...")
            self._execute_deployment_strategy(deploy_config)
            steps_completed.append("执行部署策略")
            
            # 步骤6: 等待服务启动
            logger.info("等待服务启动...")
            self._wait_for_services(deploy_config)
            steps_completed.append("等待服务启动")
            
            # 步骤7: 健康检查
            logger.info("执行健康检查...")
            health_status = self._perform_health_checks(deploy_config)
            if not health_status:
                raise RuntimeError("健康检查失败")
            steps_completed.append("健康检查")
            
            # 步骤8: 运行测试
            if deploy_config.run_tests:
                logger.info("运行部署后测试...")
                test_results = self._run_post_deployment_tests(deploy_config)
                if not test_results:
                    warnings.append("部分测试失败")
                steps_completed.append("运行测试")
            
            # 步骤9: 发送成功通知
            if deploy_config.notification_enabled:
                logger.info("发送部署成功通知...")
                self._send_notification("success", deploy_config, version)
                steps_completed.append("发送通知")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"部署成功完成，耗时: {duration:.1f}秒")
            
            return DeploymentResult(
                success=True,
                environment=environment,
                version=version,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                steps_completed=steps_completed,
                errors=errors,
                warnings=warnings,
                rollback_performed=rollback_performed,
                snapshot_id=snapshot_id
            )
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            errors.append(str(e))
            
            logger.error(f"部署失败: {e}")
            
            # 执行回滚
            if deploy_config.rollback_on_failure and snapshot_id and self.rollback_manager:
                logger.info("执行自动回滚...")
                try:
                    rollback_report = self.rollback_manager.execute_rollback(snapshot_id)
                    rollback_performed = rollback_report.success
                    if rollback_performed:
                        logger.info("自动回滚成功")
                    else:
                        logger.error("自动回滚失败")
                        errors.extend(rollback_report.errors)
                except Exception as rollback_error:
                    logger.error(f"回滚执行异常: {rollback_error}")
                    errors.append(f"回滚失败: {rollback_error}")
            
            # 发送失败通知
            if deploy_config.notification_enabled:
                self._send_notification("failure", deploy_config, version, str(e))
            
            return DeploymentResult(
                success=False,
                environment=environment,
                version=version,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                steps_completed=steps_completed,
                errors=errors,
                warnings=warnings,
                rollback_performed=rollback_performed,
                snapshot_id=snapshot_id
            )
    
    def _pre_deployment_checks(self, config: DeploymentConfig, force: bool):
        """预部署检查"""
        # 检查Docker和Docker Compose
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
            subprocess.run(['docker-compose', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Docker或Docker Compose未安装或不可用")
        
        # 检查配置文件存在性
        compose_file = Path(config.docker_compose_file)
        if not compose_file.exists():
            raise RuntimeError(f"Docker Compose文件不存在: {config.docker_compose_file}")
        
        # 检查磁盘空间
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
        if free_space < 5.0:  # 至少5GB空闲空间
            if not force:
                raise RuntimeError(f"磁盘空间不足: {free_space:.1f}GB")
            else:
                logger.warning(f"磁盘空间不足但强制部署: {free_space:.1f}GB")
        
        # 检查端口占用
        self._check_port_availability()
        
        # 检查环境变量
        required_env_vars = ['YOLOS_ENV', 'POSTGRES_PASSWORD', 'REDIS_PASSWORD']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logger.warning(f"缺少环境变量: {missing_vars}")
    
    def _check_port_availability(self):
        """检查端口可用性"""
        import socket
        
        ports_to_check = [8000, 8080, 5432, 6379, 80, 9090, 3000]
        
        for port in ports_to_check:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    logger.warning(f"端口 {port} 已被占用")
            except Exception:
                pass
    
    def _build_and_push_images(self, version: str):
        """构建和推送镜像"""
        # 构建主应用镜像
        logger.info("构建YOLOS主应用镜像...")
        subprocess.run([
            'docker', 'build',
            '-t', f'yolos-app:{version}',
            '-t', 'yolos-app:latest',
            '--target', 'production',
            '-f', 'deployment/docker/optimized_dockerfile',
            '.'
        ], check=True)
        
        # 构建Web界面镜像
        logger.info("构建YOLOS Web界面镜像...")
        subprocess.run([
            'docker', 'build',
            '-t', f'yolos-web:{version}',
            '-t', 'yolos-web:latest',
            '--target', 'production',
            '-f', 'deployment/docker/optimized_dockerfile',
            '.'
        ], check=True)
        
        # 如果配置了私有仓库，推送镜像
        registry = self.config.get('docker_registry')
        if registry:
            logger.info(f"推送镜像到私有仓库: {registry}")
            
            for image in [f'yolos-app:{version}', f'yolos-web:{version}']:
                # 标记镜像
                tagged_image = f"{registry}/{image}"
                subprocess.run(['docker', 'tag', image, tagged_image], check=True)
                
                # 推送镜像
                subprocess.run(['docker', 'push', tagged_image], check=True)
    
    def _update_configuration(self, config: DeploymentConfig):
        """更新配置文件"""
        # 设置环境变量
        os.environ['YOLOS_VERSION'] = config.version
        os.environ['YOLOS_ENV'] = config.environment
        
        # 更新Docker Compose环境变量文件
        env_file = Path('.env')
        env_content = f"""
# YOLOS部署环境变量
YOLOS_VERSION={config.version}
YOLOS_ENV={config.environment}
COMPOSE_PROJECT_NAME=yolos-{config.environment}
COMPOSE_FILE={config.docker_compose_file}

# 数据库配置
POSTGRES_DB=yolos
POSTGRES_USER=yolos
POSTGRES_PASSWORD={os.getenv('POSTGRES_PASSWORD', 'yolos_password')}

# Redis配置
REDIS_PASSWORD={os.getenv('REDIS_PASSWORD', 'redis_password')}

# 应用配置
DEBUG={'true' if config.environment == 'development' else 'false'}
LOG_LEVEL={'DEBUG' if config.environment == 'development' else 'INFO'}

# 监控配置
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
"""
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content.strip())
    
    def _execute_deployment_strategy(self, config: DeploymentConfig):
        """执行部署策略"""
        strategies = self.config.get('deployment_strategies', {})
        
        if strategies.get('blue_green', {}).get('enabled'):
            self._blue_green_deployment(config)
        elif strategies.get('rolling_update', {}).get('enabled'):
            self._rolling_update_deployment(config)
        else:
            self._recreate_deployment(config)
    
    def _recreate_deployment(self, config: DeploymentConfig):
        """重新创建部署"""
        logger.info("执行重新创建部署策略...")
        
        # 停止现有服务
        try:
            subprocess.run([
                'docker-compose', '-f', config.docker_compose_file, 'down'
            ], check=True, timeout=120)
        except subprocess.TimeoutExpired:
            logger.warning("停止服务超时，强制终止")
            subprocess.run([
                'docker-compose', '-f', config.docker_compose_file, 'kill'
            ])
        
        # 启动新服务
        subprocess.run([
            'docker-compose', '-f', config.docker_compose_file, 'up', '-d'
        ], check=True)
    
    def _rolling_update_deployment(self, config: DeploymentConfig):
        """滚动更新部署"""
        logger.info("执行滚动更新部署策略...")
        
        services = self.config.get('services', {}).keys()
        
        for service in services:
            logger.info(f"更新服务: {service}")
            
            # 更新单个服务
            subprocess.run([
                'docker-compose', '-f', config.docker_compose_file,
                'up', '-d', '--no-deps', service
            ], check=True)
            
            # 等待服务启动
            time.sleep(30)
            
            # 检查服务健康状态
            if not self._check_service_health(service):
                raise RuntimeError(f"服务 {service} 健康检查失败")
    
    def _blue_green_deployment(self, config: DeploymentConfig):
        """蓝绿部署"""
        logger.info("执行蓝绿部署策略...")
        
        # 蓝绿部署需要更复杂的实现
        # 这里提供基本框架
        
        # 1. 启动绿色环境
        green_compose_file = config.docker_compose_file.replace('.yml', '-green.yml')
        
        # 2. 等待绿色环境就绪
        # 3. 切换流量到绿色环境
        # 4. 停止蓝色环境
        
        # 暂时使用重新创建策略
        self._recreate_deployment(config)
    
    def _wait_for_services(self, config: DeploymentConfig):
        """等待服务启动"""
        services = self.config.get('services', {})
        max_wait_time = 300  # 最大等待5分钟
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            all_ready = True
            
            for service_name, service_config in services.items():
                if not self._check_service_health(service_name):
                    all_ready = False
                    break
            
            if all_ready:
                logger.info("所有服务已启动")
                return
            
            logger.info("等待服务启动...")
            time.sleep(10)
        
        raise RuntimeError("服务启动超时")
    
    def _check_service_health(self, service_name: str) -> bool:
        """检查单个服务健康状态"""
        service_config = self.config.get('services', {}).get(service_name, {})
        
        try:
            if 'health_endpoint' in service_config:
                response = requests.get(
                    service_config['health_endpoint'],
                    timeout=5
                )
                return response.status_code == 200
            elif 'health_command' in service_config:
                result = subprocess.run(
                    service_config['health_command'].split(),
                    capture_output=True,
                    timeout=5
                )
                return result.returncode == 0
        except Exception:
            pass
        
        return False
    
    def _perform_health_checks(self, config: DeploymentConfig) -> bool:
        """执行健康检查"""
        services = self.config.get('services', {})
        failed_services = []
        
        for service_name, service_config in services.items():
            if not self._check_service_health(service_name):
                if service_config.get('critical', False):
                    failed_services.append(service_name)
                else:
                    logger.warning(f"非关键服务 {service_name} 健康检查失败")
        
        if failed_services:
            logger.error(f"关键服务健康检查失败: {failed_services}")
            return False
        
        return True
    
    def _run_post_deployment_tests(self, config: DeploymentConfig) -> bool:
        """运行部署后测试"""
        test_config = self.config.get('testing', {})
        test_results = []
        
        # 运行烟雾测试
        if test_config.get('smoke_tests', True):
            logger.info("运行烟雾测试...")
            try:
                result = subprocess.run([
                    'python', 'tests/smoke_tests.py',
                    '--environment', config.environment
                ], capture_output=True, text=True, timeout=120)
                
                test_results.append(result.returncode == 0)
                
                if result.returncode != 0:
                    logger.error(f"烟雾测试失败: {result.stderr}")
                
            except Exception as e:
                logger.error(f"烟雾测试执行异常: {e}")
                test_results.append(False)
        
        # 运行集成测试
        if test_config.get('integration_tests', False):
            logger.info("运行集成测试...")
            try:
                result = subprocess.run([
                    'python', 'src/api/integration_test_fixer.py'
                ], capture_output=True, text=True, timeout=300)
                
                test_results.append(result.returncode == 0)
                
            except Exception as e:
                logger.error(f"集成测试执行异常: {e}")
                test_results.append(False)
        
        # 运行性能测试
        if test_config.get('performance_tests', False):
            logger.info("运行性能测试...")
            # 性能测试实现
            test_results.append(True)  # 占位符
        
        return all(test_results) if test_results else True
    
    def _send_notification(self, status: str, config: DeploymentConfig, version: str, error_msg: str = None):
        """发送通知"""
        notification_config = self.config.get('notifications', {})
        
        message = {
            'environment': config.environment,
            'version': version,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'error': error_msg if error_msg else None
        }
        
        # Webhook通知
        webhook_url = notification_config.get('webhook_url')
        if webhook_url:
            try:
                requests.post(webhook_url, json=message, timeout=10)
            except Exception as e:
                logger.warning(f"Webhook通知发送失败: {e}")
        
        # Slack通知
        slack_webhook = notification_config.get('slack_webhook')
        if slack_webhook:
            try:
                slack_message = {
                    'text': f"YOLOS部署{status}: {config.environment} - {version}",
                    'attachments': [{
                        'color': 'good' if status == 'success' else 'danger',
                        'fields': [
                            {'title': '环境', 'value': config.environment, 'short': True},
                            {'title': '版本', 'value': version, 'short': True},
                            {'title': '状态', 'value': status, 'short': True},
                            {'title': '时间', 'value': message['timestamp'], 'short': True}
                        ]
                    }]
                }
                
                if error_msg:
                    slack_message['attachments'][0]['fields'].append({
                        'title': '错误信息',
                        'value': error_msg,
                        'short': False
                    })
                
                requests.post(slack_webhook, json=slack_message, timeout=10)
                
            except Exception as e:
                logger.warning(f"Slack通知发送失败: {e}")
    
    def list_deployments(self, environment: str = None) -> List[Dict[str, Any]]:
        """列出部署历史"""
        deployments = []
        
        # 从日志文件或数据库中获取部署历史
        logs_dir = Path('deployment/logs')
        if logs_dir.exists():
            for log_file in logs_dir.glob('deployment_*.json'):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        deployment_data = json.load(f)
                        
                    if environment is None or deployment_data.get('environment') == environment:
                        deployments.append(deployment_data)
                        
                except Exception as e:
                    logger.warning(f"读取部署日志失败 {log_file}: {e}")
        
        return sorted(deployments, key=lambda x: x.get('start_time', ''), reverse=True)
    
    def save_deployment_log(self, result: DeploymentResult):
        """保存部署日志"""
        logs_dir = Path('deployment/logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = result.start_time.strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f"deployment_{result.environment}_{timestamp}.json"
        
        log_data = {
            'success': result.success,
            'environment': result.environment,
            'version': result.version,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'duration': result.duration,
            'steps_completed': result.steps_completed,
            'errors': result.errors,
            'warnings': result.warnings,
            'rollback_performed': result.rollback_performed,
            'snapshot_id': result.snapshot_id
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"部署日志已保存: {log_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOS自动化部署脚本')
    parser.add_argument('action', choices=['deploy', 'list', 'status'],
                       help='执行的操作')
    parser.add_argument('--environment', '-e', required=True,
                       choices=['development', 'staging', 'production'],
                       help='部署环境')
    parser.add_argument('--version', '-v', help='部署版本')
    parser.add_argument('--force', '-f', action='store_true',
                       help='强制部署（忽略检查）')
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建部署管理器
    config_path = args.config or "deployment/config/deploy_config.yml"
    manager = DeploymentManager(config_path)
    
    if args.action == 'deploy':
        if not args.version:
            print("部署需要指定版本号")
            return 1
        
        # 执行部署
        result = manager.deploy(args.environment, args.version, args.force)
        
        # 保存部署日志
        manager.save_deployment_log(result)
        
        # 输出结果
        print(f"\n部署{'成功' if result.success else '失败'}:")
        print(f"  环境: {result.environment}")
        print(f"  版本: {result.version}")
        print(f"  耗时: {result.duration:.1f}秒")
        print(f"  完成步骤: {len(result.steps_completed)}")
        
        if result.errors:
            print("\n错误:")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.warnings:
            print("\n警告:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        if result.rollback_performed:
            print("\n已执行自动回滚")
        
        return 0 if result.success else 1
        
    elif args.action == 'list':
        # 列出部署历史
        deployments = manager.list_deployments(args.environment)
        
        print(f"\n{args.environment} 环境部署历史:")
        for deployment in deployments[:10]:  # 显示最近10次部署
            status = "✓" if deployment.get('success') else "✗"
            print(f"  {status} {deployment.get('version', 'unknown')} - {deployment.get('start_time', 'unknown')}")
        
    elif args.action == 'status':
        # 显示当前状态
        print(f"\n{args.environment} 环境状态:")
        
        # 检查服务状态
        services = manager.config.get('services', {})
        for service_name in services.keys():
            health = manager._check_service_health(service_name)
            status = "运行中" if health else "停止"
            print(f"  {service_name}: {status}")
    
    return 0

if __name__ == "__main__":
    exit(main())