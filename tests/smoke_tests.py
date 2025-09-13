#!/usr/bin/env python3
"""
YOLOS 烟雾测试套件
用于快速验证系统基本功能和服务可用性
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import requests
import yaml
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """测试结果"""
    name: str
    success: bool
    duration: float
    message: str
    details: Optional[Dict[str, Any]] = None

@dataclass
class SmokeTestReport:
    """烟雾测试报告"""
    start_time: datetime
    end_time: datetime
    total_duration: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    results: List[TestResult]
    environment: str
    version: Optional[str] = None

class SmokeTestRunner:
    """烟雾测试运行器"""
    
    def __init__(self, config_path: str = "deployment/config/rollback_config.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            return {
                'services': {
                    'yolos-app': {
                        'health_endpoint': 'http://localhost:8000/health',
                        'critical': True
                    },
                    'yolos-web': {
                        'health_endpoint': 'http://localhost:8080/',
                        'critical': False
                    }
                },
                'testing': {
                    'smoke_tests': {
                        'timeout': 120,
                        'tests': [
                            {
                                'name': 'API健康检查',
                                'type': 'http',
                                'url': 'http://localhost:8000/health',
                                'expected_status': 200
                            }
                        ]
                    }
                }
            }
    
    def run_all_tests(self, environment: str = "production") -> SmokeTestReport:
        """运行所有烟雾测试"""
        logger.info("开始运行烟雾测试套件")
        self.start_time = datetime.now()
        self.results = []
        
        try:
            # 基础连接测试
            self._run_connectivity_tests()
            
            # 服务健康检查
            self._run_health_checks()
            
            # API功能测试
            self._run_api_tests()
            
            # 数据库连接测试
            self._run_database_tests()
            
            # 缓存服务测试
            self._run_cache_tests()
            
            # 文件系统测试
            self._run_filesystem_tests()
            
            # 配置文件测试
            self._run_configuration_tests()
            
            # 性能基准测试
            self._run_performance_tests()
            
        except Exception as e:
            logger.error(f"测试执行异常: {e}")
            self.results.append(TestResult(
                name="测试执行异常",
                success=False,
                duration=0.0,
                message=str(e)
            ))
        
        self.end_time = datetime.now()
        return self._generate_report(environment)
    
    def _run_connectivity_tests(self):
        """运行连接性测试"""
        logger.info("运行连接性测试...")
        
        # 测试网络连接
        self._test_network_connectivity()
        
        # 测试端口可用性
        self._test_port_availability()
        
        # 测试DNS解析
        self._test_dns_resolution()
    
    def _test_network_connectivity(self):
        """测试网络连接"""
        start_time = time.time()
        
        try:
            # 测试本地回环
            response = requests.get('http://localhost:8000/health', timeout=5)
            success = True
            message = f"网络连接正常，状态码: {response.status_code}"
        except requests.exceptions.ConnectionError:
            success = False
            message = "无法连接到本地服务"
        except requests.exceptions.Timeout:
            success = False
            message = "连接超时"
        except Exception as e:
            success = False
            message = f"网络连接异常: {e}"
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name="网络连接测试",
            success=success,
            duration=duration,
            message=message
        ))
    
    def _test_port_availability(self):
        """测试端口可用性"""
        import socket
        
        ports_to_test = [
            (8000, "YOLOS API"),
            (8080, "YOLOS Web"),
            (5432, "PostgreSQL"),
            (6379, "Redis"),
            (80, "Nginx"),
            (9090, "Prometheus"),
            (3000, "Grafana")
        ]
        
        for port, service_name in ports_to_test:
            start_time = time.time()
            
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                success = result == 0
                message = f"{service_name} 端口 {port} {'可用' if success else '不可用'}"
                
            except Exception as e:
                success = False
                message = f"{service_name} 端口测试异常: {e}"
            
            duration = time.time() - start_time
            self.results.append(TestResult(
                name=f"端口可用性测试 - {service_name}",
                success=success,
                duration=duration,
                message=message
            ))
    
    def _test_dns_resolution(self):
        """测试DNS解析"""
        import socket
        
        start_time = time.time()
        
        try:
            # 测试本地主机解析
            socket.gethostbyname('localhost')
            success = True
            message = "DNS解析正常"
        except Exception as e:
            success = False
            message = f"DNS解析失败: {e}"
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name="DNS解析测试",
            success=success,
            duration=duration,
            message=message
        ))
    
    def _run_health_checks(self):
        """运行健康检查"""
        logger.info("运行服务健康检查...")
        
        services = self.config.get('services', {})
        
        for service_name, service_config in services.items():
            self._test_service_health(service_name, service_config)
    
    def _test_service_health(self, service_name: str, service_config: Dict[str, Any]):
        """测试单个服务健康状态"""
        start_time = time.time()
        
        try:
            if 'health_endpoint' in service_config:
                # HTTP健康检查
                response = requests.get(
                    service_config['health_endpoint'],
                    timeout=10
                )
                success = response.status_code == 200
                message = f"HTTP健康检查 - 状态码: {response.status_code}"
                
                details = {
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'content_length': len(response.content)
                }
                
                # 尝试解析JSON响应
                try:
                    json_data = response.json()
                    details['response_data'] = json_data
                except:
                    pass
                    
            elif 'health_command' in service_config:
                # 命令健康检查
                result = subprocess.run(
                    service_config['health_command'].split(),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                success = result.returncode == 0
                message = f"命令健康检查 - 退出码: {result.returncode}"
                
                details = {
                    'exit_code': result.returncode,
                    'stdout': result.stdout.strip(),
                    'stderr': result.stderr.strip()
                }
            else:
                success = False
                message = "未配置健康检查方法"
                details = None
                
        except requests.exceptions.ConnectionError:
            success = False
            message = "连接失败 - 服务可能未启动"
            details = None
        except requests.exceptions.Timeout:
            success = False
            message = "健康检查超时"
            details = None
        except subprocess.TimeoutExpired:
            success = False
            message = "命令执行超时"
            details = None
        except Exception as e:
            success = False
            message = f"健康检查异常: {e}"
            details = None
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name=f"服务健康检查 - {service_name}",
            success=success,
            duration=duration,
            message=message,
            details=details
        ))
    
    def _run_api_tests(self):
        """运行API功能测试"""
        logger.info("运行API功能测试...")
        
        # 测试基本API端点
        api_tests = [
            {
                'name': 'API根路径',
                'url': 'http://localhost:8000/',
                'method': 'GET',
                'expected_status': [200, 404]  # 可能返回404或重定向
            },
            {
                'name': 'API健康检查',
                'url': 'http://localhost:8000/health',
                'method': 'GET',
                'expected_status': [200]
            },
            {
                'name': 'API版本信息',
                'url': 'http://localhost:8000/api/v1/version',
                'method': 'GET',
                'expected_status': [200, 404]
            },
            {
                'name': 'API文档',
                'url': 'http://localhost:8000/docs',
                'method': 'GET',
                'expected_status': [200, 404]
            }
        ]
        
        for test_config in api_tests:
            self._test_api_endpoint(test_config)
    
    def _test_api_endpoint(self, test_config: Dict[str, Any]):
        """测试API端点"""
        start_time = time.time()
        
        try:
            method = test_config.get('method', 'GET')
            url = test_config['url']
            expected_status = test_config.get('expected_status', [200])
            
            if method.upper() == 'GET':
                response = requests.get(url, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, timeout=10)
            else:
                response = requests.request(method, url, timeout=10)
            
            success = response.status_code in expected_status
            message = f"API测试 - 状态码: {response.status_code}"
            
            details = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(response.content)
            }
            
        except requests.exceptions.ConnectionError:
            success = False
            message = "API连接失败"
            details = None
        except requests.exceptions.Timeout:
            success = False
            message = "API请求超时"
            details = None
        except Exception as e:
            success = False
            message = f"API测试异常: {e}"
            details = None
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name=f"API测试 - {test_config['name']}",
            success=success,
            duration=duration,
            message=message,
            details=details
        ))
    
    def _run_database_tests(self):
        """运行数据库连接测试"""
        logger.info("运行数据库连接测试...")
        
        # PostgreSQL连接测试
        self._test_postgresql_connection()
        
        # 数据库基本操作测试
        self._test_database_operations()
    
    def _test_postgresql_connection(self):
        """测试PostgreSQL连接"""
        start_time = time.time()
        
        try:
            # 使用pg_isready命令测试
            result = subprocess.run([
                'docker-compose', '-f', 'deployment/docker/docker-compose.optimized.yml',
                'exec', '-T', 'postgres',
                'pg_isready', '-U', 'yolos', '-d', 'yolos'
            ], capture_output=True, text=True, timeout=15)
            
            success = result.returncode == 0
            message = f"PostgreSQL连接测试 - 退出码: {result.returncode}"
            
            details = {
                'exit_code': result.returncode,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip()
            }
            
        except subprocess.TimeoutExpired:
            success = False
            message = "PostgreSQL连接测试超时"
            details = None
        except FileNotFoundError:
            success = False
            message = "Docker Compose未找到"
            details = None
        except Exception as e:
            success = False
            message = f"PostgreSQL连接测试异常: {e}"
            details = None
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name="PostgreSQL连接测试",
            success=success,
            duration=duration,
            message=message,
            details=details
        ))
    
    def _test_database_operations(self):
        """测试数据库基本操作"""
        start_time = time.time()
        
        try:
            # 执行简单的SQL查询
            sql_query = "SELECT version();"
            result = subprocess.run([
                'docker-compose', '-f', 'deployment/docker/docker-compose.optimized.yml',
                'exec', '-T', 'postgres',
                'psql', '-U', 'yolos', '-d', 'yolos', '-c', sql_query
            ], capture_output=True, text=True, timeout=15)
            
            success = result.returncode == 0 and 'PostgreSQL' in result.stdout
            message = f"数据库操作测试 - {'成功' if success else '失败'}"
            
            details = {
                'exit_code': result.returncode,
                'query': sql_query,
                'output_length': len(result.stdout)
            }
            
        except Exception as e:
            success = False
            message = f"数据库操作测试异常: {e}"
            details = None
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name="数据库操作测试",
            success=success,
            duration=duration,
            message=message,
            details=details
        ))
    
    def _run_cache_tests(self):
        """运行缓存服务测试"""
        logger.info("运行缓存服务测试...")
        
        # Redis连接测试
        self._test_redis_connection()
        
        # Redis基本操作测试
        self._test_redis_operations()
    
    def _test_redis_connection(self):
        """测试Redis连接"""
        start_time = time.time()
        
        try:
            result = subprocess.run([
                'docker-compose', '-f', 'deployment/docker/docker-compose.optimized.yml',
                'exec', '-T', 'redis',
                'redis-cli', 'ping'
            ], capture_output=True, text=True, timeout=10)
            
            success = result.returncode == 0 and 'PONG' in result.stdout
            message = f"Redis连接测试 - {'成功' if success else '失败'}"
            
            details = {
                'exit_code': result.returncode,
                'response': result.stdout.strip()
            }
            
        except Exception as e:
            success = False
            message = f"Redis连接测试异常: {e}"
            details = None
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name="Redis连接测试",
            success=success,
            duration=duration,
            message=message,
            details=details
        ))
    
    def _test_redis_operations(self):
        """测试Redis基本操作"""
        start_time = time.time()
        
        try:
            # 设置和获取测试键值
            test_key = f"smoke_test_{int(time.time())}"
            test_value = "test_value"
            
            # 设置值
            set_result = subprocess.run([
                'docker-compose', '-f', 'deployment/docker/docker-compose.optimized.yml',
                'exec', '-T', 'redis',
                'redis-cli', 'set', test_key, test_value
            ], capture_output=True, text=True, timeout=10)
            
            # 获取值
            get_result = subprocess.run([
                'docker-compose', '-f', 'deployment/docker/docker-compose.optimized.yml',
                'exec', '-T', 'redis',
                'redis-cli', 'get', test_key
            ], capture_output=True, text=True, timeout=10)
            
            # 删除测试键
            subprocess.run([
                'docker-compose', '-f', 'deployment/docker/docker-compose.optimized.yml',
                'exec', '-T', 'redis',
                'redis-cli', 'del', test_key
            ], capture_output=True, text=True, timeout=10)
            
            success = (set_result.returncode == 0 and 
                      get_result.returncode == 0 and 
                      test_value in get_result.stdout)
            
            message = f"Redis操作测试 - {'成功' if success else '失败'}"
            
            details = {
                'set_success': set_result.returncode == 0,
                'get_success': get_result.returncode == 0,
                'value_match': test_value in get_result.stdout
            }
            
        except Exception as e:
            success = False
            message = f"Redis操作测试异常: {e}"
            details = None
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name="Redis操作测试",
            success=success,
            duration=duration,
            message=message,
            details=details
        ))
    
    def _run_filesystem_tests(self):
        """运行文件系统测试"""
        logger.info("运行文件系统测试...")
        
        # 测试关键目录存在性
        critical_paths = [
            'models/',
            'config/',
            'deployment/',
            'src/',
            'tests/'
        ]
        
        for path in critical_paths:
            self._test_path_exists(path)
        
        # 测试文件读写权限
        self._test_file_permissions()
    
    def _test_path_exists(self, path: str):
        """测试路径是否存在"""
        start_time = time.time()
        
        try:
            path_obj = Path(path)
            exists = path_obj.exists()
            
            success = exists
            message = f"路径 {path} {'存在' if exists else '不存在'}"
            
            details = {
                'path': str(path_obj.absolute()),
                'exists': exists,
                'is_dir': path_obj.is_dir() if exists else None,
                'is_file': path_obj.is_file() if exists else None
            }
            
        except Exception as e:
            success = False
            message = f"路径检查异常: {e}"
            details = None
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name=f"路径存在性测试 - {path}",
            success=success,
            duration=duration,
            message=message,
            details=details
        ))
    
    def _test_file_permissions(self):
        """测试文件读写权限"""
        start_time = time.time()
        
        try:
            # 创建临时测试文件
            test_file = Path('temp_smoke_test.txt')
            test_content = f"Smoke test - {datetime.now()}"
            
            # 写入测试
            test_file.write_text(test_content, encoding='utf-8')
            
            # 读取测试
            read_content = test_file.read_text(encoding='utf-8')
            
            # 删除测试文件
            test_file.unlink()
            
            success = read_content == test_content
            message = f"文件读写权限测试 - {'成功' if success else '失败'}"
            
            details = {
                'write_success': True,
                'read_success': True,
                'content_match': read_content == test_content
            }
            
        except PermissionError:
            success = False
            message = "文件读写权限不足"
            details = None
        except Exception as e:
            success = False
            message = f"文件权限测试异常: {e}"
            details = None
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name="文件读写权限测试",
            success=success,
            duration=duration,
            message=message,
            details=details
        ))
    
    def _run_configuration_tests(self):
        """运行配置文件测试"""
        logger.info("运行配置文件测试...")
        
        # 测试关键配置文件
        config_files = [
            'deployment/docker/docker-compose.optimized.yml',
            'deployment/config/rollback_config.yml',
            'deployment/monitoring/prometheus/prometheus.yml'
        ]
        
        for config_file in config_files:
            self._test_config_file(config_file)
    
    def _test_config_file(self, config_file: str):
        """测试配置文件"""
        start_time = time.time()
        
        try:
            config_path = Path(config_file)
            
            if not config_path.exists():
                success = False
                message = f"配置文件不存在: {config_file}"
                details = None
            else:
                # 尝试解析配置文件
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if config_file.endswith('.yml') or config_file.endswith('.yaml'):
                    yaml.safe_load(content)
                elif config_file.endswith('.json'):
                    json.loads(content)
                
                success = True
                message = f"配置文件有效: {config_file}"
                details = {
                    'file_size': len(content),
                    'line_count': len(content.splitlines())
                }
                
        except yaml.YAMLError as e:
            success = False
            message = f"YAML配置文件格式错误: {e}"
            details = None
        except json.JSONDecodeError as e:
            success = False
            message = f"JSON配置文件格式错误: {e}"
            details = None
        except Exception as e:
            success = False
            message = f"配置文件测试异常: {e}"
            details = None
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name=f"配置文件测试 - {Path(config_file).name}",
            success=success,
            duration=duration,
            message=message,
            details=details
        ))
    
    def _run_performance_tests(self):
        """运行性能基准测试"""
        logger.info("运行性能基准测试...")
        
        # 简单的响应时间测试
        self._test_response_time()
        
        # 系统资源使用测试
        self._test_system_resources()
    
    def _test_response_time(self):
        """测试响应时间"""
        start_time = time.time()
        
        try:
            # 测试API响应时间
            response_times = []
            
            for i in range(5):  # 测试5次
                try:
                    response = requests.get('http://localhost:8000/health', timeout=10)
                    response_times.append(response.elapsed.total_seconds())
                except:
                    pass
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                
                # 响应时间阈值：平均<2秒，最大<5秒
                success = avg_response_time < 2.0 and max_response_time < 5.0
                message = f"响应时间测试 - 平均: {avg_response_time:.3f}s, 最大: {max_response_time:.3f}s"
                
                details = {
                    'average_response_time': avg_response_time,
                    'max_response_time': max_response_time,
                    'min_response_time': min_response_time,
                    'sample_count': len(response_times)
                }
            else:
                success = False
                message = "无法获取响应时间数据"
                details = None
                
        except Exception as e:
            success = False
            message = f"响应时间测试异常: {e}"
            details = None
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name="API响应时间测试",
            success=success,
            duration=duration,
            message=message,
            details=details
        ))
    
    def _test_system_resources(self):
        """测试系统资源使用"""
        start_time = time.time()
        
        try:
            import psutil
            
            # 获取系统资源使用情况
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # 资源使用阈值
            cpu_threshold = 90.0
            memory_threshold = 90.0
            disk_threshold = 95.0
            
            success = (cpu_percent < cpu_threshold and 
                      memory.percent < memory_threshold and 
                      disk.percent < disk_threshold)
            
            message = f"系统资源测试 - CPU: {cpu_percent:.1f}%, 内存: {memory.percent:.1f}%, 磁盘: {disk.percent:.1f}%"
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
            
        except ImportError:
            success = True  # psutil不可用时跳过测试
            message = "系统资源测试跳过 - psutil未安装"
            details = None
        except Exception as e:
            success = False
            message = f"系统资源测试异常: {e}"
            details = None
        
        duration = time.time() - start_time
        self.results.append(TestResult(
            name="系统资源使用测试",
            success=success,
            duration=duration,
            message=message,
            details=details
        ))
    
    def _generate_report(self, environment: str) -> SmokeTestReport:
        """生成测试报告"""
        total_duration = (self.end_time - self.start_time).total_seconds()
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        return SmokeTestReport(
            start_time=self.start_time,
            end_time=self.end_time,
            total_duration=total_duration,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            results=self.results,
            environment=environment
        )
    
    def save_report(self, report: SmokeTestReport, output_file: str = None):
        """保存测试报告"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"deployment/logs/smoke_test_report_{timestamp}.json"
        
        # 确保输出目录存在
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化的格式
        report_data = {
            'start_time': report.start_time.isoformat(),
            'end_time': report.end_time.isoformat(),
            'total_duration': report.total_duration,
            'total_tests': report.total_tests,
            'passed_tests': report.passed_tests,
            'failed_tests': report.failed_tests,
            'success_rate': report.success_rate,
            'environment': report.environment,
            'version': report.version,
            'results': [
                {
                    'name': result.name,
                    'success': result.success,
                    'duration': result.duration,
                    'message': result.message,
                    'details': result.details
                }
                for result in report.results
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试报告已保存: {output_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOS烟雾测试套件')
    parser.add_argument('--environment', default='production', help='测试环境')
    parser.add_argument('--output', help='报告输出文件')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 运行烟雾测试
    runner = SmokeTestRunner()
    report = runner.run_all_tests(args.environment)
    
    # 保存报告
    runner.save_report(report, args.output)
    
    # 输出结果摘要
    print(f"\n烟雾测试完成:")
    print(f"  总测试数: {report.total_tests}")
    print(f"  通过测试: {report.passed_tests}")
    print(f"  失败测试: {report.failed_tests}")
    print(f"  成功率: {report.success_rate:.1f}%")
    print(f"  总耗时: {report.total_duration:.1f}秒")
    
    if report.failed_tests > 0:
        print("\n失败的测试:")
        for result in report.results:
            if not result.success:
                print(f"  - {result.name}: {result.message}")
    
    # 返回适当的退出码
    return 0 if report.success_rate >= 80 else 1

if __name__ == "__main__":
    exit(main())