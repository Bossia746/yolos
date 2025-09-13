#!/usr/bin/env python3
"""
YOLOS 回滚和恢复管理器
确保部署安全性和系统可靠性
"""

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml
import requests
from packaging import version

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentSnapshot:
    """部署快照"""
    id: str
    timestamp: datetime
    version: str
    environment: str
    services: Dict[str, str]  # service_name -> image_tag
    config_hash: str
    database_backup_path: Optional[str] = None
    volumes_backup_path: Optional[str] = None
    health_status: str = "unknown"
    rollback_tested: bool = False

@dataclass
class RollbackPlan:
    """回滚计划"""
    target_snapshot_id: str
    rollback_steps: List[str]
    estimated_duration: int  # 秒
    risk_level: str  # low, medium, high
    prerequisites: List[str]
    validation_checks: List[str]

@dataclass
class RecoveryReport:
    """恢复报告"""
    operation_type: str  # rollback, recovery, backup
    success: bool
    start_time: datetime
    end_time: datetime
    duration: float
    steps_completed: List[str]
    errors: List[str]
    warnings: List[str]
    final_status: str

class RollbackRecoveryManager:
    """回滚和恢复管理器"""
    
    def __init__(self, config_path: str = "deployment/config/rollback_config.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.snapshots_dir = Path(self.config.get('snapshots_dir', 'deployment/snapshots'))
        self.backups_dir = Path(self.config.get('backups_dir', 'deployment/backups'))
        self.logs_dir = Path(self.config.get('logs_dir', 'deployment/logs'))
        
        # 确保目录存在
        for directory in [self.snapshots_dir, self.backups_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.snapshots: List[DeploymentSnapshot] = []
        self.load_snapshots()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            default_config = {
                'max_snapshots': 10,
                'retention_days': 30,
                'health_check_timeout': 300,
                'rollback_timeout': 600,
                'services': {
                    'yolos-app': {
                        'health_endpoint': 'http://localhost:8000/health',
                        'critical': True
                    },
                    'yolos-web': {
                        'health_endpoint': 'http://localhost:8080/',
                        'critical': False
                    },
                    'postgres': {
                        'health_command': 'pg_isready -U yolos',
                        'critical': True
                    },
                    'redis': {
                        'health_command': 'redis-cli ping',
                        'critical': True
                    }
                },
                'docker_compose_file': 'deployment/docker/docker-compose.optimized.yml',
                'backup_retention_days': 7,
                'notification_webhook': None
            }
            
            # 保存默认配置
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            return default_config
    
    def load_snapshots(self):
        """加载现有快照"""
        snapshots_file = self.snapshots_dir / 'snapshots.json'
        if snapshots_file.exists():
            try:
                with open(snapshots_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for snapshot_data in data.get('snapshots', []):
                    snapshot = DeploymentSnapshot(
                        id=snapshot_data['id'],
                        timestamp=datetime.fromisoformat(snapshot_data['timestamp']),
                        version=snapshot_data['version'],
                        environment=snapshot_data['environment'],
                        services=snapshot_data['services'],
                        config_hash=snapshot_data['config_hash'],
                        database_backup_path=snapshot_data.get('database_backup_path'),
                        volumes_backup_path=snapshot_data.get('volumes_backup_path'),
                        health_status=snapshot_data.get('health_status', 'unknown'),
                        rollback_tested=snapshot_data.get('rollback_tested', False)
                    )
                    self.snapshots.append(snapshot)
                    
                logger.info(f"加载了 {len(self.snapshots)} 个部署快照")
                
            except Exception as e:
                logger.error(f"加载快照失败: {e}")
                self.snapshots = []
    
    def save_snapshots(self):
        """保存快照到文件"""
        snapshots_file = self.snapshots_dir / 'snapshots.json'
        
        data = {
            'snapshots': [
                {
                    **asdict(snapshot),
                    'timestamp': snapshot.timestamp.isoformat()
                }
                for snapshot in self.snapshots
            ]
        }
        
        with open(snapshots_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def create_snapshot(self, version: str, environment: str = "production") -> DeploymentSnapshot:
        """创建部署快照"""
        logger.info(f"创建部署快照: {version} ({environment})")
        
        snapshot_id = f"{environment}_{version}_{int(time.time())}"
        timestamp = datetime.now()
        
        # 获取当前服务状态
        services = self._get_current_services()
        
        # 计算配置哈希
        config_hash = self._calculate_config_hash()
        
        # 创建数据库备份
        database_backup_path = self._backup_database(snapshot_id)
        
        # 创建数据卷备份
        volumes_backup_path = self._backup_volumes(snapshot_id)
        
        # 检查健康状态
        health_status = self._check_system_health()
        
        snapshot = DeploymentSnapshot(
            id=snapshot_id,
            timestamp=timestamp,
            version=version,
            environment=environment,
            services=services,
            config_hash=config_hash,
            database_backup_path=database_backup_path,
            volumes_backup_path=volumes_backup_path,
            health_status=health_status
        )
        
        self.snapshots.append(snapshot)
        self._cleanup_old_snapshots()
        self.save_snapshots()
        
        logger.info(f"快照创建完成: {snapshot_id}")
        return snapshot
    
    def _get_current_services(self) -> Dict[str, str]:
        """获取当前服务状态"""
        services = {}
        
        try:
            # 使用docker-compose获取服务信息
            compose_file = self.config.get('docker_compose_file')
            if compose_file and Path(compose_file).exists():
                result = subprocess.run(
                    ['docker-compose', '-f', compose_file, 'images', '--format', 'json'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                for line in result.stdout.strip().split('\n'):
                    if line:
                        service_info = json.loads(line)
                        services[service_info['Service']] = service_info['Tag']
            
        except Exception as e:
            logger.warning(f"获取服务状态失败: {e}")
            
        return services
    
    def _calculate_config_hash(self) -> str:
        """计算配置文件哈希"""
        import hashlib
        
        config_files = [
            'deployment/docker/docker-compose.optimized.yml',
            'deployment/config',
            'src/config'
        ]
        
        hasher = hashlib.sha256()
        
        for config_path in config_files:
            path = Path(config_path)
            if path.exists():
                if path.is_file():
                    with open(path, 'rb') as f:
                        hasher.update(f.read())
                elif path.is_dir():
                    for file_path in sorted(path.rglob('*')):
                        if file_path.is_file():
                            with open(file_path, 'rb') as f:
                                hasher.update(f.read())
        
        return hasher.hexdigest()[:16]
    
    def _backup_database(self, snapshot_id: str) -> Optional[str]:
        """备份数据库"""
        try:
            backup_file = self.backups_dir / f"db_{snapshot_id}.sql"
            
            # PostgreSQL备份
            result = subprocess.run([
                'docker-compose', '-f', self.config.get('docker_compose_file'),
                'exec', '-T', 'postgres',
                'pg_dump', '-U', 'yolos', '-d', 'yolos'
            ], capture_output=True, text=True, check=True)
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            
            logger.info(f"数据库备份完成: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"数据库备份失败: {e}")
            return None
    
    def _backup_volumes(self, snapshot_id: str) -> Optional[str]:
        """备份数据卷"""
        try:
            volumes_backup_dir = self.backups_dir / f"volumes_{snapshot_id}"
            volumes_backup_dir.mkdir(exist_ok=True)
            
            # 备份重要数据卷
            important_volumes = [
                'deployment/docker/volumes/data',
                'deployment/docker/volumes/logs',
                'models',
                'config'
            ]
            
            for volume_path in important_volumes:
                source = Path(volume_path)
                if source.exists():
                    dest = volumes_backup_dir / source.name
                    if source.is_dir():
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source, dest)
            
            logger.info(f"数据卷备份完成: {volumes_backup_dir}")
            return str(volumes_backup_dir)
            
        except Exception as e:
            logger.error(f"数据卷备份失败: {e}")
            return None
    
    def _check_system_health(self) -> str:
        """检查系统健康状态"""
        health_checks = []
        
        for service_name, service_config in self.config.get('services', {}).items():
            try:
                if 'health_endpoint' in service_config:
                    # HTTP健康检查
                    response = requests.get(
                        service_config['health_endpoint'],
                        timeout=10
                    )
                    health_checks.append(response.status_code == 200)
                    
                elif 'health_command' in service_config:
                    # 命令健康检查
                    result = subprocess.run(
                        service_config['health_command'].split(),
                        capture_output=True,
                        timeout=10
                    )
                    health_checks.append(result.returncode == 0)
                    
            except Exception as e:
                logger.warning(f"健康检查失败 {service_name}: {e}")
                health_checks.append(False)
        
        if all(health_checks):
            return "healthy"
        elif any(health_checks):
            return "degraded"
        else:
            return "unhealthy"
    
    def create_rollback_plan(self, target_snapshot_id: str) -> RollbackPlan:
        """创建回滚计划"""
        target_snapshot = None
        for snapshot in self.snapshots:
            if snapshot.id == target_snapshot_id:
                target_snapshot = snapshot
                break
        
        if not target_snapshot:
            raise ValueError(f"找不到目标快照: {target_snapshot_id}")
        
        current_services = self._get_current_services()
        
        rollback_steps = [
            "1. 停止当前服务",
            "2. 恢复数据库备份",
            "3. 恢复数据卷",
            "4. 更新服务镜像",
            "5. 启动服务",
            "6. 验证健康状态",
            "7. 运行烟雾测试"
        ]
        
        # 评估风险等级
        risk_level = "low"
        if target_snapshot.health_status != "healthy":
            risk_level = "high"
        elif not target_snapshot.rollback_tested:
            risk_level = "medium"
        
        # 估算持续时间
        estimated_duration = 300  # 基础5分钟
        if target_snapshot.database_backup_path:
            estimated_duration += 120  # 数据库恢复
        if target_snapshot.volumes_backup_path:
            estimated_duration += 180  # 数据卷恢复
        
        prerequisites = [
            "确认目标快照完整性",
            "通知相关团队",
            "准备回滚后验证清单"
        ]
        
        validation_checks = [
            "服务健康检查",
            "API功能测试",
            "数据库连接测试",
            "关键业务流程验证"
        ]
        
        return RollbackPlan(
            target_snapshot_id=target_snapshot_id,
            rollback_steps=rollback_steps,
            estimated_duration=estimated_duration,
            risk_level=risk_level,
            prerequisites=prerequisites,
            validation_checks=validation_checks
        )
    
    def execute_rollback(self, target_snapshot_id: str) -> RecoveryReport:
        """执行回滚"""
        start_time = datetime.now()
        steps_completed = []
        errors = []
        warnings = []
        
        logger.info(f"开始执行回滚到快照: {target_snapshot_id}")
        
        try:
            # 获取目标快照
            target_snapshot = None
            for snapshot in self.snapshots:
                if snapshot.id == target_snapshot_id:
                    target_snapshot = snapshot
                    break
            
            if not target_snapshot:
                raise ValueError(f"找不到目标快照: {target_snapshot_id}")
            
            # 步骤1: 停止当前服务
            logger.info("停止当前服务...")
            self._stop_services()
            steps_completed.append("停止当前服务")
            
            # 步骤2: 恢复数据库备份
            if target_snapshot.database_backup_path:
                logger.info("恢复数据库备份...")
                self._restore_database(target_snapshot.database_backup_path)
                steps_completed.append("恢复数据库备份")
            
            # 步骤3: 恢复数据卷
            if target_snapshot.volumes_backup_path:
                logger.info("恢复数据卷...")
                self._restore_volumes(target_snapshot.volumes_backup_path)
                steps_completed.append("恢复数据卷")
            
            # 步骤4: 更新服务镜像
            logger.info("更新服务镜像...")
            self._update_service_images(target_snapshot.services)
            steps_completed.append("更新服务镜像")
            
            # 步骤5: 启动服务
            logger.info("启动服务...")
            self._start_services()
            steps_completed.append("启动服务")
            
            # 步骤6: 验证健康状态
            logger.info("验证健康状态...")
            health_status = self._wait_for_healthy_status()
            if health_status != "healthy":
                warnings.append(f"系统健康状态: {health_status}")
            steps_completed.append("验证健康状态")
            
            # 步骤7: 运行烟雾测试
            logger.info("运行烟雾测试...")
            smoke_test_result = self._run_smoke_tests()
            if not smoke_test_result:
                warnings.append("烟雾测试部分失败")
            steps_completed.append("运行烟雾测试")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"回滚完成，耗时: {duration:.1f}秒")
            
            return RecoveryReport(
                operation_type="rollback",
                success=True,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                steps_completed=steps_completed,
                errors=errors,
                warnings=warnings,
                final_status="success"
            )
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            errors.append(str(e))
            
            logger.error(f"回滚失败: {e}")
            
            return RecoveryReport(
                operation_type="rollback",
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                steps_completed=steps_completed,
                errors=errors,
                warnings=warnings,
                final_status="failed"
            )
    
    def _stop_services(self):
        """停止服务"""
        compose_file = self.config.get('docker_compose_file')
        subprocess.run([
            'docker-compose', '-f', compose_file, 'down'
        ], check=True)
    
    def _start_services(self):
        """启动服务"""
        compose_file = self.config.get('docker_compose_file')
        subprocess.run([
            'docker-compose', '-f', compose_file, 'up', '-d'
        ], check=True)
    
    def _restore_database(self, backup_path: str):
        """恢复数据库"""
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = f.read()
        
        # 启动临时数据库容器进行恢复
        compose_file = self.config.get('docker_compose_file')
        subprocess.run([
            'docker-compose', '-f', compose_file, 'up', '-d', 'postgres'
        ], check=True)
        
        # 等待数据库启动
        time.sleep(10)
        
        # 恢复数据
        process = subprocess.Popen([
            'docker-compose', '-f', compose_file,
            'exec', '-T', 'postgres',
            'psql', '-U', 'yolos', '-d', 'yolos'
        ], stdin=subprocess.PIPE, text=True)
        
        process.communicate(input=backup_data)
        
        if process.returncode != 0:
            raise RuntimeError("数据库恢复失败")
    
    def _restore_volumes(self, backup_path: str):
        """恢复数据卷"""
        backup_dir = Path(backup_path)
        
        # 恢复各个数据卷
        volume_mappings = {
            'data': 'deployment/docker/volumes/data',
            'logs': 'deployment/docker/volumes/logs',
            'models': 'models',
            'config': 'config'
        }
        
        for volume_name, target_path in volume_mappings.items():
            source = backup_dir / volume_name
            target = Path(target_path)
            
            if source.exists():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
    
    def _update_service_images(self, services: Dict[str, str]):
        """更新服务镜像"""
        # 这里可以实现镜像标签更新逻辑
        # 例如修改docker-compose文件或使用docker tag命令
        pass
    
    def _wait_for_healthy_status(self, timeout: int = 300) -> str:
        """等待系统健康"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            health_status = self._check_system_health()
            if health_status == "healthy":
                return health_status
            
            time.sleep(10)
        
        return self._check_system_health()
    
    def _run_smoke_tests(self) -> bool:
        """运行烟雾测试"""
        try:
            # 运行基本的API测试
            result = subprocess.run([
                'python', 'tests/smoke_tests.py'
            ], capture_output=True, timeout=60)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.warning(f"烟雾测试失败: {e}")
            return False
    
    def _cleanup_old_snapshots(self):
        """清理旧快照"""
        max_snapshots = self.config.get('max_snapshots', 10)
        retention_days = self.config.get('retention_days', 30)
        
        # 按时间排序
        self.snapshots.sort(key=lambda x: x.timestamp, reverse=True)
        
        # 删除超过数量限制的快照
        if len(self.snapshots) > max_snapshots:
            old_snapshots = self.snapshots[max_snapshots:]
            self.snapshots = self.snapshots[:max_snapshots]
            
            for snapshot in old_snapshots:
                self._delete_snapshot_files(snapshot)
        
        # 删除超过时间限制的快照
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        remaining_snapshots = []
        
        for snapshot in self.snapshots:
            if snapshot.timestamp > cutoff_date:
                remaining_snapshots.append(snapshot)
            else:
                self._delete_snapshot_files(snapshot)
        
        self.snapshots = remaining_snapshots
    
    def _delete_snapshot_files(self, snapshot: DeploymentSnapshot):
        """删除快照文件"""
        try:
            if snapshot.database_backup_path:
                Path(snapshot.database_backup_path).unlink(missing_ok=True)
            
            if snapshot.volumes_backup_path:
                shutil.rmtree(snapshot.volumes_backup_path, ignore_errors=True)
                
        except Exception as e:
            logger.warning(f"删除快照文件失败 {snapshot.id}: {e}")
    
    def list_snapshots(self) -> List[DeploymentSnapshot]:
        """列出所有快照"""
        return sorted(self.snapshots, key=lambda x: x.timestamp, reverse=True)
    
    def get_latest_healthy_snapshot(self) -> Optional[DeploymentSnapshot]:
        """获取最新的健康快照"""
        for snapshot in sorted(self.snapshots, key=lambda x: x.timestamp, reverse=True):
            if snapshot.health_status == "healthy":
                return snapshot
        return None
    
    def test_rollback(self, snapshot_id: str) -> bool:
        """测试回滚（不实际执行）"""
        try:
            plan = self.create_rollback_plan(snapshot_id)
            logger.info(f"回滚计划创建成功: {plan.target_snapshot_id}")
            logger.info(f"预计耗时: {plan.estimated_duration}秒")
            logger.info(f"风险等级: {plan.risk_level}")
            return True
        except Exception as e:
            logger.error(f"回滚测试失败: {e}")
            return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOS回滚和恢复管理器')
    parser.add_argument('action', choices=['snapshot', 'rollback', 'list', 'test'],
                       help='执行的操作')
    parser.add_argument('--version', help='版本号（用于创建快照）')
    parser.add_argument('--environment', default='production', help='环境名称')
    parser.add_argument('--snapshot-id', help='快照ID（用于回滚）')
    
    args = parser.parse_args()
    
    manager = RollbackRecoveryManager()
    
    if args.action == 'snapshot':
        if not args.version:
            print("创建快照需要指定版本号")
            return 1
        
        snapshot = manager.create_snapshot(args.version, args.environment)
        print(f"快照创建成功: {snapshot.id}")
        
    elif args.action == 'rollback':
        if not args.snapshot_id:
            print("回滚需要指定快照ID")
            return 1
        
        report = manager.execute_rollback(args.snapshot_id)
        print(f"回滚{'成功' if report.success else '失败'}")
        print(f"耗时: {report.duration:.1f}秒")
        
        if report.errors:
            print("错误:")
            for error in report.errors:
                print(f"  - {error}")
        
        if report.warnings:
            print("警告:")
            for warning in report.warnings:
                print(f"  - {warning}")
        
        return 0 if report.success else 1
        
    elif args.action == 'list':
        snapshots = manager.list_snapshots()
        print(f"共有 {len(snapshots)} 个快照:")
        
        for snapshot in snapshots:
            print(f"  {snapshot.id}:")
            print(f"    版本: {snapshot.version}")
            print(f"    时间: {snapshot.timestamp}")
            print(f"    环境: {snapshot.environment}")
            print(f"    健康状态: {snapshot.health_status}")
            print(f"    已测试回滚: {snapshot.rollback_tested}")
            print()
        
    elif args.action == 'test':
        if not args.snapshot_id:
            print("测试回滚需要指定快照ID")
            return 1
        
        success = manager.test_rollback(args.snapshot_id)
        return 0 if success else 1
    
    return 0

if __name__ == "__main__":
    exit(main())