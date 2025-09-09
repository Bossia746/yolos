#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS Security Manager
安全管理器 - 统一管理系统安全功能

主要功能:
- JWT 认证和授权
- API 访问控制
- 数据加密和脱敏
- 审计日志
- 医疗数据安全
- 入侵检测
"""

import os
import jwt
import hashlib
import hmac
import secrets
import logging
import yaml
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from functools import wraps
from pathlib import Path
import json
import re
import ipaddress
import threading
import time
from collections import defaultdict, deque
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class SecurityManager:
    """安全管理器 - 统一管理系统安全功能"""
    
    def __init__(self, config_path: str = None):
        """初始化安全管理器"""
        self.config_path = config_path or "config/security_config.yaml"
        self.config = self._load_security_config()
        self.logger = self._setup_security_logger()
        
        # 初始化加密器
        self.cipher_suite = self._init_encryption()
        
        # 用户会话管理
        self.active_sessions = {}
        self.failed_attempts = defaultdict(list)
        self.blocked_ips = set()
        self.rate_limits = defaultdict(lambda: deque())
        
        # 审计日志
        self.audit_events = []
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 启动清理任务
        self._start_cleanup_tasks()
        
        self.logger.info("Security Manager initialized")
    
    def _load_security_config(self) -> Dict[str, Any]:
        """加载安全配置"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                # 返回默认安全配置
                return self._get_default_security_config()
        except Exception as e:
            print(f"Failed to load security config: {e}")
            return self._get_default_security_config()
    
    def _get_default_security_config(self) -> Dict[str, Any]:
        """获取默认安全配置"""
        return {
            'api_security': {
                'authentication': {'enabled': True, 'method': 'jwt'},
                'authorization': {'enabled': True, 'rbac_enabled': True},
                'access_control': {'rate_limiting': {'enabled': True}}
            },
            'data_protection': {
                'encryption': {'enabled': True},
                'data_masking': {'enabled': True}
            },
            'audit_logging': {'enabled': True},
            'medical_security': {'hipaa_compliance': {'enabled': True}}
        }
    
    def _setup_security_logger(self) -> logging.Logger:
        """设置安全日志记录器"""
        logger = logging.getLogger('security')
        logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 文件处理器
        handler = logging.FileHandler(log_dir / "security_audit.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _init_encryption(self) -> Fernet:
        """初始化加密器"""
        # 从环境变量获取密钥，如果没有则生成新的
        key = os.environ.get('YOLOS_ENCRYPTION_KEY')
        if not key:
            key = Fernet.generate_key()
            # 在生产环境中，应该安全地存储这个密钥
            print(f"Generated new encryption key: {key.decode()}")
            print("Please set YOLOS_ENCRYPTION_KEY environment variable")
        else:
            key = key.encode()
        
        return Fernet(key)
    
    # JWT 认证相关方法
    def generate_jwt_token(self, user_id: str, role: str, 
                          permissions: List[str] = None) -> str:
        """生成 JWT 令牌"""
        if not self.config['api_security']['authentication']['enabled']:
            return None
        
        jwt_config = self.config['api_security']['authentication']['jwt']
        
        payload = {
            'user_id': user_id,
            'role': role,
            'permissions': permissions or [],
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(
                hours=jwt_config.get('expiration_hours', 24)
            )
        }
        
        secret_key = os.environ.get('JWT_SECRET_KEY', 'default-secret-key')
        token = jwt.encode(payload, secret_key, 
                          algorithm=jwt_config.get('algorithm', 'HS256'))
        
        # 记录审计日志
        self._log_security_event('token_generated', {
            'user_id': user_id,
            'role': role
        })
        
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证 JWT 令牌"""
        if not self.config['api_security']['authentication']['enabled']:
            return {'user_id': 'anonymous', 'role': 'viewer'}
        
        try:
            jwt_config = self.config['api_security']['authentication']['jwt']
            secret_key = os.environ.get('JWT_SECRET_KEY', 'default-secret-key')
            
            payload = jwt.decode(token, secret_key, 
                               algorithms=[jwt_config.get('algorithm', 'HS256')])
            
            # 检查令牌是否在黑名单中
            if self._is_token_blacklisted(token):
                return None
            
            return payload
        except jwt.ExpiredSignatureError:
            self._log_security_event('token_expired', {'token': token[:20]})
            return None
        except jwt.InvalidTokenError:
            self._log_security_event('token_invalid', {'token': token[:20]})
            return None
    
    def _is_token_blacklisted(self, token: str) -> bool:
        """检查令牌是否在黑名单中"""
        # 这里应该检查数据库或缓存中的黑名单
        # 简化实现，实际应该使用 Redis 或数据库
        return False
    
    # 权限控制
    def check_permission(self, user_payload: Dict[str, Any], 
                        required_permission: str) -> bool:
        """检查用户权限"""
        if not self.config['api_security']['authorization']['enabled']:
            return True
        
        user_permissions = user_payload.get('permissions', [])
        user_role = user_payload.get('role', '')
        
        # 检查直接权限
        if required_permission in user_permissions:
            return True
        
        # 检查通配符权限
        for permission in user_permissions:
            if permission == '*' or self._match_permission_pattern(permission, required_permission):
                return True
        
        # 检查角色权限
        role_permissions = self._get_role_permissions(user_role)
        if required_permission in role_permissions or '*' in role_permissions:
            return True
        
        return False
    
    def _match_permission_pattern(self, pattern: str, permission: str) -> bool:
        """匹配权限模式"""
        # 将通配符模式转换为正则表达式
        regex_pattern = pattern.replace('*', '.*')
        return re.match(f'^{regex_pattern}$', permission) is not None
    
    def _get_role_permissions(self, role: str) -> List[str]:
        """获取角色权限"""
        roles_config = self.config['api_security']['authorization'].get('roles', {})
        role_config = roles_config.get(role, {})
        return role_config.get('permissions', [])
    
    # 数据加密和脱敏
    def encrypt_data(self, data: str) -> str:
        """加密数据"""
        if not self.config['data_protection']['encryption']['enabled']:
            return data
        
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """解密数据"""
        if not self.config['data_protection']['encryption']['enabled']:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """脱敏敏感数据"""
        if not self.config['data_protection']['data_masking']['enabled']:
            return data
        
        masked_data = data.copy()
        pii_fields = self.config['data_protection']['data_masking'].get('pii_fields', [])
        medical_fields = self.config['data_protection']['data_masking'].get('medical_fields', [])
        
        sensitive_fields = pii_fields + medical_fields
        
        for field in sensitive_fields:
            if field in masked_data:
                masked_data[field] = self._mask_field_value(masked_data[field])
        
        return masked_data
    
    def _mask_field_value(self, value: str) -> str:
        """脱敏字段值"""
        if not value or len(value) < 3:
            return "***"
        
        # 保留前后各一个字符，中间用*替代
        return value[0] + '*' * (len(value) - 2) + value[-1]
    
    # 审计日志
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """记录安全事件"""
        if not self.config['audit_logging']['enabled']:
            return
        
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details,
            'source_ip': details.get('source_ip', 'unknown')
        }
        
        self.audit_events.append(event)
        self.logger.info(f"Security event: {event_type} - {json.dumps(details)}")
    
    # 医疗数据安全
    def validate_medical_access(self, user_payload: Dict[str, Any], 
                               patient_id: str) -> bool:
        """验证医疗数据访问权限"""
        if not self.config['medical_security']['hipaa_compliance']['enabled']:
            return True
        
        # 检查用户是否有医疗数据访问权限
        if not self.check_permission(user_payload, 'medical:read'):
            self._log_security_event('medical_access_denied', {
                'user_id': user_payload.get('user_id'),
                'patient_id': patient_id,
                'reason': 'insufficient_permissions'
            })
            return False
        
        # 记录医疗数据访问
        self._log_security_event('medical_data_accessed', {
            'user_id': user_payload.get('user_id'),
            'patient_id': patient_id
        })
        
        return True
    
    # 速率限制
    def check_rate_limit(self, identifier: str, limit_type: str = 'api') -> bool:
        """检查速率限制"""
        rate_config = self.config.get('rate_limiting', {})
        if not rate_config.get('enabled', True):
            return True
        
        limits = rate_config.get(limit_type, {'requests': 100, 'window': 60})
        max_requests = limits['requests']
        window_seconds = limits['window']
        
        with self._lock:
            now = time.time()
            requests = self.rate_limits[f"{identifier}:{limit_type}"]
            
            # 清理过期的请求记录
            while requests and requests[0] < now - window_seconds:
                requests.popleft()
            
            # 检查是否超过限制
            if len(requests) >= max_requests:
                self._log_security_event('rate_limit_exceeded', {
                    'identifier': identifier,
                    'limit_type': limit_type,
                    'requests_count': len(requests)
                })
                return False
            
            # 记录当前请求
            requests.append(now)
            return True
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """检查IP是否被阻止"""
        return ip_address in self.blocked_ips
    
    def block_ip(self, ip_address: str, reason: str = 'suspicious_activity'):
        """阻止IP地址"""
        with self._lock:
            self.blocked_ips.add(ip_address)
            self._log_security_event('ip_blocked', {
                'ip_address': ip_address,
                'reason': reason
            })
    
    def unblock_ip(self, ip_address: str):
        """解除IP阻止"""
        with self._lock:
            self.blocked_ips.discard(ip_address)
            self._log_security_event('ip_unblocked', {
                'ip_address': ip_address
            })
    
    # 会话管理
    def create_session(self, user_id: str, session_data: Dict[str, Any]) -> str:
        """创建用户会话"""
        session_id = secrets.token_urlsafe(32)
        session_info = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'data': session_data
        }
        
        with self._lock:
            self.active_sessions[session_id] = session_info
        
        self._log_security_event('session_created', {
            'user_id': user_id,
            'session_id': session_id
        })
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """验证会话"""
        session_config = self.config.get('session_management', {})
        timeout_minutes = session_config.get('timeout_minutes', 30)
        
        with self._lock:
            session_info = self.active_sessions.get(session_id)
            if not session_info:
                return None
            
            # 检查会话是否过期
            last_activity = session_info['last_activity']
            if datetime.utcnow() - last_activity > timedelta(minutes=timeout_minutes):
                del self.active_sessions[session_id]
                self._log_security_event('session_expired', {
                    'session_id': session_id,
                    'user_id': session_info['user_id']
                })
                return None
            
            # 更新最后活动时间
            session_info['last_activity'] = datetime.utcnow()
            return session_info
    
    def destroy_session(self, session_id: str):
        """销毁会话"""
        with self._lock:
            session_info = self.active_sessions.pop(session_id, None)
            if session_info:
                self._log_security_event('session_destroyed', {
                    'session_id': session_id,
                    'user_id': session_info['user_id']
                })
    
    # 入侵检测
    def detect_suspicious_activity(self, user_id: str, activity: str, 
                                 source_ip: str) -> bool:
        """检测可疑活动"""
        monitoring_config = self.config.get('security_monitoring', {})
        if not monitoring_config.get('intrusion_detection', {}).get('enabled', False):
            return False
        
        # 检查失败登录次数
        if activity == 'login_failed':
            with self._lock:
                now = time.time()
                attempts = self.failed_attempts[user_id]
                
                # 清理过期的失败尝试记录（1小时内）
                attempts[:] = [attempt_time for attempt_time in attempts 
                             if now - attempt_time < 3600]
                
                # 添加当前失败尝试
                attempts.append(now)
                
                threshold = monitoring_config['intrusion_detection'].get('failed_login_threshold', 5)
                
                if len(attempts) >= threshold:
                    # 自动阻止IP
                    self.block_ip(source_ip, 'multiple_failed_logins')
                    
                    self._log_security_event('suspicious_activity_detected', {
                        'user_id': user_id,
                        'activity': activity,
                        'source_ip': source_ip,
                        'failed_attempts': len(attempts)
                    })
                    return True
        
        return False
    
    def _start_cleanup_tasks(self):
        """启动清理任务"""
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_expired_data()
                    time.sleep(300)  # 每5分钟清理一次
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_data(self):
        """清理过期数据"""
        with self._lock:
            now = time.time()
            
            # 清理过期的速率限制记录
            for key, requests in list(self.rate_limits.items()):
                while requests and requests[0] < now - 3600:  # 1小时
                    requests.popleft()
                if not requests:
                    del self.rate_limits[key]
            
            # 清理过期的失败尝试记录
            for user_id, attempts in list(self.failed_attempts.items()):
                attempts[:] = [attempt_time for attempt_time in attempts 
                             if now - attempt_time < 3600]
                if not attempts:
                    del self.failed_attempts[user_id]
            
            # 清理过期的审计事件（保留最近1000条）
            if len(self.audit_events) > 1000:
                self.audit_events = self.audit_events[-1000:]
    
    # 装饰器
    def require_auth(self, required_permission: str = None):
        """认证装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 这里应该从请求中获取令牌
                # 简化实现，实际应该从 HTTP 头中获取
                token = kwargs.get('auth_token')
                if not token:
                    raise PermissionError("Authentication required")
                
                user_payload = self.verify_jwt_token(token)
                if not user_payload:
                    raise PermissionError("Invalid or expired token")
                
                if required_permission and not self.check_permission(user_payload, required_permission):
                    raise PermissionError(f"Permission denied: {required_permission}")
                
                # 将用户信息添加到参数中
                kwargs['user_payload'] = user_payload
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_medical_auth(self, func):
        """医疗数据访问装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_payload = kwargs.get('user_payload')
            patient_id = kwargs.get('patient_id')
            
            if not self.validate_medical_access(user_payload, patient_id):
                raise PermissionError("Medical data access denied")
            
            return func(*args, **kwargs)
        return wrapper
    
    # 配置管理
    def update_security_config(self, new_config: Dict[str, Any]):
        """更新安全配置"""
        self.config.update(new_config)
        
        # 保存到文件
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            self._log_security_event('security_config_updated', {
                'updated_keys': list(new_config.keys())
            })
        except Exception as e:
            self.logger.error(f"Failed to save security config: {e}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """获取安全状态"""
        return {
            'authentication_enabled': self.config['api_security']['authentication']['enabled'],
            'authorization_enabled': self.config['api_security']['authorization']['enabled'],
            'encryption_enabled': self.config['data_protection']['encryption']['enabled'],
            'audit_logging_enabled': self.config['audit_logging']['enabled'],
            'medical_compliance_enabled': self.config['medical_security']['hipaa_compliance']['enabled'],
            'active_sessions': len(self.active_sessions),
            'recent_security_events': len(self.audit_events[-100:])  # 最近100个事件
        }

# 全局安全管理器实例
security_manager = SecurityManager()

# 导出的装饰器
require_auth = security_manager.require_auth
require_medical_auth = security_manager.require_medical_auth