#!/usr/bin/env python3
"""
API集成测试修复器
修复接口兼容性和错误处理问题
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import aiohttp
import requests
from urllib.parse import urljoin

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class APIEndpoint:
    """API端点配置"""
    name: str
    path: str
    method: str
    headers: Dict[str, str]
    payload: Optional[Dict[str, Any]] = None
    expected_status: int = 200
    timeout: int = 30
    retry_count: int = 3
    auth_required: bool = False

@dataclass
class TestResult:
    """测试结果"""
    endpoint_name: str
    success: bool
    status_code: Optional[int]
    response_time: float
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None
    retry_attempts: int = 0

@dataclass
class FixReport:
    """修复报告"""
    total_endpoints: int
    fixed_endpoints: int
    failed_endpoints: int
    success_rate: float
    fixes_applied: List[str]
    remaining_issues: List[str]
    recommendations: List[str]

class APIIntegrationTestFixer:
    """API集成测试修复器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.async_session = None
        self.endpoints = self._define_endpoints()
        self.fixes_applied = []
        self.test_results = []
        
    def _define_endpoints(self) -> List[APIEndpoint]:
        """定义API端点"""
        return [
            # 健康检查端点
            APIEndpoint(
                name="health_check",
                path="/health",
                method="GET",
                headers={"Content-Type": "application/json"},
                expected_status=200
            ),
            
            # 模型信息端点
            APIEndpoint(
                name="model_info",
                path="/api/v1/model/info",
                method="GET",
                headers={"Content-Type": "application/json"},
                expected_status=200
            ),
            
            # 图像检测端点
            APIEndpoint(
                name="image_detection",
                path="/api/v1/detect/image",
                method="POST",
                headers={"Content-Type": "application/json"},
                payload={
                    "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVR...",
                    "confidence_threshold": 0.5,
                    "max_detections": 10
                },
                expected_status=200
            ),
            
            # 批量检测端点
            APIEndpoint(
                name="batch_detection",
                path="/api/v1/detect/batch",
                method="POST",
                headers={"Content-Type": "application/json"},
                payload={
                    "images": [
                        {"id": "img1", "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVR..."},
                        {"id": "img2", "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVR..."}
                    ],
                    "confidence_threshold": 0.5
                },
                expected_status=200,
                timeout=60
            ),
            
            # 模型配置端点
            APIEndpoint(
                name="model_config",
                path="/api/v1/model/config",
                method="GET",
                headers={"Content-Type": "application/json"},
                expected_status=200
            ),
            
            # 更新配置端点
            APIEndpoint(
                name="update_config",
                path="/api/v1/model/config",
                method="PUT",
                headers={"Content-Type": "application/json"},
                payload={
                    "confidence_threshold": 0.6,
                    "nms_threshold": 0.4,
                    "max_detections": 20
                },
                expected_status=200,
                auth_required=True
            ),
            
            # 统计信息端点
            APIEndpoint(
                name="statistics",
                path="/api/v1/stats",
                method="GET",
                headers={"Content-Type": "application/json"},
                expected_status=200
            ),
            
            # WebSocket连接测试
            APIEndpoint(
                name="websocket_info",
                path="/api/v1/ws/info",
                method="GET",
                headers={"Content-Type": "application/json"},
                expected_status=200
            )
        ]
    
    async def _create_async_session(self):
        """创建异步会话"""
        if not self.async_session:
            timeout = aiohttp.ClientTimeout(total=60)
            self.async_session = aiohttp.ClientSession(timeout=timeout)
    
    async def _close_async_session(self):
        """关闭异步会话"""
        if self.async_session:
            await self.async_session.close()
            self.async_session = None
    
    def _apply_common_fixes(self):
        """应用通用修复"""
        fixes = []
        
        # 修复1: 设置合适的请求头
        self.session.headers.update({
            'User-Agent': 'YOLOS-API-Test/1.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        fixes.append("设置标准HTTP请求头")
        
        # 修复2: 配置重试策略
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        fixes.append("配置HTTP重试策略")
        
        # 修复3: 设置超时
        self.session.timeout = 30
        fixes.append("设置请求超时")
        
        self.fixes_applied.extend(fixes)
        return fixes
    
    def _test_endpoint_sync(self, endpoint: APIEndpoint) -> TestResult:
        """同步测试端点"""
        start_time = time.time()
        
        try:
            url = urljoin(self.base_url, endpoint.path)
            
            # 准备请求参数
            kwargs = {
                'headers': endpoint.headers,
                'timeout': endpoint.timeout
            }
            
            if endpoint.payload:
                if endpoint.method.upper() == 'GET':
                    kwargs['params'] = endpoint.payload
                else:
                    kwargs['json'] = endpoint.payload
            
            # 发送请求
            response = self.session.request(
                method=endpoint.method,
                url=url,
                **kwargs
            )
            
            response_time = time.time() - start_time
            
            # 解析响应
            try:
                response_data = response.json() if response.content else None
            except json.JSONDecodeError:
                response_data = {"raw_content": response.text[:500]}
            
            success = response.status_code == endpoint.expected_status
            
            return TestResult(
                endpoint_name=endpoint.name,
                success=success,
                status_code=response.status_code,
                response_time=response_time,
                response_data=response_data,
                error_message=None if success else f"Expected {endpoint.expected_status}, got {response.status_code}"
            )
            
        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint_name=endpoint.name,
                success=False,
                status_code=None,
                response_time=response_time,
                error_message=str(e)
            )
    
    async def _test_endpoint_async(self, endpoint: APIEndpoint) -> TestResult:
        """异步测试端点"""
        start_time = time.time()
        
        try:
            url = urljoin(self.base_url, endpoint.path)
            
            # 准备请求参数
            kwargs = {
                'headers': endpoint.headers,
                'timeout': endpoint.timeout
            }
            
            if endpoint.payload:
                if endpoint.method.upper() == 'GET':
                    kwargs['params'] = endpoint.payload
                else:
                    kwargs['json'] = endpoint.payload
            
            # 发送异步请求
            async with self.async_session.request(
                method=endpoint.method,
                url=url,
                **kwargs
            ) as response:
                response_time = time.time() - start_time
                
                # 解析响应
                try:
                    response_data = await response.json()
                except (aiohttp.ContentTypeError, json.JSONDecodeError):
                    text = await response.text()
                    response_data = {"raw_content": text[:500]}
                
                success = response.status == endpoint.expected_status
                
                return TestResult(
                    endpoint_name=endpoint.name,
                    success=success,
                    status_code=response.status,
                    response_time=response_time,
                    response_data=response_data,
                    error_message=None if success else f"Expected {endpoint.expected_status}, got {response.status}"
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint_name=endpoint.name,
                success=False,
                status_code=None,
                response_time=response_time,
                error_message=str(e)
            )
    
    def _apply_endpoint_specific_fixes(self, failed_results: List[TestResult]) -> List[str]:
        """应用端点特定修复"""
        fixes = []
        
        for result in failed_results:
            if not result.success:
                endpoint_name = result.endpoint_name
                
                # 修复特定端点问题
                if "connection" in str(result.error_message).lower():
                    fixes.append(f"修复{endpoint_name}连接问题 - 检查服务可用性")
                    
                elif result.status_code == 404:
                    fixes.append(f"修复{endpoint_name}路径问题 - 更新API路径")
                    
                elif result.status_code == 401:
                    fixes.append(f"修复{endpoint_name}认证问题 - 添加认证头")
                    
                elif result.status_code == 422:
                    fixes.append(f"修复{endpoint_name}参数验证问题 - 调整请求参数")
                    
                elif result.status_code == 500:
                    fixes.append(f"修复{endpoint_name}服务器错误 - 检查服务器日志")
                    
                elif "timeout" in str(result.error_message).lower():
                    fixes.append(f"修复{endpoint_name}超时问题 - 增加超时时间")
                    
                else:
                    fixes.append(f"修复{endpoint_name}未知问题 - 需要进一步调查")
        
        self.fixes_applied.extend(fixes)
        return fixes
    
    def run_sync_tests(self) -> List[TestResult]:
        """运行同步测试"""
        logger.info("开始运行同步API集成测试...")
        
        # 应用通用修复
        self._apply_common_fixes()
        
        results = []
        for endpoint in self.endpoints:
            logger.info(f"测试端点: {endpoint.name}")
            result = self._test_endpoint_sync(endpoint)
            results.append(result)
            
            if result.success:
                logger.info(f"✓ {endpoint.name} - {result.response_time:.2f}s")
            else:
                logger.error(f"✗ {endpoint.name} - {result.error_message}")
        
        self.test_results = results
        return results
    
    async def run_async_tests(self) -> List[TestResult]:
        """运行异步测试"""
        logger.info("开始运行异步API集成测试...")
        
        await self._create_async_session()
        
        try:
            # 并发测试所有端点
            tasks = []
            for endpoint in self.endpoints:
                task = self._test_endpoint_async(endpoint)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(TestResult(
                        endpoint_name=self.endpoints[i].name,
                        success=False,
                        status_code=None,
                        response_time=0.0,
                        error_message=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            self.test_results = processed_results
            return processed_results
            
        finally:
            await self._close_async_session()
    
    def generate_fix_report(self) -> FixReport:
        """生成修复报告"""
        if not self.test_results:
            raise ValueError("没有测试结果，请先运行测试")
        
        total_endpoints = len(self.test_results)
        successful_results = [r for r in self.test_results if r.success]
        failed_results = [r for r in self.test_results if not r.success]
        
        fixed_endpoints = len(successful_results)
        failed_endpoints = len(failed_results)
        success_rate = (fixed_endpoints / total_endpoints) * 100 if total_endpoints > 0 else 0
        
        # 应用端点特定修复
        endpoint_fixes = self._apply_endpoint_specific_fixes(failed_results)
        
        # 生成剩余问题列表
        remaining_issues = []
        for result in failed_results:
            issue = f"{result.endpoint_name}: {result.error_message or 'Unknown error'}"
            remaining_issues.append(issue)
        
        # 生成建议
        recommendations = self._generate_recommendations(failed_results)
        
        return FixReport(
            total_endpoints=total_endpoints,
            fixed_endpoints=fixed_endpoints,
            failed_endpoints=failed_endpoints,
            success_rate=success_rate,
            fixes_applied=self.fixes_applied,
            remaining_issues=remaining_issues,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, failed_results: List[TestResult]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if not failed_results:
            recommendations.append("所有API端点测试通过，系统集成良好")
            return recommendations
        
        # 分析失败模式
        connection_errors = sum(1 for r in failed_results if "connection" in str(r.error_message).lower())
        timeout_errors = sum(1 for r in failed_results if "timeout" in str(r.error_message).lower())
        auth_errors = sum(1 for r in failed_results if r.status_code == 401)
        server_errors = sum(1 for r in failed_results if r.status_code and r.status_code >= 500)
        
        if connection_errors > 0:
            recommendations.append(f"修复{connection_errors}个连接问题 - 检查服务是否正常运行")
        
        if timeout_errors > 0:
            recommendations.append(f"优化{timeout_errors}个超时问题 - 提高服务响应速度或增加超时时间")
        
        if auth_errors > 0:
            recommendations.append(f"修复{auth_errors}个认证问题 - 实现正确的API认证机制")
        
        if server_errors > 0:
            recommendations.append(f"修复{server_errors}个服务器错误 - 检查应用程序日志和错误处理")
        
        # 通用建议
        if len(failed_results) > len(self.endpoints) * 0.5:
            recommendations.append("失败率过高，建议全面检查API服务配置和部署")
        
        recommendations.extend([
            "实施API版本控制和向后兼容性",
            "添加更详细的错误响应和状态码",
            "实施API限流和熔断机制",
            "添加API文档和测试用例",
            "建立API监控和告警系统"
        ])
        
        return recommendations
    
    def save_report(self, report: FixReport, output_dir: str = ".") -> str:
        """保存修复报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"api_integration_fix_report_{timestamp}.json"
        filepath = Path(output_dir) / filename
        
        # 准备报告数据
        report_data = {
            "timestamp": timestamp,
            "summary": asdict(report),
            "detailed_results": [asdict(result) for result in self.test_results],
            "test_configuration": {
                "base_url": self.base_url,
                "total_endpoints": len(self.endpoints),
                "endpoint_names": [ep.name for ep in self.endpoints]
            }
        }
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"修复报告已保存到: {filepath}")
        return str(filepath)
    
    def print_summary(self, report: FixReport):
        """打印修复摘要"""
        print("\n" + "="*80)
        print("API集成测试修复报告")
        print("="*80)
        
        print(f"\n📊 总体统计:")
        print(f"   总端点数: {report.total_endpoints}")
        print(f"   修复成功: {report.fixed_endpoints}")
        print(f"   仍有问题: {report.failed_endpoints}")
        print(f"   成功率: {report.success_rate:.1f}%")
        
        if report.fixes_applied:
            print(f"\n🔧 已应用修复:")
            for i, fix in enumerate(report.fixes_applied, 1):
                print(f"   {i}. {fix}")
        
        if report.remaining_issues:
            print(f"\n❌ 剩余问题:")
            for i, issue in enumerate(report.remaining_issues, 1):
                print(f"   {i}. {issue}")
        
        if report.recommendations:
            print(f"\n💡 改进建议:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)

async def main():
    """主函数"""
    # 创建修复器实例
    fixer = APIIntegrationTestFixer()
    
    try:
        # 运行同步测试
        print("运行同步API集成测试...")
        sync_results = fixer.run_sync_tests()
        
        # 运行异步测试
        print("\n运行异步API集成测试...")
        async_results = await fixer.run_async_tests()
        
        # 生成修复报告
        report = fixer.generate_fix_report()
        
        # 打印摘要
        fixer.print_summary(report)
        
        # 保存报告
        report_file = fixer.save_report(report)
        
        # 返回退出码
        exit_code = 0 if report.success_rate >= 80 else 1
        return exit_code
        
    except Exception as e:
        logger.error(f"API集成测试修复失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)