#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS系统验证脚本

自动化验证系统的多平台兼容性和商用标准合规性

Usage:
    python validate_system.py [--output-format json|html] [--save-report path]

Author: YOLOS Team
Version: 1.0.0
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from core.platform_compatibility import get_platform_manager
    from core.commercial_standards import get_standards_validator
    from core.performance_monitor import get_performance_monitor
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在YOLOS项目根目录下运行此脚本")
    sys.exit(1)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('system_validation.log')
        ]
    )
    return logging.getLogger("yolos.validation")


def validate_platform_compatibility(logger: logging.Logger) -> Dict[str, Any]:
    """验证平台兼容性"""
    logger.info("开始平台兼容性验证...")
    
    try:
        platform_manager = get_platform_manager()
        
        # 获取平台信息
        platform_info = platform_manager.get_platform_info()
        if not platform_info:
            raise Exception("无法获取平台信息")
        
        logger.info(f"检测到平台: {platform_info.platform_type.value} ({platform_info.device_type.value})")
        logger.info(f"系统: {platform_info.os_name} {platform_info.os_version}")
        logger.info(f"硬件: {platform_info.hardware.cpu_count}核CPU, {platform_info.hardware.memory_total_gb:.1f}GB内存")
        
        # 生成兼容性报告
        compatibility_report = platform_manager.generate_compatibility_report()
        
        # 获取支持的应用类型
        supported_apps = platform_manager.get_supported_applications()
        logger.info(f"支持的应用类型: {', '.join(supported_apps)}")
        
        # 获取优化建议
        optimization_suggestions = platform_manager.get_optimization_suggestions()
        if optimization_suggestions:
            logger.info("平台优化建议:")
            for suggestion in optimization_suggestions:
                logger.info(f"  - {suggestion}")
        
        return {
            'status': 'success',
            'platform_info': platform_info.to_dict(),
            'compatibility_report': compatibility_report,
            'supported_applications': supported_apps,
            'optimization_suggestions': optimization_suggestions
        }
        
    except Exception as e:
        logger.error(f"平台兼容性验证失败: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def validate_commercial_standards(logger: logging.Logger) -> Dict[str, Any]:
    """验证商用标准"""
    logger.info("开始商用标准验证...")
    
    try:
        standards_validator = get_standards_validator()
        
        # 生成商用标准报告
        standards_report = standards_validator.generate_commercial_standards_report()
        
        logger.info(f"商用标准等级: {standards_report.overall_level.value}")
        logger.info(f"总体评分: {standards_report.overall_score:.2f}")
        
        # 显示性能基准结果
        logger.info("性能基准测试结果:")
        for benchmark in standards_report.performance_benchmarks:
            status = "✓" if benchmark.result.value == "pass" else "✗"
            logger.info(f"  {status} {benchmark.test_name}: {benchmark.score}{benchmark.unit} (阈值: {benchmark.threshold}{benchmark.unit})")
        
        # 显示稳定性测试结果
        logger.info("稳定性测试结果:")
        for test in standards_report.stability_tests:
            status = "✓" if test.result.value == "pass" else "✗"
            logger.info(f"  {status} {test.test_name}: {test.score:.2%} 成功率")
        
        # 显示安全检查结果
        logger.info("安全检查结果:")
        for check in standards_report.security_checks:
            status_map = {"pass": "✓", "fail": "✗", "warning": "⚠"}
            status = status_map.get(check.result.value, "?")
            logger.info(f"  {status} {check.check_name}: {check.description} ({check.severity})")
        
        # 显示改进建议
        if standards_report.recommendations:
            logger.info("改进建议:")
            for recommendation in standards_report.recommendations[:5]:  # 只显示前5个
                logger.info(f"  - {recommendation}")
        
        return {
            'status': 'success',
            'standards_report': standards_report.to_dict()
        }
        
    except Exception as e:
        logger.error(f"商用标准验证失败: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def validate_performance_monitoring(logger: logging.Logger) -> Dict[str, Any]:
    """验证性能监控系统"""
    logger.info("开始性能监控系统验证...")
    
    try:
        performance_monitor = get_performance_monitor()
        
        # 启动监控
        performance_monitor.start_monitoring()
        
        # 等待收集一些数据
        import time
        time.sleep(2)
        
        # 获取当前性能指标
        current_metrics = performance_monitor.get_current_metrics()
        if current_metrics:
            logger.info(f"当前性能指标:")
            logger.info(f"  CPU使用率: {current_metrics.cpu_percent:.1f}%")
            logger.info(f"  内存使用率: {current_metrics.memory_percent:.1f}%")
            logger.info(f"  内存使用量: {current_metrics.memory_used_mb:.1f}MB")
            if current_metrics.gpu_percent is not None:
                logger.info(f"  GPU使用率: {current_metrics.gpu_percent:.1f}%")
        
        # 获取系统信息
        system_info = performance_monitor.get_system_info()
        logger.info(f"系统信息: {system_info['cpu_count']}核CPU, {system_info['memory_total_gb']:.1f}GB内存")
        if system_info['gpu_count'] > 0:
            logger.info(f"GPU信息: {system_info['gpu_count']}个GPU")
        
        # 生成性能报告
        performance_report = performance_monitor.generate_performance_report()
        
        # 停止监控
        performance_monitor.stop_monitoring()
        
        return {
            'status': 'success',
            'current_metrics': current_metrics.to_dict() if current_metrics else None,
            'system_info': system_info,
            'performance_report': performance_report
        }
        
    except Exception as e:
        logger.error(f"性能监控系统验证失败: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def generate_html_report(validation_results: Dict[str, Any], output_path: str):
    """生成HTML格式的验证报告"""
    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOS系统验证报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #333; }
        .header { text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .success { background-color: #d4edda; border-color: #c3e6cb; }
        .warning { background-color: #fff3cd; border-color: #ffeaa7; }
        .error { background-color: #f8d7da; border-color: #f5c6cb; }
        .metric { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #eee; }
        .metric:last-child { border-bottom: none; }
        .status-pass { color: #28a745; font-weight: bold; }
        .status-fail { color: #dc3545; font-weight: bold; }
        .status-warning { color: #ffc107; font-weight: bold; }
        .recommendations { background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin-top: 15px; }
        .recommendations ul { margin: 0; padding-left: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>YOLOS系统验证报告</h1>
            <p class="timestamp">生成时间: {timestamp}</p>
        </div>
        
        {content}
    </div>
</body>
</html>
    """
    
    content_sections = []
    
    # 平台兼容性部分
    if 'platform_compatibility' in validation_results:
        platform_data = validation_results['platform_compatibility']
        if platform_data['status'] == 'success':
            platform_info = platform_data['platform_info']
            compatibility_report = platform_data['compatibility_report']
            
            section_class = "success" if compatibility_report['overall_score'] > 0.8 else "warning"
            
            platform_section = f"""
            <div class="section {section_class}">
                <h2>平台兼容性验证</h2>
                <div class="metric">
                    <span>平台类型:</span>
                    <span>{platform_info['platform_type']} ({platform_info['device_type']})</span>
                </div>
                <div class="metric">
                    <span>操作系统:</span>
                    <span>{platform_info['os_name']} {platform_info['os_version']}</span>
                </div>
                <div class="metric">
                    <span>硬件配置:</span>
                    <span>{platform_info['hardware']['cpu_count']}核CPU, {platform_info['hardware']['memory_total_gb']:.1f}GB内存</span>
                </div>
                <div class="metric">
                    <span>总体兼容性评分:</span>
                    <span>{compatibility_report['overall_score']:.2f}</span>
                </div>
                <div class="metric">
                    <span>支持的应用类型:</span>
                    <span>{', '.join(platform_data['supported_applications'])}</span>
                </div>
            </div>
            """
            content_sections.append(platform_section)
    
    # 商用标准部分
    if 'commercial_standards' in validation_results:
        standards_data = validation_results['commercial_standards']
        if standards_data['status'] == 'success':
            report = standards_data['standards_report']
            
            level_colors = {
                'enterprise': 'success',
                'production': 'success', 
                'beta': 'warning',
                'development': 'error'
            }
            section_class = level_colors.get(report['overall_level'], 'warning')
            
            # 性能基准表格
            perf_table = "<table><tr><th>测试项目</th><th>结果</th><th>得分</th><th>阈值</th></tr>"
            for benchmark in report['performance_benchmarks']:
                status_class = f"status-{benchmark['result']}"
                perf_table += f"""
                <tr>
                    <td>{benchmark['test_name']}</td>
                    <td class="{status_class}">{benchmark['result'].upper()}</td>
                    <td>{benchmark['score']}{benchmark['unit']}</td>
                    <td>{benchmark['threshold']}{benchmark['unit']}</td>
                </tr>
                """
            perf_table += "</table>"
            
            standards_section = f"""
            <div class="section {section_class}">
                <h2>商用标准验证</h2>
                <div class="metric">
                    <span>标准等级:</span>
                    <span>{report['overall_level'].upper()}</span>
                </div>
                <div class="metric">
                    <span>总体评分:</span>
                    <span>{report['overall_score']:.2f}</span>
                </div>
                <h3>性能基准测试</h3>
                {perf_table}
            </div>
            """
            content_sections.append(standards_section)
    
    # 性能监控部分
    if 'performance_monitoring' in validation_results:
        perf_data = validation_results['performance_monitoring']
        if perf_data['status'] == 'success' and perf_data['current_metrics']:
            metrics = perf_data['current_metrics']
            
            perf_section = f"""
            <div class="section success">
                <h2>性能监控系统</h2>
                <div class="metric">
                    <span>CPU使用率:</span>
                    <span>{metrics['cpu_percent']:.1f}%</span>
                </div>
                <div class="metric">
                    <span>内存使用率:</span>
                    <span>{metrics['memory_percent']:.1f}%</span>
                </div>
                <div class="metric">
                    <span>内存使用量:</span>
                    <span>{metrics['memory_used_mb']:.1f}MB</span>
                </div>
            </div>
            """
            content_sections.append(perf_section)
    
    # 生成完整HTML
    html_content = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        content='\n'.join(content_sections)
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLOS系统验证脚本")
    parser.add_argument(
        '--output-format',
        choices=['json', 'html'],
        default='json',
        help='输出格式 (默认: json)'
    )
    parser.add_argument(
        '--save-report',
        type=str,
        help='保存报告到指定路径'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    logger.info("开始YOLOS系统验证")
    logger.info("=" * 50)
    
    # 执行各项验证
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'platform_compatibility': validate_platform_compatibility(logger),
        'commercial_standards': validate_commercial_standards(logger),
        'performance_monitoring': validate_performance_monitoring(logger)
    }
    
    # 计算总体状态
    all_success = all(
        result.get('status') == 'success' 
        for result in validation_results.values() 
        if isinstance(result, dict) and 'status' in result
    )
    
    validation_results['overall_status'] = 'success' if all_success else 'partial_success'
    
    logger.info("=" * 50)
    logger.info(f"验证完成，总体状态: {validation_results['overall_status']}")
    
    # 输出结果
    if args.save_report:
        if args.output_format == 'json':
            output_path = args.save_report if args.save_report.endswith('.json') else f"{args.save_report}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON报告已保存到: {output_path}")
        
        elif args.output_format == 'html':
            output_path = args.save_report if args.save_report.endswith('.html') else f"{args.save_report}.html"
            generate_html_report(validation_results, output_path)
            logger.info(f"HTML报告已保存到: {output_path}")
    
    else:
        # 输出到控制台
        if args.output_format == 'json':
            print(json.dumps(validation_results, ensure_ascii=False, indent=2))
        else:
            print("\n验证结果摘要:")
            print(f"总体状态: {validation_results['overall_status']}")
            
            for key, result in validation_results.items():
                if isinstance(result, dict) and 'status' in result:
                    status_symbol = "✓" if result['status'] == 'success' else "✗"
                    print(f"{status_symbol} {key}: {result['status']}")
    
    # 返回适当的退出码
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()