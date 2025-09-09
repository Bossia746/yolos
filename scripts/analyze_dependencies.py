#!/usr/bin/env python3
"""
YOLOS项目依赖分析工具
检查循环依赖、重复设计和架构问题
"""

import os
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
from collections import defaultdict, deque
import re

class DependencyAnalyzer:
    """依赖关系分析器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.src_root = self.project_root / "src"
        self.dependencies = defaultdict(set)
        self.modules = {}
        self.circular_deps = []
        self.redundant_modules = []
        self.complexity_issues = []
        
    def analyze_project(self):
        """分析整个项目"""
        print("🔍 开始分析YOLOS项目依赖关系...")
        
        # 1. 扫描所有Python文件
        self._scan_python_files()
        
        # 2. 分析导入依赖
        self._analyze_imports()
        
        # 3. 检查循环依赖
        self._detect_circular_dependencies()
        
        # 4. 检查重复和冗余
        self._detect_redundancy()
        
        # 5. 分析复杂度
        self._analyze_complexity()
        
        # 6. 生成报告
        self._generate_report()
        
    def _scan_python_files(self):
        """扫描所有Python文件"""
        print("📁 扫描Python文件...")
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 解析AST
                tree = ast.parse(content)
                
                # 获取模块路径
                module_path = self._get_module_path(py_file)
                
                self.modules[module_path] = {
                    'file_path': str(py_file),
                    'ast': tree,
                    'content': content,
                    'imports': [],
                    'classes': [],
                    'functions': [],
                    'lines': len(content.splitlines())
                }
                
                # 分析模块内容
                self._analyze_module_content(module_path, tree)
                
            except Exception as e:
                print(f"⚠️ 无法解析文件 {py_file}: {e}")
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否跳过文件"""
        skip_patterns = [
            '__pycache__',
            '.git',
            'yolos_env',
            'archive',
            'logs',
            'test_results'
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _get_module_path(self, file_path: Path) -> str:
        """获取模块路径"""
        try:
            rel_path = file_path.relative_to(self.project_root)
            module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            return '.'.join(module_parts)
        except ValueError:
            return str(file_path.stem)
    
    def _analyze_module_content(self, module_path: str, tree: ast.AST):
        """分析模块内容"""
        module_info = self.modules[module_path]
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._extract_imports(module_path, node)
            elif isinstance(node, ast.ClassDef):
                module_info['classes'].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                module_info['functions'].append(node.name)
    
    def _extract_imports(self, module_path: str, node: ast.AST):
        """提取导入信息"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_name = alias.name
                self.dependencies[module_path].add(import_name)
                self.modules[module_path]['imports'].append(import_name)
                
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                import_name = node.module
                self.dependencies[module_path].add(import_name)
                self.modules[module_path]['imports'].append(import_name)
    
    def _analyze_imports(self):
        """分析导入依赖"""
        print("🔗 分析导入依赖...")
        
        # 过滤内部依赖
        internal_deps = defaultdict(set)
        
        for module, deps in self.dependencies.items():
            for dep in deps:
                # 只关注项目内部依赖
                if self._is_internal_dependency(dep):
                    internal_deps[module].add(dep)
        
        self.dependencies = internal_deps
    
    def _is_internal_dependency(self, dep_name: str) -> bool:
        """判断是否为内部依赖"""
        # 检查是否为项目内部模块
        internal_prefixes = ['src', 'config', 'scripts', 'tests']
        
        return any(dep_name.startswith(prefix) for prefix in internal_prefixes) or \
               dep_name in self.modules
    
    def _detect_circular_dependencies(self):
        """检测循环依赖"""
        print("🔄 检测循环依赖...")
        
        def dfs(node, path, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependencies.get(node, []):
                if neighbor in self.modules:
                    if neighbor in rec_stack:
                        # 找到循环依赖
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        self.circular_deps.append(cycle)
                    elif neighbor not in visited:
                        dfs(neighbor, path[:], visited, rec_stack)
            
            rec_stack.remove(node)
        
        visited = set()
        for module in self.modules:
            if module not in visited:
                dfs(module, [], visited, set())
    
    def _detect_redundancy(self):
        """检测重复和冗余"""
        print("🔍 检测重复和冗余...")
        
        # 1. 检查相似的类名和函数名
        all_classes = defaultdict(list)
        all_functions = defaultdict(list)
        
        for module_path, module_info in self.modules.items():
            for class_name in module_info['classes']:
                all_classes[class_name].append(module_path)
            for func_name in module_info['functions']:
                all_functions[func_name].append(module_path)
        
        # 找出重复的类和函数
        duplicate_classes = {name: modules for name, modules in all_classes.items() if len(modules) > 1}
        duplicate_functions = {name: modules for name, modules in all_functions.items() if len(modules) > 1}
        
        # 2. 检查相似的模块功能
        self._detect_similar_modules()
        
        self.redundant_modules.extend([
            {'type': 'duplicate_classes', 'data': duplicate_classes},
            {'type': 'duplicate_functions', 'data': duplicate_functions}
        ])
    
    def _detect_similar_modules(self):
        """检测相似模块"""
        # 基于文件名和功能相似性检测
        gui_modules = []
        detector_modules = []
        training_modules = []
        
        for module_path, module_info in self.modules.items():
            if 'gui' in module_path.lower():
                gui_modules.append(module_path)
            elif 'detect' in module_path.lower() or 'recognition' in module_path.lower():
                detector_modules.append(module_path)
            elif 'train' in module_path.lower():
                training_modules.append(module_path)
        
        # 检查是否有过多相似模块
        if len(gui_modules) > 3:
            self.redundant_modules.append({
                'type': 'excessive_gui_modules',
                'data': gui_modules,
                'recommendation': '考虑合并相似的GUI模块'
            })
        
        if len(detector_modules) > 5:
            self.redundant_modules.append({
                'type': 'excessive_detector_modules', 
                'data': detector_modules,
                'recommendation': '考虑统一检测器接口'
            })
    
    def _analyze_complexity(self):
        """分析复杂度"""
        print("📊 分析代码复杂度...")
        
        for module_path, module_info in self.modules.items():
            # 1. 文件行数复杂度
            if module_info['lines'] > 500:
                self.complexity_issues.append({
                    'type': 'large_file',
                    'module': module_path,
                    'lines': module_info['lines'],
                    'severity': 'high' if module_info['lines'] > 1000 else 'medium'
                })
            
            # 2. 导入数量复杂度
            if len(module_info['imports']) > 20:
                self.complexity_issues.append({
                    'type': 'too_many_imports',
                    'module': module_path,
                    'import_count': len(module_info['imports']),
                    'severity': 'high' if len(module_info['imports']) > 30 else 'medium'
                })
            
            # 3. 类和函数数量
            total_definitions = len(module_info['classes']) + len(module_info['functions'])
            if total_definitions > 20:
                self.complexity_issues.append({
                    'type': 'too_many_definitions',
                    'module': module_path,
                    'definition_count': total_definitions,
                    'severity': 'medium'
                })
    
    def _generate_report(self):
        """生成分析报告"""
        print("📋 生成分析报告...")
        
        report = {
            'project_summary': {
                'total_modules': len(self.modules),
                'total_dependencies': sum(len(deps) for deps in self.dependencies.values()),
                'circular_dependencies': len(self.circular_deps),
                'redundancy_issues': len(self.redundant_modules),
                'complexity_issues': len(self.complexity_issues)
            },
            'circular_dependencies': self.circular_deps,
            'redundant_modules': self.redundant_modules,
            'complexity_issues': self.complexity_issues,
            'module_details': {}
        }
        
        # 添加模块详情
        for module_path, module_info in self.modules.items():
            report['module_details'][module_path] = {
                'file_path': module_info['file_path'],
                'lines': module_info['lines'],
                'imports': len(module_info['imports']),
                'classes': len(module_info['classes']),
                'functions': len(module_info['functions'])
            }
        
        # 保存报告
        docs_dir = self.project_root / "docs"
        docs_dir.mkdir(exist_ok=True)
        report_file = docs_dir / "dependency_analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
        
        print(f"✅ 分析完成！报告已保存到: {report_file}")
        
    def _generate_markdown_report(self, report: dict):
        """生成Markdown格式报告"""
        md_content = f"""# YOLOS项目依赖分析报告

## 📊 项目概览

- **总模块数**: {report['project_summary']['total_modules']}
- **总依赖数**: {report['project_summary']['total_dependencies']}
- **循环依赖**: {report['project_summary']['circular_dependencies']} 个
- **冗余问题**: {report['project_summary']['redundancy_issues']} 个
- **复杂度问题**: {report['project_summary']['complexity_issues']} 个

## 🔄 循环依赖分析

"""
        
        if report['circular_dependencies']:
            md_content += "⚠️ **发现循环依赖**:\n\n"
            for i, cycle in enumerate(report['circular_dependencies'], 1):
                md_content += f"{i}. {' → '.join(cycle)}\n"
        else:
            md_content += "✅ **未发现循环依赖**\n"
        
        md_content += "\n## 🔍 冗余分析\n\n"
        
        if report['redundant_modules']:
            for redundancy in report['redundant_modules']:
                if redundancy['type'] == 'duplicate_classes':
                    md_content += "### 重复类名\n\n"
                    for class_name, modules in redundancy['data'].items():
                        md_content += f"- **{class_name}**: {', '.join(modules)}\n"
                elif redundancy['type'] == 'duplicate_functions':
                    md_content += "### 重复函数名\n\n"
                    for func_name, modules in redundancy['data'].items():
                        md_content += f"- **{func_name}**: {', '.join(modules)}\n"
                elif redundancy['type'] == 'excessive_gui_modules':
                    md_content += f"### 过多GUI模块\n\n"
                    md_content += f"发现 {len(redundancy['data'])} 个GUI模块:\n"
                    for module in redundancy['data']:
                        md_content += f"- {module}\n"
                    md_content += f"\n**建议**: {redundancy['recommendation']}\n\n"
        else:
            md_content += "✅ **未发现明显冗余**\n"
        
        md_content += "\n## 📊 复杂度分析\n\n"
        
        if report['complexity_issues']:
            for issue in report['complexity_issues']:
                severity_icon = "🔴" if issue['severity'] == 'high' else "🟡"
                md_content += f"{severity_icon} **{issue['type']}**: {issue['module']}\n"
                
                if issue['type'] == 'large_file':
                    md_content += f"  - 行数: {issue['lines']}\n"
                elif issue['type'] == 'too_many_imports':
                    md_content += f"  - 导入数: {issue['import_count']}\n"
                elif issue['type'] == 'too_many_definitions':
                    md_content += f"  - 定义数: {issue['definition_count']}\n"
                md_content += "\n"
        else:
            md_content += "✅ **复杂度在合理范围内**\n"
        
        md_content += f"""
## 📋 模块详情

| 模块 | 行数 | 导入 | 类 | 函数 |
|------|------|------|----|----- |
"""
        
        for module_path, details in report['module_details'].items():
            md_content += f"| {module_path} | {details['lines']} | {details['imports']} | {details['classes']} | {details['functions']} |\n"
        
        md_content += f"""
## 🎯 优化建议

### 高优先级
1. **解决循环依赖**: 重构模块间的依赖关系
2. **合并重复模块**: 消除功能重复的模块
3. **拆分大文件**: 将超过500行的文件进行模块化拆分

### 中优先级
1. **减少导入数量**: 优化模块间的耦合度
2. **统一接口设计**: 建立标准化的模块接口
3. **代码重构**: 提高代码的内聚性

### 低优先级
1. **性能优化**: 基于实际使用情况优化性能
2. **文档完善**: 补充模块间的依赖关系文档

---

*报告生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存Markdown报告
        md_file = self.project_root / "docs" / "dependency_analysis_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

def main():
    """主函数"""
    analyzer = DependencyAnalyzer()
    analyzer.analyze_project()

if __name__ == "__main__":
    main()