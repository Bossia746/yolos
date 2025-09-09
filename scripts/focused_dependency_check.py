#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚焦的依赖分析工具
只分析项目核心代码，排除第三方库
"""

import os
import ast
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

class FocusedDependencyAnalyzer:
    """聚焦依赖分析器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        self.files: List[str] = []
        
        # 只分析这些目录
        self.include_dirs = {'src', 'tests', 'examples', 'scripts', 'configs'}
        # 排除这些目录
        self.exclude_dirs = {'yolos_env', '__pycache__', '.git', 'node_modules', 'archive'}
        
    def scan_project(self):
        """扫描项目文件"""
        print("🔍 扫描项目核心文件...")
        
        for py_file in self.project_root.rglob("*.py"):
            if not self._should_include_file(py_file):
                continue
                
            rel_path = str(py_file.relative_to(self.project_root)).replace('\\', '/')
            self.files.append(rel_path)
            
            try:
                self._analyze_file(py_file, rel_path)
            except Exception as e:
                print(f"⚠️ 跳过文件 {rel_path}: {e}")
        
        print(f"📁 扫描完成，共 {len(self.files)} 个核心文件")
    
    def _should_include_file(self, file_path: Path) -> bool:
        """判断是否包含文件"""
        parts = file_path.relative_to(self.project_root).parts
        
        # 排除特定目录
        if any(exclude in parts for exclude in self.exclude_dirs):
            return False
        
        # 根目录的Python文件也包含
        if len(parts) == 1:
            return True
            
        # 包含特定目录
        return parts[0] in self.include_dirs
    
    def _analyze_file(self, file_path: Path, rel_path: str):
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except:
                return
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # 只记录项目内的导入
                        if self._is_project_import(alias.name):
                            self.imports[rel_path].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and self._is_project_import(node.module):
                        self.imports[rel_path].add(node.module)
        except SyntaxError:
            pass  # 跳过语法错误的文件
    
    def _is_project_import(self, import_name: str) -> bool:
        """判断是否是项目内的导入"""
        # 项目内的模块通常以这些开头
        project_prefixes = ['src', 'tests', 'examples', 'scripts', 'configs']
        
        # 相对导入
        if import_name.startswith('.'):
            return True
            
        # 检查是否是项目模块
        return any(import_name.startswith(prefix) for prefix in project_prefixes)
    
    def find_duplicate_classes(self) -> Dict[str, List[str]]:
        """查找重复的类定义"""
        print("🔍 检查重复类...")
        
        class_definitions = defaultdict(list)
        
        for file_path in self.files:
            try:
                full_path = self.project_root / file_path
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(full_path, 'r', encoding='gbk') as f:
                        content = f.read()
                except:
                    continue
            
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_definitions[node.name].append(file_path)
            except SyntaxError:
                continue
        
        # 只返回有重复的类
        duplicates = {name: files for name, files in class_definitions.items() 
                     if len(files) > 1}
        
        return duplicates
    
    def find_duplicate_functions(self) -> Dict[str, List[str]]:
        """查找重复的函数定义"""
        print("🔍 检查重复函数...")
        
        func_definitions = defaultdict(list)
        
        for file_path in self.files:
            try:
                full_path = self.project_root / file_path
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(full_path, 'r', encoding='gbk') as f:
                        content = f.read()
                except:
                    continue
            
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # 排除常见的函数名
                        if node.name not in ['__init__', '__str__', '__repr__', 'main', 'test']:
                            func_definitions[node.name].append(file_path)
            except SyntaxError:
                continue
        
        # 只返回有重复的函数
        duplicates = {name: files for name, files in func_definitions.items() 
                     if len(files) > 1}
        
        return duplicates
    
    def analyze_complexity(self) -> Dict[str, int]:
        """分析文件复杂度"""
        print("📊 分析代码复杂度...")
        
        complexity = {}
        
        for file_path in self.files:
            try:
                full_path = self.project_root / file_path
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                try:
                    with open(full_path, 'r', encoding='gbk') as f:
                        lines = f.readlines()
                except:
                    complexity[file_path] = 0
                    continue
            
            # 计算复杂度
            line_count = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            try:
                tree = ast.parse(''.join(lines))
                class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
                func_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
                
                complexity[file_path] = line_count + class_count * 5 + func_count * 2
            except:
                complexity[file_path] = line_count
        
        return complexity
    
    def check_naming_consistency(self) -> Dict[str, List[str]]:
        """检查命名一致性"""
        print("📝 检查命名一致性...")
        
        issues = defaultdict(list)
        
        for file_path in self.files:
            # 检查文件命名
            filename = Path(file_path).stem
            
            # 检查是否使用了一致的命名风格
            if '_' in filename and any(c.isupper() for c in filename):
                issues['mixed_case_files'].append(file_path)
            
            # 检查是否有过长的文件名
            if len(filename) > 30:
                issues['long_filenames'].append(file_path)
        
        return dict(issues)
    
    def generate_report(self):
        """生成分析报告"""
        print("\n" + "="*60)
        print("📋 YOLOS项目核心架构分析报告")
        print("="*60)
        
        # 重复类
        duplicate_classes = self.find_duplicate_classes()
        print(f"\n🔍 重复类检查:")
        if duplicate_classes:
            print(f"❌ 发现 {len(duplicate_classes)} 个重复类:")
            for class_name, files in list(duplicate_classes.items())[:10]:
                print(f"   - {class_name}: {', '.join(files)}")
            if len(duplicate_classes) > 10:
                print(f"   ... 还有 {len(duplicate_classes) - 10} 个")
        else:
            print("✅ 未发现重复类定义")
        
        # 重复函数
        duplicate_functions = self.find_duplicate_functions()
        print(f"\n🔧 重复函数检查:")
        if duplicate_functions:
            print(f"❌ 发现 {len(duplicate_functions)} 个重复函数:")
            for func_name, files in list(duplicate_functions.items())[:10]:
                print(f"   - {func_name}: {', '.join(files)}")
            if len(duplicate_functions) > 10:
                print(f"   ... 还有 {len(duplicate_functions) - 10} 个")
        else:
            print("✅ 未发现重复函数定义")
        
        # 复杂度分析
        complexity = self.analyze_complexity()
        high_complexity = {f: c for f, c in complexity.items() if c > 200}
        
        print(f"\n📊 复杂度分析:")
        print(f"   总核心文件数: {len(self.files)}")
        print(f"   高复杂度文件 (>200): {len(high_complexity)}")
        
        if high_complexity:
            print("   最复杂的文件:")
            sorted_complex = sorted(high_complexity.items(), key=lambda x: x[1], reverse=True)
            for file, comp in sorted_complex[:10]:
                print(f"     - {file}: {comp}")
        
        # 命名一致性
        naming_issues = self.check_naming_consistency()
        print(f"\n📝 命名一致性检查:")
        if naming_issues:
            for issue_type, files in naming_issues.items():
                print(f"   - {issue_type}: {len(files)} 个文件")
                for file in files[:5]:
                    print(f"     * {file}")
                if len(files) > 5:
                    print(f"     ... 还有 {len(files) - 5} 个")
        else:
            print("✅ 命名风格一致")
        
        # 项目结构分析
        print(f"\n📁 项目结构分析:")
        dir_counts = defaultdict(int)
        for file in self.files:
            if '/' in file:
                dir_counts[file.split('/')[0]] += 1
            else:
                dir_counts['root'] += 1
        
        for dir_name, count in sorted(dir_counts.items()):
            print(f"   - {dir_name}: {count} 个文件")
        
        # 优化建议
        print(f"\n💡 架构优化建议:")
        suggestions = []
        
        if duplicate_classes:
            suggestions.append("合并重复类定义，提取公共基类或接口")
        if duplicate_functions:
            suggestions.append("重构重复函数，提取到工具模块中")
        if high_complexity:
            suggestions.append("拆分高复杂度文件，遵循单一职责原则")
        if naming_issues:
            suggestions.append("统一命名风格，建议使用snake_case")
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
        else:
            print("   ✅ 项目架构良好，符合最佳实践")
        
        # 高内聚低耦合评估
        print(f"\n🏗️ 架构质量评估:")
        total_issues = len(duplicate_classes) + len(duplicate_functions) + len(high_complexity)
        
        if total_issues == 0:
            print("   ✅ 优秀 - 高内聚低耦合，架构清晰")
        elif total_issues <= 5:
            print("   ⚠️ 良好 - 有少量优化空间")
        elif total_issues <= 15:
            print("   ⚠️ 一般 - 需要重构优化")
        else:
            print("   ❌ 需要改进 - 存在较多架构问题")

def main():
    """主函数"""
    project_root = os.getcwd()
    analyzer = FocusedDependencyAnalyzer(project_root)
    
    try:
        analyzer.scan_project()
        analyzer.generate_report()
        
        print(f"\n🎯 分析完成！项目已准备好进行GitHub推送。")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())