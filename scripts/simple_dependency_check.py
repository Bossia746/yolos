#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的依赖分析工具
检查项目的关键架构问题
"""

import os
import ast
import sys
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

class SimpleDependencyAnalyzer:
    """简化依赖分析器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        self.files: List[str] = []
        
    def scan_project(self):
        """扫描项目文件"""
        print("🔍 扫描项目文件...")
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            rel_path = str(py_file.relative_to(self.project_root))
            self.files.append(rel_path)
            
            try:
                self._analyze_file(py_file, rel_path)
            except Exception as e:
                print(f"⚠️ 跳过文件 {rel_path}: {e}")
        
        print(f"📁 扫描完成，共 {len(self.files)} 个文件")
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否跳过文件"""
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules'}
        return any(part in skip_dirs for part in file_path.parts)
    
    def _analyze_file(self, file_path: Path, rel_path: str):
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.imports[rel_path].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.imports[rel_path].add(node.module)
        except Exception:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
                tree = ast.parse(content)
                # 处理导入...
            except Exception:
                raise
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """查找循环依赖"""
        print("🔄 检查循环依赖...")
        
        # 构建依赖图
        graph = defaultdict(set)
        for file, imports in self.imports.items():
            for imp in imports:
                # 转换为相对路径
                target_file = self._resolve_import_to_file(imp)
                if target_file and target_file in self.files:
                    graph[file].add(target_file)
        
        # 使用DFS检测循环
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # 找到循环
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                dfs(neighbor, path + [neighbor])
            
            rec_stack.remove(node)
        
        for file in self.files:
            if file not in visited:
                dfs(file, [file])
        
        return cycles
    
    def _resolve_import_to_file(self, import_name: str) -> str:
        """将导入名称解析为文件路径"""
        # 简化的解析逻辑
        if import_name.startswith('.'):
            return None  # 相对导入，暂时跳过
        
        # 转换为文件路径
        parts = import_name.split('.')
        
        # 检查是否是项目内的模块
        for i in range(len(parts), 0, -1):
            potential_path = '/'.join(parts[:i]) + '.py'
            if potential_path in self.files:
                return potential_path
            
            # 检查包结构
            potential_dir = '/'.join(parts[:i])
            init_file = f"{potential_dir}/__init__.py"
            if init_file in self.files:
                return init_file
        
        return None
    
    def find_duplicate_classes(self) -> Dict[str, List[str]]:
        """查找重复的类定义"""
        print("🔍 检查重复类...")
        
        class_definitions = defaultdict(list)
        
        for file_path in self.files:
            try:
                full_path = self.project_root / file_path
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_definitions[node.name].append(file_path)
            except Exception:
                continue
        
        # 只返回有重复的类
        duplicates = {name: files for name, files in class_definitions.items() 
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
                
                # 简单的复杂度计算：行数 + 类数 + 函数数
                line_count = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
                
                try:
                    tree = ast.parse(''.join(lines))
                    class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
                    func_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
                    
                    complexity[file_path] = line_count + class_count * 5 + func_count * 2
                except:
                    complexity[file_path] = line_count
                    
            except Exception:
                complexity[file_path] = 0
        
        return complexity
    
    def generate_report(self):
        """生成分析报告"""
        print("\n" + "="*50)
        print("📋 YOLOS项目架构分析报告")
        print("="*50)
        
        # 循环依赖
        cycles = self.find_circular_dependencies()
        print(f"\n🔄 循环依赖检查:")
        if cycles:
            print(f"❌ 发现 {len(cycles)} 个循环依赖:")
            for i, cycle in enumerate(cycles[:5], 1):
                print(f"   {i}. {' -> '.join(cycle)}")
            if len(cycles) > 5:
                print(f"   ... 还有 {len(cycles) - 5} 个")
        else:
            print("✅ 未发现循环依赖")
        
        # 重复类
        duplicates = self.find_duplicate_classes()
        print(f"\n🔍 重复类检查:")
        if duplicates:
            print(f"❌ 发现 {len(duplicates)} 个重复类:")
            for class_name, files in list(duplicates.items())[:5]:
                print(f"   - {class_name}: {', '.join(files)}")
            if len(duplicates) > 5:
                print(f"   ... 还有 {len(duplicates) - 5} 个")
        else:
            print("✅ 未发现重复类定义")
        
        # 复杂度分析
        complexity = self.analyze_complexity()
        high_complexity = {f: c for f, c in complexity.items() if c > 200}
        
        print(f"\n📊 复杂度分析:")
        print(f"   总文件数: {len(self.files)}")
        print(f"   高复杂度文件 (>200): {len(high_complexity)}")
        
        if high_complexity:
            print("   最复杂的文件:")
            sorted_complex = sorted(high_complexity.items(), key=lambda x: x[1], reverse=True)
            for file, comp in sorted_complex[:5]:
                print(f"     - {file}: {comp}")
        
        # 建议
        print(f"\n💡 优化建议:")
        if cycles:
            print("   - 重构循环依赖，使用依赖注入或接口抽象")
        if duplicates:
            print("   - 合并重复类，提取公共基类")
        if high_complexity:
            print("   - 拆分高复杂度文件，遵循单一职责原则")
        
        if not cycles and not duplicates and not high_complexity:
            print("   ✅ 项目架构良好，无需特别优化")

def main():
    """主函数"""
    project_root = os.getcwd()
    analyzer = SimpleDependencyAnalyzer(project_root)
    
    try:
        analyzer.scan_project()
        analyzer.generate_report()
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())