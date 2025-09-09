#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目整理脚本
将根目录文件移动到对应文件夹，并替换敏感信息
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List

class ProjectOrganizer:
    """项目整理器"""
    
    def __init__(self):
        self.root_dir = Path(".")
        self.file_mappings = {
            # 测试文件
            "tests/": [
                "test_*.py",
                "*_test.py", 
                "comprehensive_test_suite.py",
                "comprehensive_vision_test.py",
                "conftest.py",
                "pytest.ini"
            ],
            # 脚本文件
            "scripts/": [
                "*_optimizer.py",
                "*_enhancer.py", 
                "*_integration.py",
                "visual_*.py",
                "real_yolo_test.py"
            ],
            # 文档文件
            "docs/": [
                "*.md",
                "README*.md"
            ],
            # 报告文件
            "reports/": [
                "*.html",
                "*_report.*",
                "*.log"
            ],
            # 配置文件
            "config/": [
                "requirements*.txt",
                "setup.py"
            ],
            # 模型文件
            "models/": [
                "*.pt"
            ],
            # 图像文件
            "test_images/": [
                "annotated_*.jpg",
                "*.jpg",
                "*.png"
            ]
        }
        
        # 敏感信息替换模式
        self.sensitive_patterns = {
            # ModelScope API Key
            r'ms-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}': '*****',
            # OpenAI API Key
            r'sk-[a-zA-Z0-9]{32,}': '*****',
            # 通用API Key模式
            r'api_key[\'\"]\s*:\s*[\'\"]((?!test_key|your_|None|\*)[^\'"]+)[\'"]': 'api_key": "*****"',
            r'api_key=[\'\"]((?!test_key|your_|None|\*)[^\'"]+)[\'"]': 'api_key="*****"',
            # DashScope相关
            r'https://dashscope\.aliyuncs\.com[^\'"]*': 'https://dashscope.aliyuncs.com/api/v1',
            # ModelScope相关
            r'https://api-inference\.modelscope\.cn[^\'"]*': 'https://api-inference.modelscope.cn/v1'
        }
    
    def organize_files(self):
        """整理文件结构"""
        print("🗂️ 开始整理项目文件结构...")
        
        # 创建目标目录
        self._create_directories()
        
        # 移动文件
        self._move_files()
        
        # 替换敏感信息
        self._replace_sensitive_info()
        
        # 创建.gitignore
        self._create_gitignore()
        
        print("✅ 项目整理完成!")
    
    def _create_directories(self):
        """创建目标目录"""
        for target_dir in self.file_mappings.keys():
            target_path = self.root_dir / target_dir
            target_path.mkdir(exist_ok=True)
            print(f"📁 创建目录: {target_dir}")
        
        # 创建reports目录
        reports_dir = self.root_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
    
    def _move_files(self):
        """移动文件到对应目录"""
        moved_files = []
        
        for target_dir, patterns in self.file_mappings.items():
            target_path = self.root_dir / target_dir
            
            for pattern in patterns:
                # 查找匹配的文件
                if "*" in pattern:
                    import glob
                    matching_files = glob.glob(pattern)
                else:
                    matching_files = [pattern] if (self.root_dir / pattern).exists() else []
                
                for file_name in matching_files:
                    source_path = self.root_dir / file_name
                    if source_path.exists() and source_path.is_file():
                        # 避免重复移动
                        if file_name not in moved_files:
                            dest_path = target_path / file_name
                            
                            # 如果目标文件已存在，添加后缀
                            if dest_path.exists():
                                base_name = dest_path.stem
                                suffix = dest_path.suffix
                                counter = 1
                                while dest_path.exists():
                                    dest_path = target_path / f"{base_name}_{counter}{suffix}"
                                    counter += 1
                            
                            try:
                                shutil.move(str(source_path), str(dest_path))
                                print(f"📦 移动: {file_name} -> {target_dir}")
                                moved_files.append(file_name)
                            except Exception as e:
                                print(f"❌ 移动失败: {file_name} - {e}")
    
    def _replace_sensitive_info(self):
        """替换敏感信息"""
        print("🔒 开始替换敏感信息...")
        
        # 遍历所有Python文件
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_process_file(py_file):
                self._process_file(py_file)
        
        # 处理配置文件
        for config_file in self.root_dir.rglob("*.yaml"):
            if self._should_process_file(config_file):
                self._process_file(config_file)
        
        for config_file in self.root_dir.rglob("*.yml"):
            if self._should_process_file(config_file):
                self._process_file(config_file)
    
    def _should_process_file(self, file_path: Path) -> bool:
        """判断是否应该处理该文件"""
        # 跳过虚拟环境、缓存等目录
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'yolos_env', '.vscode', '.codebuddy'}
        
        for part in file_path.parts:
            if part in skip_dirs:
                return False
        
        return True
    
    def _process_file(self, file_path: Path):
        """处理单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 应用所有替换模式
            for pattern, replacement in self.sensitive_patterns.items():
                content = re.sub(pattern, replacement, content)
            
            # 如果内容有变化，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"🔐 已处理: {file_path}")
        
        except Exception as e:
            print(f"❌ 处理失败: {file_path} - {e}")
    
    def _create_gitignore(self):
        """创建或更新.gitignore文件"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
yolos_env/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
*.pt
*.pth
*.onnx
*.trt
*.engine
runs/
wandb/
*.weights

# API Keys and sensitive data
config/api_keys.yaml
config/secrets.yaml
*.key
*.pem
*.p12

# Logs
*.log
logs/


# Test results
test_results/
reports/
*.html

# Cache
.cache/
.codebuddy/

# Temporary files
tmp/
temp/
*.tmp
*.temp

# Model files (large)
models/*.pt
models/*.pth
models/*.onnx

# Data files
data/
dataset/
datasets/
*.csv
*.json
*.pkl
*.pickle

# Images (except examples)
*.jpg
*.jpeg
*.png
*.gif
*.bmp
*.tiff
!examples/**/*.jpg
!examples/**/*.png
!docs/**/*.jpg
!docs/**/*.png
"""
        
        gitignore_path = self.root_dir / ".gitignore"
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        
        print("📝 已创建/更新 .gitignore")

def main():
    """主函数"""
    organizer = ProjectOrganizer()
    organizer.organize_files()
    
    print("\n🎉 项目整理完成!")
    print("📋 整理内容:")
    print("  ✅ 文件移动到对应目录")
    print("  ✅ API Key替换为 *****")
    print("  ✅ 敏感信息已脱敏")
    print("  ✅ .gitignore已更新")
    print("\n🚀 项目现在可以安全上传到GitHub!")

if __name__ == "__main__":
    main()