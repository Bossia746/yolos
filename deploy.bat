@echo off
REM YOLOS 部署启动脚本 (Windows)
REM 用法: deploy.bat <environment> <version> [options]

setlocal enabledelayedexpansion

REM 设置脚本目录
set SCRIPT_DIR=%~dp0
set DEPLOY_SCRIPT=%SCRIPT_DIR%deployment\scripts\deploy.py

REM 检查Python是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Python未安装或不在PATH中
    exit /b 1
)

REM 检查部署脚本是否存在
if not exist "%DEPLOY_SCRIPT%" (
    echo 错误: 部署脚本不存在: %DEPLOY_SCRIPT%
    exit /b 1
)

REM 设置环境变量
set PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%

REM 如果没有参数，显示帮助信息
if "%1"=="" (
    echo YOLOS 自动化部署工具
    echo.
    echo 用法:
    echo   deploy.bat ^<environment^> ^<version^> [options]
    echo.
    echo 环境:
    echo   development  - 开发环境
    echo   staging      - 预发布环境
    echo   production   - 生产环境
    echo.
    echo 选项:
    echo   --force      - 强制部署（忽略检查）
    echo   --verbose    - 详细输出
    echo   --config     - 指定配置文件路径
    echo.
    echo 示例:
    echo   deploy.bat development v1.0.0
    echo   deploy.bat production v1.0.0 --force
    echo   deploy.bat staging v1.0.0 --verbose
    echo.
    echo 其他操作:
    echo   deploy.bat list development     - 查看部署历史
    echo   deploy.bat status production   - 查看环境状态
    exit /b 0
)

REM 特殊操作处理
if "%1"=="list" (
    python "%DEPLOY_SCRIPT%" list --environment %2 %3 %4 %5 %6 %7 %8 %9
    exit /b !errorlevel!
)

if "%1"=="status" (
    python "%DEPLOY_SCRIPT%" status --environment %2 %3 %4 %5 %6 %7 %8 %9
    exit /b !errorlevel!
)

REM 执行部署
echo 开始部署 YOLOS %2 到 %1 环境...
echo.

python "%DEPLOY_SCRIPT%" deploy --environment %1 --version %2 %3 %4 %5 %6 %7 %8 %9

set DEPLOY_RESULT=!errorlevel!

if !DEPLOY_RESULT! equ 0 (
    echo.
    echo ✅ 部署成功完成！
) else (
    echo.
    echo ❌ 部署失败，请检查日志
)

exit /b !DEPLOY_RESULT!