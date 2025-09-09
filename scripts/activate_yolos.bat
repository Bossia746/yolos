@echo off
echo 激活YOLOS虚拟环境...
call yolos_env\Scripts\activate.bat
echo 虚拟环境已激活！
echo 运行 'python self_learning_demo_gui.py' 启动GUI演示
echo 运行 'python test_self_learning_system.py' 进行系统测试
cmd /k
