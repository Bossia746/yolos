#!/bin/bash

# YOLOS ROS安装脚本
# 支持ROS1 Noetic和ROS2 Humble

echo "YOLOS ROS安装脚本"
echo "=================="

# 检测系统版本
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    echo "无法检测系统版本"
    exit 1
fi

echo "检测到系统: $OS $VER"

# 选择ROS版本
echo "请选择要安装的ROS版本:"
echo "1) ROS1 Noetic (Ubuntu 20.04)"
echo "2) ROS2 Humble (Ubuntu 22.04)"
echo "3) 自动检测"
read -p "请输入选择 (1-3): " choice

case $choice in
    1)
        ROS_VERSION="noetic"
        ROS_TYPE="ros1"
        ;;
    2)
        ROS_VERSION="humble"
        ROS_TYPE="ros2"
        ;;
    3)
        if [[ "$VER" == "20.04" ]]; then
            ROS_VERSION="noetic"
            ROS_TYPE="ros1"
        elif [[ "$VER" == "22.04" ]]; then
            ROS_VERSION="humble"
            ROS_TYPE="ros2"
        else
            echo "不支持的系统版本: $VER"
            exit 1
        fi
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo "将安装 ROS $ROS_VERSION ($ROS_TYPE)"

# 安装ROS1 Noetic
install_ros1_noetic() {
    echo "安装ROS1 Noetic..."
    
    # 设置sources.list
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    
    # 设置密钥
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    
    # 更新包列表
    sudo apt update
    
    # 安装ROS
    sudo apt install -y ros-noetic-desktop-full
    
    # 初始化rosdep
    sudo rosdep init
    rosdep update
    
    # 设置环境
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
    source ~/.bashrc
    
    # 安装构建工具
    sudo apt install -y python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
    
    # 安装额外包
    sudo apt install -y \
        ros-noetic-cv-bridge \
        ros-noetic-image-transport \
        ros-noetic-sensor-msgs \
        ros-noetic-geometry-msgs \
        ros-noetic-std-msgs \
        ros-noetic-message-generation \
        ros-noetic-message-runtime
}

# 安装ROS2 Humble
install_ros2_humble() {
    echo "安装ROS2 Humble..."
    
    # 设置locale
    sudo apt update && sudo apt install -y locales
    sudo locale-gen en_US en_US.UTF-8
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LANG=en_US.UTF-8
    
    # 设置sources
    sudo apt install -y software-properties-common
    sudo add-apt-repository universe
    
    sudo apt update && sudo apt install -y curl gnupg lsb-release
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    
    # 更新并安装ROS2
    sudo apt update
    sudo apt upgrade -y
    sudo apt install -y ros-humble-desktop
    
    # 安装开发工具
    sudo apt install -y \
        python3-flake8-docstrings \
        python3-pip \
        python3-pytest-cov \
        ros-dev-tools
    
    # 安装colcon
    sudo apt install -y python3-colcon-common-extensions
    
    # 设置环境
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    source ~/.bashrc
    
    # 安装额外包
    sudo apt install -y \
        ros-humble-cv-bridge \
        ros-humble-image-transport \
        ros-humble-sensor-msgs \
        ros-humble-geometry-msgs \
        ros-humble-std-msgs
}

# 执行安装
if [[ "$ROS_TYPE" == "ros1" ]]; then
    install_ros1_noetic
else
    install_ros2_humble
fi

# 创建工作空间
echo "创建ROS工作空间..."
if [[ "$ROS_TYPE" == "ros1" ]]; then
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws
    catkin_make
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
else
    mkdir -p ~/ros2_ws/src
    cd ~/ros2_ws
    colcon build
    echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
fi

# 复制YOLOS ROS包
echo "设置YOLOS ROS包..."
if [[ "$ROS_TYPE" == "ros1" ]]; then
    cp -r ros_workspace/src/yolos_ros ~/catkin_ws/src/
    cd ~/catkin_ws
    catkin_make
else
    cp -r ros_workspace/src/yolos_ros ~/ros2_ws/src/
    cd ~/ros2_ws
    colcon build --packages-select yolos_ros
fi

# 创建启动脚本
echo "创建启动脚本..."
if [[ "$ROS_TYPE" == "ros1" ]]; then
    cat > start_yolos_ros.sh << 'EOF'
#!/bin/bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch yolos_ros detection.launch
EOF
else
    cat > start_yolos_ros.sh << 'EOF'
#!/bin/bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch yolos_ros detection.launch.py
EOF
fi

chmod +x start_yolos_ros.sh

# 创建测试脚本
echo "创建测试脚本..."
if [[ "$ROS_TYPE" == "ros1" ]]; then
    cat > test_yolos_ros.sh << 'EOF'
#!/bin/bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

echo "启动roscore..."
roscore &
sleep 3

echo "启动YOLOS检测节点..."
rosrun yolos_ros detection_node.py &

echo "发布测试图像..."
rosrun yolos_ros image_publisher.py

echo "测试完成"
killall roscore
EOF
else
    cat > test_yolos_ros.sh << 'EOF'
#!/bin/bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

echo "启动YOLOS检测节点..."
ros2 run yolos_ros detection_node.py &

echo "发布测试图像..."
ros2 run yolos_ros image_publisher.py

echo "测试完成"
EOF
fi

chmod +x test_yolos_ros.sh

echo ""
echo "ROS安装完成!"
echo "=============="
echo "ROS版本: $ROS_VERSION ($ROS_TYPE)"
if [[ "$ROS_TYPE" == "ros1" ]]; then
    echo "工作空间: ~/catkin_ws"
    echo "启动命令: ./start_yolos_ros.sh"
    echo "测试命令: ./test_yolos_ros.sh"
else
    echo "工作空间: ~/ros2_ws"
    echo "启动命令: ./start_yolos_ros.sh"
    echo "测试命令: ./test_yolos_ros.sh"
fi
echo ""
echo "请重新打开终端或运行 'source ~/.bashrc' 来加载环境变量"