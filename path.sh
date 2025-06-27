#!/bin/bash

# 设置环境变量
source ~/chatp4/chapt_ws/devel/setup.bash

# 定义清理函数
cleanup() {
    echo "终止所有ROS节点并保存路径..."
    
    # 在终止节点前调用保存路径服务
    rosservice call /save_path 2>/dev/null
    
    # 终止节点
    kill $LIVOX_PID $MAPPING_PID 2>/dev/null
    
    # 等待节点完全退出
    sleep 1
    
    echo "系统已安全关闭"
    exit 0
}

# 捕获退出信号
trap cleanup SIGINT SIGTERM

# 启动livox驱动节点
echo "启动 Livox ROS 驱动..."
roslaunch livox_ros_driver2 msg_MID360.launch &
LIVOX_PID=$!

# 等待节点初始化
sleep 2
echo "Livox 驱动已启动 (PID: $LIVOX_PID)"

# 检查驱动节点是否正常运行
if ! ps -p $LIVOX_PID > /dev/null; then
    echo "错误: Livox 驱动启动失败!"
    exit 1
fi

# 启动SLAM节点
echo "启动 Fast-LIO-SAM 建图..."
roslaunch fast_lio_sam mid360.launch &
MAPPING_PID=$!
echo "建图节点已启动 (PID: $MAPPING_PID)"

# 保持脚本运行并显示状态
echo ""
echo "系统已启动:"
echo "1. Livox 驱动 [PID $LIVOX_PID]"
echo "2. Fast-LIO-SAM 建图 [PID $MAPPING_PID]"
echo ""
echo "按 Ctrl+C 停止所有节点并保存路径"

# 等待子进程退出
wait
