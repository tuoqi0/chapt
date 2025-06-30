#!/usr/bin/env python3
#狗头
import rospy
import os
import yaml
import math
import numpy as np
from scipy.interpolate import splprep, splev
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
from visualization_msgs.msg import Marker
import tf.transformations as tf_trans
from std_msgs.msg import Float32MultiArray
from fast_lio_sam.srv import AddPoint, AddPointResponse  # 自定义服务

class PathManager:
    def __init__(self):
        rospy.init_node('path_manager')

        # 运行模式参数
        self.mode = rospy.get_param("~mode", "click")  # click或record
        self.auto_load = rospy.get_param("~auto_load", False)
        self.min_distance = rospy.get_param("~min_distance", 0.5)  # 最小记录距离
        self.smooth_factor = rospy.get_param("~smooth_factor", 10.0)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 1.5)  # 前视距离
        self.goal_threshold = rospy.get_param("~goal_threshold", 0.10)  # 目标点切换阈值
        
        # 路径数据结构
        self.path = Path()
        self.path.header.frame_id = "map"
        self.smoothed_path = Path()
        self.smoothed_path.header.frame_id = "map"
        
        # 发布器
        self.raw_path_pub = rospy.Publisher('/raw_path', Path, queue_size=10, latch=True)
        self.smooth_path_pub = rospy.Publisher('/smooth_path', Path, queue_size=10, latch=True)
        self.marker_pub = rospy.Publisher('/path_markers', Marker, queue_size=10)
        self.goal_info_pub = rospy.Publisher('/goal_info', Float32MultiArray, queue_size=10)
        
        # 订阅器
        rospy.Subscriber('/clicked_point', PointStamped, self.point_callback)
        rospy.Subscriber('/Odometry', Odometry, self.odom_callback)
        rospy.Subscriber('/distance', Float32MultiArray, self.distance_callback)  # 订阅距离信息
        # 服务 - 路径管理
        self.save_service = rospy.Service('/save_path', Trigger, self.handle_save_request)
        self.load_service = rospy.Service('/load_path', Trigger, self.handle_load_request)
        self.smooth_service = rospy.Service('/smooth_path', Trigger, self.handle_smooth_request)
        
        # 服务 - 手动设点
        self.add_point_service = rospy.Service('/add_point', AddPoint, self.handle_add_point)
        self.toggle_mode_service = rospy.Service('/toggle_mode', SetBool, self.handle_toggle_mode)
        
        # 保存路径目录
        self.save_dir = os.path.join(os.path.expanduser("/home/toe/"), "saved_paths")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 状态变量
        self.robot_position = None
        self.robot_yaw = 0.0
        self.goal_position = None
        self.last_recorded_point = None
        self.is_recording = False  # 记录模式状态
        self.current_region = None  # 当前所在的区域
        self.current_region_index = 0  # 当前区域索引
        self.current_goal_index = 0  # 当前目标点索引
        self.front_distance = 0.0  # 前方障碍物距离
        # 定义区域边界（x轴范围）
        self.region_boundaries = {
            "竖杆":  (0.0, 7.0),
            "斜坡":  (10, 13.5),
            "断桥":  (16.0, 19.8),
            "高墙":  (22.0, 24.1),
            "沙坑":  (27.1, 30.1),
            "匍匐架":(33.0, 36.1)
        }

        # 区域索引映射
        self.region_index_map = {
            "竖杆": 1,
            "斜坡": 2,
            "断桥": 3,
            "高墙": 4,
            "沙坑": 5,
            "匍匐架": 6
        }
        
        # 自动加载路径
        if self.auto_load:
            self.load_path_from_file()
            
        rospy.loginfo(f"2D雷达路径管理器已启动 (模式: {self.mode})")
        rospy.loginfo(f"最小记录距离: {self.min_distance}m, 前视距离: {self.lookahead_distance}m")
        
        if self.mode == "record":
            self.is_recording = True
            rospy.loginfo("记录模式已启用，使用 /save_path 服务保存记录的路径")

    def angle_diff(self, a, b):
        """计算两个角度之间的差值（考虑圆周）"""
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    def add_point_to_path(self, x, y, frame_id="map"):
        """添加点到路径"""
        # 如果是第一个点，设置为起点
        if len(self.path.poses) == 0 and self.robot_position:
            start_pose = PoseStamped()
            start_pose.header.frame_id = frame_id
            start_pose.pose.position.x = self.robot_position[1]
            start_pose.pose.position.y = 0.0
            start_pose.pose.position.z = self.robot_position[0]
            start_pose.pose.orientation.w = 1.0
            self.path.poses.append(start_pose)
            rospy.loginfo(f"已设定路径起点为当前位置: x={self.robot_position[0]:.2f}, y={self.robot_position[1]:.2f}")
            self.current_goal_index = 0 
        
        # 添加目标点
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = frame_id
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.pose.position.x = y
        goal_pose.pose.position.z = x
        goal_pose.pose.position.y = 0.0
        goal_pose.pose.orientation.w = 1.0
        self.path.poses.append(goal_pose)
        self.path.header.stamp = rospy.Time.now()
        self.goal_position = (x, y)  # 更新最终目标点
        self.raw_path_pub.publish(self.path)

        rospy.loginfo(f"添加目标点: x={x:.2f}, y={y:.2f}")
        self.visualize_path()
   
    def point_callback(self, msg):
        """处理点击点消息"""
        if self.mode != "click":
            return
            
        if msg.header.frame_id != "map":
            rospy.logwarn(f"点坐标系 {msg.header.frame_id} 不是'map'，已忽略")
            return

        if not self.robot_position and len(self.path.poses) == 0:
            rospy.logwarn("未收到机器人当前位置，无法设定起点")
            return

        self.add_point_to_path(msg.point.x, msg.point.y, msg.header.frame_id)

    def check_region(self, x):
        """检查机器人所在的区域(基于x坐标)"""
        for region_name, (min_x, max_x) in self.region_boundaries.items():
            if min_x <= x <= max_x:
                return region_name
        return None
    def get_front_boundary_distance(self, x):
        """获取前方区域边界距离"""
        for region_name, (min_x, max_x) in self.region_boundaries.items():
            if min_x <= x <= max_x:
                if x < max_x:
                    return max_x - x
    def distance_callback(self, msg):
        # 输出前方障碍物距离消息
        if msg.data:
            self.front_distance = msg.data[0]
            rospy.loginfo(f"前方障碍物距离: {self.front_distance}m")
            
    def odom_callback(self, msg):
        """处理里程计消息"""
        # 更新机器人位置
        raw_x = msg.pose.pose.position.z
        raw_y = msg.pose.pose.position.x
        transformed_x = raw_x
        transformed_y = raw_y
        self.robot_position = (transformed_x, transformed_y)        
        
        # 更新偏航角
        quat = (
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.w
        )
        _, _, self.robot_yaw = tf_trans.euler_from_quaternion(quat)
        
        # 记录模式：自动记录路径点
        if self.is_recording:
            current_pos = (transformed_x, transformed_y)
            
            # 检查是否是新点（距离上一个点足够远）
            if self.last_recorded_point:
                dx = current_pos[0] - self.last_recorded_point[0]
                dy = current_pos[1] - self.last_recorded_point[1]
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < self.min_distance:
                    return  # 距离太小，不记录
            
            # 记录新点
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = current_pos[1]  # 注意：x和y的顺序
            pose.pose.position.z= current_pos[0]
            pose.pose.position.y = 0.0
            pose.pose.orientation.w = 1.0
            self.path.poses.append(pose)
            self.last_recorded_point = current_pos
            
            # 发布原始路径
            self.path.header.stamp = rospy.Time.now()
            self.raw_path_pub.publish(self.path)
            
            # 更新目标点
            self.goal_position = current_pos
            rospy.loginfo(f"记录路径点: x={current_pos[0]:.2f}, y={current_pos[1]:.2f}")
            self.visualize_path()
        
      
        # 导航信息处理
        if self.path.poses:
            # 确保当前目标索引有效
            if self.current_goal_index >= len(self.path.poses):
                self.current_goal_index = len(self.path.poses) - 1
                rospy.logwarn(f"目标索引超出范围，重置为最后一点: {self.current_goal_index}")
              # 检查当前区域
            self.current_region = self.check_region(self.robot_position[0])
            self.current_region_index=self.region_index_map.get(self.current_region, 0)
        # 获取前方区域边界距离
            front_boundary_distance = self.get_front_boundary_distance(self.robot_position[0])
            if  front_boundary_distance is None:
                front_boundary_distance = 0.0
            # 获取当前目标点位置
            current_goal_pose = self.path.poses[self.current_goal_index]
            goal_pos = current_goal_pose.pose.position
            goal_position = (goal_pos.x, goal_pos.y)
            
            # 计算到当前目标点的距离
            goal_dx = goal_position[0] - self.robot_position[0]
            goal_dy = self.robot_position[1] - goal_position[1]
            distance_to_goal = math.hypot(goal_dx, goal_dy)
            
            # 如果接近当前目标点，切换到下一个目标点
            if self.current_region_index != 3:
                if goal_dx < 0.1 and abs(goal_dy) < 0.1:
                    if self.current_goal_index < len(self.path.poses) - 1:
                        self.current_goal_index += 1
                        new_goal = self.path.poses[self.current_goal_index].pose.position
                        rospy.loginfo(f"切换到下一个目标点: {self.current_goal_index}/{len(self.path.poses)-1} "
                                  f"({new_goal.x:.2f}, {new_goal.y:.2f})")
                    else:
                        rospy.loginfo("已到达最终目标点!")
            else: 
                if goal_dx < 0.05 and abs(goal_dy) < 0.05:
                    if self.current_goal_index < len(self.path.poses) - 1:
                        self.current_goal_index += 1
                        new_goal = self.path.poses[self.current_goal_index].pose.position
                        rospy.loginfo(f"切换到下一个目标点: {self.current_goal_index}/{len(self.path.poses)-1} "
                                  f"({new_goal.x:.2f}, {new_goal.y:.2f})")
                    else:
                        rospy.loginfo("已到达最终目标点!")
            # 获取当前目标点（可能已更新）
            current_goal_pose = self.path.poses[self.current_goal_index]
            goal_pos = current_goal_pose.pose.position
            self.goal_position = (goal_pos.x, goal_pos.y)
            
            # 重新计算到当前目标点的向量
            goal_dx = self.goal_position[0] - self.robot_position[0]
            goal_dy = self.robot_position[1] - self.goal_position[1]
            distance_to_goal = math.hypot(goal_dx, goal_dy)
            if self.current_goal_index > 0:
                # 正常情况：使用上一个目标点和当前目标点形成的路径段
                prev_goal_pose = self.path.poses[self.current_goal_index - 1]
                A = (prev_goal_pose.pose.position.x, prev_goal_pose.pose.position.y)
                B = self.goal_position
            else:
                # 当处于第一个目标点时：使用原点和当前目标点形成的路径段
                A = (0, 0)  # 原点
                B = self.goal_position
            # 计算路径向量 (从A点到B点)
            path_vector = (B[0] - A[0], B[1] - A[1])
            # 计算从A点到机器人位置的向量
            robot_vector = (self.robot_position[0] - A[0], self.robot_position[1] - A[1])
        
            # 计算路径向量的模长
            path_length = math.hypot(path_vector[0], path_vector[1])
            
            if path_length > 1e-6:  # 避免除以零
                
                # 使用叉积计算横向偏移 (带符号)
                if self.current_region_index != 6:  # 匍匐架区域特殊处理
                    lateral_distance = -goal_dy  # 竖杆区域，直接使用y坐标差
                else:
                    cross_product = path_vector[0] * robot_vector[1] - path_vector[1] * robot_vector[0]
                    lateral_distance = cross_product / path_length
            else:
                lateral_distance = 0.0

            goal_heading = math.atan2(goal_dy, goal_dx)
            
            # 计算航向偏差（机器人当前朝向与目标方向的角度差）  
            if self.current_region_index != 6:  # 匍匐架区域特殊处理

                goal_heading_error = self.angle_diff(self.robot_yaw, 0.0)
                heading_error_deg = math.degrees(goal_heading_error)
                if heading_error_deg < -180:
                    heading_error_deg += 360
            else:    
                goal_heading_error = self.angle_diff(goal_heading, self.robot_yaw)
                heading_error_deg = math.degrees(goal_heading_error)
                if heading_error_deg < -180:
                    heading_error_deg += 360

            rospy.loginfo_throttle(0.1,
                f"当前位置: ({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f}) | "
                f"目标点: ({goal_pos.x:.2f}, {goal_pos.y:.2f}) | "    
                f"目标点 {self.current_goal_index}/{len(self.path.poses)-1} → "
                f"距离={goal_dx:.2f}m | 横向偏移={lateral_distance:.2f}m | 航向偏差={heading_error_deg:.1f}°"
                f" | 前方边界距离={front_boundary_distance:.2f}m | "
                f"当前区域: {self.current_region} (索引: {self.current_region_index}) | "
                )

            # 发布目标点信息
            data = [
                float(goal_dx),           # 到当前目标点的y方向距离
                float(lateral_distance),   # 横向偏移距离
                float(heading_error_deg),  # 航向偏差
                float(front_boundary_distance),  # 到区域前方y轴边界距离
                float(self.current_region_index),  # 区域索
            ]
            
            # 创建并发布消息
            array_msg = Float32MultiArray(data=data)
            self.goal_info_pub.publish(array_msg)

    def smooth_path(self):
        """平滑路径"""
        if len(self.path.poses) < 3:
            rospy.logwarn("路径点不足，无法平滑")
            return False
            
        try:
            # 提取路径点
            points = []
            for pose in self.path.poses:
                points.append([pose.pose.position.x, pose.pose.position.y])
            
            # 转换为NumPy数组
            points = np.array(points)
            
            # 样条曲线平滑
            tck, u = splprep(points.T, u=None, s=self.smooth_factor, per=0) 
            u_new = np.linspace(u.min(), u.max(), 100)
            x_new, y_new = splev(u_new, tck, der=0)
            
            # 创建平滑路径
            self.smoothed_path = Path()
            self.smoothed_path.header.frame_id = "map"
            self.smoothed_path.header.stamp = rospy.Time.now()
            
            for i in range(len(x_new)):
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = x_new[i]
                pose.pose.position.z = y_new[i]
                pose.pose.position.y = 0.0
                pose.pose.orientation.w = 1.0
                self.smoothed_path.poses.append(pose)
            
            # 发布平滑路径
            self.smooth_path_pub.publish(self.smoothed_path)
            rospy.loginfo(f"路径已平滑，原始点: {len(self.path.poses)} → 平滑点: {len(self.smoothed_path.poses)}")
            self.visualize_path()
            return True
        except Exception as e:
            rospy.logerr(f"平滑路径失败: {str(e)}")
            return False

    def visualize_path(self):
        """可视化原始路径和平滑路径"""
        # 可视化原始路径
        if len(self.path.poses) >= 2:
            marker_raw = Marker()
            marker_raw.header.frame_id = "map"
            marker_raw.header.stamp = rospy.Time.now()
            marker_raw.ns = "raw_path"
            marker_raw.id = 0
            marker_raw.type = Marker.LINE_STRIP
            marker_raw.action = Marker.ADD
            marker_raw.scale.x = 0.1
            marker_raw.color.r = 1.0
            marker_raw.color.g = 0.5
            marker_raw.color.a = 0.7
            marker_raw.pose.orientation.w = 1.0
            
            # 按顺序添加点
            for pose in self.path.poses:
                point = Point()
                point.x = pose.pose.position.x
                point.y = pose.pose.position.y
                point.z = 0.0
                marker_raw.points.append(point)
                
            self.marker_pub.publish(marker_raw)
        
        # 可视化平滑路径
        if len(self.smoothed_path.poses) >= 2:
            marker_smooth = Marker()
            marker_smooth.header.frame_id = "map"
            marker_smooth.header.stamp = rospy.Time.now()
            marker_smooth.ns = "smooth_path"
            marker_smooth.id = 1  # 使用不同的ID
            marker_smooth.type = Marker.LINE_STRIP
            marker_smooth.action = Marker.ADD
            marker_smooth.scale.x = 0.15
            marker_smooth.color.g = 1.0
            marker_smooth.color.r = 0.5
            marker_smooth.color.a = 1.0
            marker_smooth.pose.orientation.w = 1.0
            
            # 按顺序添加点
            for pose in self.smoothed_path.poses:
                point = Point()
                point.x = pose.pose.position.x
                point.y = pose.pose.position.y
                point.z = 0.0
                marker_smooth.points.append(point)
                
            self.marker_pub.publish(marker_smooth)

    def save_path_to_file(self, filename="path.yaml"):
        """保存路径到文件"""
        try:
            filepath = os.path.join(self.save_dir, filename)
          
        # 将元组转换为列表以便YAML序列化
            converted_boundaries = {}
            if self.region_boundaries:
               for k, v in self.region_boundaries.items():
                  converted_boundaries[k] = list(v)  # 元组转列表
        
            path_data = {
                "frame_id": "map",
                "points": [{"x": p.pose.position.x, "y": p.pose.position.y} for p in self.path.poses],
                "smoothed_points": [{"x": p.pose.position.x, "y": p.pose.position.y} for p in self.smoothed_path.poses],
                "lookahead_distance": self.lookahead_distance,
                "min_distance": self.min_distance,
                "smooth_factor": self.smooth_factor,
                "goal_threshold": self.goal_threshold,
                "region_boundaries": converted_boundaries,  # 使用转换后的字典
                "region_index_map": self.region_index_map
        }
        
            with open(filepath, 'w') as f:
                yaml.dump(path_data, f)
        
            rospy.loginfo(f"路径已保存到: {filepath}")
            return filepath
        except Exception as e:
            rospy.logerr(f"保存路径失败: {str(e)}")
            return None
    def load_path_from_file(self, filename="path.yaml"):
        """从文件加载路径"""
        try:
            filepath = os.path.join(self.save_dir, filename)
            if not os.path.exists(filepath):
                rospy.logerr(f"路径文件不存在: {filepath}")
                return False 
            
            with open(filepath, 'r') as f:
                path_data = yaml.safe_load(f)
        
        # 重置路径
            self.path = Path()
            self.path.header.frame_id = "map"
            self.smoothed_path = Path()
            self.smoothed_path.header.frame_id = "map"
        
        # 加载原始路径点
            for point in path_data["points"]:
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.pose.position.x = point["x"]
                pose.pose.position.y = point["y"]
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                self.path.poses.append(pose)
        
        # 加载平滑路径点
            if "smoothed_points" in path_data:
                for point in path_data["smoothed_points"]:
                    pose = PoseStamped()
                    pose.header.frame_id = "map"
                    pose.pose.position.x = point["x"]
                    pose.pose.position.y = point["y"]
                    pose.pose.position.z = 0.0
                    pose.pose.orientation.w = 1.0
                    self.smoothed_path.poses.append(pose)
        
        # 加载参数
            if "lookahead_distance" in path_data:
                self.lookahead_distance = path_data["lookahead_distance"]
            if "min_distance" in path_data:
                self.min_distance = path_data["min_distance"]
            if "smooth_factor" in path_data:
                self.smooth_factor = path_data["smooth_factor"]
            if "goal_threshold" in path_data:
                self.goal_threshold = path_data["goal_threshold"]
            
        # 处理区域边界（列表转元组）
            if "region_boundaries" in path_data:
                converted_boundaries = {}
                for region_name, bounds in path_data["region_boundaries"].items():
                    converted_boundaries[region_name] = tuple(bounds)  # 列表转元组
                self.region_boundaries = converted_boundaries
            
            if "region_index_map" in path_data:
                self.region_index_map = path_data["region_index_map"]
        
        # 重置目标点索引
            self.current_goal_index = 0
        
        # 设置最终目标点
            if self.path.poses:
                last_point = self.path.poses[-1].pose.position
                self.goal_position = (last_point.x, last_point.y)
                rospy.loginfo(f"设置目标点为: ({last_point.x:.2f}, {last_point.y:.2f})")
        
        # 发布路径
            self.path.header.stamp = rospy.Time.now()
            self.raw_path_pub.publish(self.path)
        
            if self.smoothed_path.poses:
                self.smoothed_path.header.stamp = rospy.Time.now()
                self.smooth_path_pub.publish(self.smoothed_path)
        
            rospy.loginfo(f"从 {filepath} 加载了 {len(self.path.poses)} 个原始点和 {len(self.smoothed_path.poses)} 个平滑点")
            self.visualize_path()
            return True
        except Exception as e:
            rospy.logerr(f"加载路径失败: {str(e)}")
            return False
    

    def handle_save_request(self, req):
        try:
            if self.save_path_to_file():
                return TriggerResponse(True, "路径保存成功")
            return TriggerResponse(False, "路径保存失败")
        except Exception as e:
            return TriggerResponse(False, f"保存失败: {str(e)}")

    def handle_load_request(self, req):
        try:
            if self.load_path_from_file():
                return TriggerResponse(True, "路径加载成功")
            return TriggerResponse(False, "路径加载失败")
        except Exception as e:
            return TriggerResponse(False, f"加载失败: {str(e)}")

    def handle_smooth_request(self, req):
        try:
            if self.smooth_path():
                return TriggerResponse(True, "路径平滑成功")
            return TriggerResponse(False, "路径平滑失败")
        except Exception as e:
            return TriggerResponse(False, f"平滑失败: {str(e)}")
            
    def handle_add_point(self, req):
        """处理手动添加点请求"""
        try:
            self.add_point_to_path(req.x, req.y)
            return AddPointResponse(True, f"点 ({req.x:.2f}, {req.y:.2f}) 添加成功")
        except Exception as e:
            return AddPointResponse(False, f"添加点失败: {str(e)}")
            
    def handle_toggle_mode(self, req):
        """切换记录模式状态"""
        if req.data:
            if not self.is_recording:
                self.is_recording = True
                self.last_recorded_point = None  # 重置最后记录点
                rospy.loginfo("记录模式已启用")
                return SetBoolResponse(True, "记录模式已启用")
            else:
                rospy.logwarn("记录模式已处于启用状态")
                return SetBoolResponse(False, "记录模式已处于启用状态")
        else:
            if self.is_recording:
                self.is_recording = False
                rospy.loginfo("记录模式已禁用")
                return SetBoolResponse(True, "记录模式已禁用")
            else:
                rospy.logwarn("记录模式已处于禁用状态")
                return SetBoolResponse(False, "记录模式已处于禁用状态")

if __name__ == '__main__':
    manager = PathManager()
    rospy.spin()