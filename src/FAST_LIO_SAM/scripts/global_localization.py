#!/usr/bin/env python3
# coding=utf8
import rospy
import numpy as np
import threading
import tf
import tf2_ros
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from std_msgs.msg import Bool
import sensor_msgs.point_cloud2 as pc2

# Global variables
global_map = None
initial_pose = None
localization_triggered = False
T_map_to_camera = None

def pointcloud2_to_xyz_array(msg):
    return np.array([[p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])

def transform_points(points, transform: TransformStamped):
    trans = transform.transform.translation
    rot = transform.transform.rotation
    T = tf.transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
    T[0, 3] = trans.x
    T[1, 3] = trans.y
    T[2, 3] = trans.z
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed = (T @ points_hom.T).T[:, :3]
    return transformed

def global_localization(T_init):
    global T_map_to_camera

    if not isinstance(T_init, TransformStamped):
        rospy.logwarn("[GlobalLocalization] Initial pose is not a valid TransformStamped.")
        return

    try:
        T_base_to_map = tf.transformations.quaternion_matrix([
            T_init.transform.rotation.x,
            T_init.transform.rotation.y,
            T_init.transform.rotation.z,
            T_init.transform.rotation.w
        ])
        T_base_to_map[0, 3] = T_init.transform.translation.x
        T_base_to_map[1, 3] = T_init.transform.translation.y
        T_base_to_map[2, 3] = T_init.transform.translation.z

        T_map_to_camera = TransformStamped()
        T_map_to_camera.header.stamp = rospy.Time.now()
        T_map_to_camera.header.frame_id = "map"
        T_map_to_camera.child_frame_id = "camera_init"

        trans = tf.transformations.translation_from_matrix(T_base_to_map)
        quat = tf.transformations.quaternion_from_matrix(T_base_to_map)

        T_map_to_camera.transform.translation.x = trans[0]
        T_map_to_camera.transform.translation.y = trans[1]
        T_map_to_camera.transform.translation.z = trans[2]
        T_map_to_camera.transform.rotation.x = quat[0]
        T_map_to_camera.transform.rotation.y = quat[1]
        T_map_to_camera.transform.rotation.z = quat[2]
        T_map_to_camera.transform.rotation.w = quat[3]

        rospy.loginfo("[GlobalLocalization] Updated map -> camera_init transform")

    except Exception as e:
        rospy.logerr(f"[GlobalLocalization] Failed to compute map->camera_init: {e}")

def load_global_map(msg: PointCloud2):
    global global_map
    rospy.loginfo("[GlobalLocalization] Received global map...")
    global_map = pointcloud2_to_xyz_array(msg)
    rospy.loginfo(f"[GlobalLocalization] Global map loaded with {global_map.shape[0]} points.")

def localization_trigger_cb(msg: Bool):
    global localization_triggered
    localization_triggered = msg.data
    rospy.loginfo(f"[GlobalLocalization] Localization triggered: {localization_triggered}")

def thread_localization():
    global localization_triggered, initial_pose

    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        if localization_triggered and global_map is not None and initial_pose is not None:
            rospy.loginfo("[GlobalLocalization] Triggered, performing localization...")
            global_localization(initial_pose)
            localization_triggered = False
        rate.sleep()

def publish_tf():
    global T_map_to_camera
    br = tf2_ros.TransformBroadcaster()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if T_map_to_camera is not None:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = rospy.Time.now()
            tf_msg.header.frame_id = "map"
            tf_msg.child_frame_id = "camera_init"
            tf_msg.transform = T_map_to_camera.transform
            br.sendTransform(tf_msg)
        rate.sleep()

def initial_pose_cb(msg: PoseWithCovarianceStamped):
    global initial_pose, localization_triggered

    rospy.loginfo("[GlobalLocalization] Received /initialpose from RViz")

    T = TransformStamped()
    T.header = msg.header
    T.child_frame_id = "camera_init"
    T.transform.translation.x = msg.pose.pose.position.x
    T.transform.translation.y = msg.pose.pose.position.y
    T.transform.translation.z = msg.pose.pose.position.z
    T.transform.rotation = msg.pose.pose.orientation

    initial_pose = T
    localization_triggered = True

def main():
    rospy.init_node("global_localization_node")

    rospy.Subscriber("/global_map", PointCloud2, load_global_map)
    rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, initial_pose_cb)
    rospy.Subscriber("/localization_trigger", Bool, localization_trigger_cb)

    threading.Thread(target=thread_localization).start()
    threading.Thread(target=publish_tf).start()

    rospy.spin()

if __name__ == "__main__":
    main()
