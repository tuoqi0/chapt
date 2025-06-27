#!/usr/bin/env python3
# coding=utf-8

from __future__ import print_function, division, absolute_import

import copy
import threading
import rospy
import tf
import tf.transformations
import numpy as np

from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Odometry

cur_odom_to_camera = None
cur_map_to_odom = None
FREQ_PUB_LOCALIZATION = 50

def pose_to_mat(pose_msg):
    position = pose_msg.pose.pose.position
    orientation = pose_msg.pose.pose.orientation

    translation = tf.transformations.translation_matrix([position.x, position.y, position.z])
    rotation = tf.transformations.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])

    return np.dot(translation, rotation)

def transform_fusion():
    global cur_odom_to_camera, cur_map_to_odom

    br = tf.TransformBroadcaster()
    rate = rospy.Rate(FREQ_PUB_LOCALIZATION)

    while not rospy.is_shutdown():
        cur_odom = copy.deepcopy(cur_odom_to_camera)
        T_map_to_odom = np.eye(4)

        if cur_map_to_odom is not None:
            T_map_to_odom = pose_to_mat(cur_map_to_odom)

            #  发布 map → odom
            br.sendTransform(tf.transformations.translation_from_matrix(T_map_to_odom),
                             tf.transformations.quaternion_from_matrix(T_map_to_odom),
                             rospy.Time.now(),
                             'odom', 'map')

        #  继续发布 map → camera_init
        if cur_odom is not None:
            T_odom_to_camera = pose_to_mat(cur_odom)
            T_map_to_camera = np.dot(T_map_to_odom, T_odom_to_camera)

            xyz = tf.transformations.translation_from_matrix(T_map_to_camera)
            quat = tf.transformations.quaternion_from_matrix(T_map_to_camera)

            localization = Odometry()
            localization.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
            localization.twist = cur_odom.twist
            localization.header.stamp = cur_odom.header.stamp
            localization.header.frame_id = 'map'
            localization.child_frame_id = 'camera_init'

            pub_localization.publish(localization)

        rate.sleep()

def cb_save_cur_odom(odom_msg):
    global cur_odom_to_camera
    cur_odom_to_camera = odom_msg

def cb_save_map_to_odom(odom_msg):
    global cur_map_to_odom
    cur_map_to_odom = odom_msg

if __name__ == '__main__':
    rospy.init_node('transform_fusion')
    rospy.loginfo('Transform Fusion Node Inited...')

    rospy.Subscriber('/Odometry', Odometry, cb_save_cur_odom, queue_size=1)
    rospy.Subscriber('/map_to_odom', Odometry, cb_save_map_to_odom, queue_size=1)

    pub_localization = rospy.Publisher('/localization', Odometry, queue_size=1)

    fusion_thread = threading.Thread(target=transform_fusion)
    fusion_thread.daemon = True
    fusion_thread.start()

    rospy.spin()
