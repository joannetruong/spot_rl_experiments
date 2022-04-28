import argparse
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from depth_map_utils import fill_in_fast, fill_in_multiscale
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import CompressedImage, Image
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2, scale_depth_img
from std_msgs.msg import ByteMultiArray, Float32, Float32MultiArray

FRONT_DEPTH_TOPIC = "/spot_cams/filtered_front_depth"
COLLISION_TOPIC = "/collision"
FRONT_GRAY_TOPIC = "/spot_cams/front_gray"
ROBOT_STATE_TOPIC = "/robot_state"

SRC2MSG = {
    SpotCamIds.FRONTLEFT_DEPTH: Image,
    SpotCamIds.FRONTRIGHT_DEPTH: Image,
    SpotCamIds.BACK_DEPTH: Image,
    SpotCamIds.LEFT_DEPTH: Image,
    SpotCamIds.RIGHT_DEPTH: Image,
}

MAX_DEPTH = 10.0
MIN_DEPTH = 0.0
FILTER_FRONT_DEPTH = True
CLAMP_DEPTH = True

class SpotRosPublisher:
    def __init__(self, spot, verbose=False):
        rospy.init_node("spot_ros_node", disable_signals=True)
        self.spot = spot

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate raw image publishers
        self.sources = list(SRC2MSG.keys())
        self.img_pubs = [
            rospy.Publisher(f"/spot_cams/{k}", v, queue_size=1)
            for k, v in SRC2MSG.items()
        ]

        # Instantiate filtered image publishers
        self.monitor_collision = (
            SpotCamIds.FRONTLEFT_DEPTH in self.sources
            and SpotCamIds.FRONTRIGHT_DEPTH in self.sources
            and SpotCamIds.BACK_DEPTH in self.sources
            and SpotCamIds.LEFT_DEPTH in self.sources
            and SpotCamIds.RIGHT_DEPTH in self.sources
        )
        if self.monitor_collision:
            self.collision_pub = rospy.Publisher(COLLISION_TOPIC, Float32, queue_size=1)

        self.last_publish = time.time()
        self.verbose = verbose
        rospy.loginfo("[spot_ros_node]: Publishing has started.")

    def publish_robot_pose(self):
        robot_position = self.spot.get_robot_position()
        robot_quat = self.spot.get_robot_quat()

        robot_pose = PoseStamped()
        robot_pose.header.stamp = rospy.Time.now()
        robot_pose.header.frame_id = "map"
        robot_pose.pose.position.x = robot_position[0]
        robot_pose.pose.position.y = robot_position[1]
        robot_pose.pose.position.z = robot_position[2]
        robot_pose.pose.orientation.x = robot_quat[0]
        robot_pose.pose.orientation.y = robot_quat[1]
        robot_pose.pose.orientation.z = robot_quat[2]
        robot_pose.pose.orientation.w = robot_quat[3]
        self.pose_pub.publish(robot_pose)

    def publish_robot_vel(self):
        robot_linear_velocity = self.spot.get_robot_linear_vel("vision")
        robot_angular_velocity = self.spot.get_robot_angular_vel("vision")

        robot_twist = TwistStamped()
        robot_twist.header.stamp = rospy.Time.now()
        robot_twist.twist.linear.x = robot_linear_velocity[0]
        robot_twist.twist.linear.y = robot_linear_velocity[1]
        robot_twist.twist.linear.z = robot_linear_velocity[2]
        robot_twist.twist.angular.x = robot_angular_velocity[0]
        robot_twist.twist.angular.y = robot_angular_velocity[1]
        robot_twist.twist.angular.z = robot_angular_velocity[2]

        self.vis_vel_pub.publish(robot_twist)

    def publish_collisions(self, depth_eyes, collision_eyes):
        # Filter
        min_x_depths = []
        min_y_depths = []
        num_collisions = 0
        all_depth_eyes = {**depth_eyes, **collision_eyes}
        x_keys = [
            SpotCamIds.FRONTRIGHT_DEPTH,
            SpotCamIds.FRONTLEFT_DEPTH,
            SpotCamIds.BACK_DEPTH,
        ]
        y_keys = [SpotCamIds.LEFT_DEPTH, SpotCamIds.RIGHT_DEPTH]

        for x in x_keys:
            scaled_depth = scale_depth_img(all_depth_eyes[x], max_depth=MAX_DEPTH)
            scaled_depth = np.uint8(scaled_depth * 255.0)
            scaled_depth = self.filter_depth(scaled_depth, MAX_DEPTH)
            min_x_depths.append(self.get_min_dist(scaled_depth))
        for y in y_keys:
            scaled_depth = scale_depth_img(all_depth_eyes[y], max_depth=MAX_DEPTH)
            scaled_depth = np.uint8(scaled_depth * 255.0)
            scaled_depth = self.filter_depth(scaled_depth, MAX_DEPTH)
            min_y_depths.append(self.get_min_dist(scaled_depth))
        if any(depth < 0.3 for depth in min_x_depths) or any(
            depth == 3.5 for depth in min_x_depths
        ):
            num_collisions = 1.0
        if any(depth < 0.2 for depth in min_y_depths) or any(
            depth == 3.5 for depth in min_y_depths
        ):
            num_collisions = 1.0
        collisions = Float32()
        collisions.data = num_collisions
        self.collision_pub.publish(collisions)

    def publish_msgs(self):
        st = time.time()
        image_responses = self.spot.get_image_responses(self.sources, quality=None)
        retrieval_time = time.time() - st
        # Publish raw images
        depth_eyes = {}
        depth_keys = [
            SpotCamIds.FRONTLEFT_DEPTH,
            SpotCamIds.FRONTRIGHT_DEPTH,
            SpotCamIds.BACK_DEPTH,
            SpotCamIds.LEFT_DEPTH,
            SpotCamIds.RIGHT_DEPTH,
        ]
        for pub, src, response in zip(self.img_pubs, self.sources, image_responses):
            img = image_response_to_cv2(response)

            # Publish filtered front depth images later
            if src in depth_keys:
                depth_eyes[src] = img

        if self.monitor_collision:
            # Filter
            min_x_depths = []
            min_y_depths = []
            num_collisions = 0
            x_keys = [
                SpotCamIds.FRONTRIGHT_DEPTH,
                SpotCamIds.FRONTLEFT_DEPTH,
                SpotCamIds.BACK_DEPTH,
            ]
            y_keys = [SpotCamIds.LEFT_DEPTH, SpotCamIds.RIGHT_DEPTH]

            for x in x_keys:
                scaled_depth = scale_depth_img(depth_eyes[x], max_depth=MAX_DEPTH)
                scaled_depth = np.uint8(scaled_depth * 255.0)
                scaled_depth = self.filter_depth(scaled_depth, MAX_DEPTH)
                min_x_depths.append(self.get_min_dist(scaled_depth))
            for y in y_keys:
                scaled_depth = scale_depth_img(depth_eyes[y], max_depth=MAX_DEPTH)
                scaled_depth = np.uint8(scaled_depth * 255.0)
                scaled_depth = self.filter_depth(scaled_depth, MAX_DEPTH)
                min_y_depths.append(self.get_min_dist(scaled_depth))
            if any(depth < 0.3 for depth in min_x_depths) or any(
                depth == 3.5 for depth in min_x_depths
            ):
                num_collisions = 1.0
            if any(depth < 0.2 for depth in min_y_depths) or any(
                depth == 3.5 for depth in min_y_depths
            ):
                num_collisions = 1.0
            collisions = Float32()
            collisions.data = num_collisions
            self.collision_pub.publish(collisions)
        if self.verbose:
            rospy.loginfo(
                f"[spot_ros_node]: Image retrieval / publish time: "
                f"{1 / retrieval_time:.4f} / {1 / (time.time() - self.last_publish):.4f} Hz"
            )
        self.last_publish = time.time()

    @staticmethod
    def filter_depth(img, max_depth):
        filtered_depth_img = (
            fill_in_multiscale(img.astype(np.float32) * (max_depth / 255.0))[0]
            * (255.0 / max_depth)
        ).astype(np.uint8)
        # Recover pixels that weren't black before but were turned black by filtering
        recovery_pixels = np.logical_and(img != 0, filtered_depth_img == 0)
        filtered_depth_img[recovery_pixels] = img[recovery_pixels]
        if CLAMP_DEPTH:
            filtered_depth_img[filtered_depth_img == 0] = 255
        return filtered_depth_img

    @staticmethod
    def median_filter_depth(img):
        num_iters = 10
        kernel_size = 9
        # Blur
        for _ in range(num_iters):
            filtered = cv2.medianBlur(img, kernel_size)
            filtered[img > 0] = img[img > 0]
            if CLAMP_DEPTH:
                filtered[filtered < 3.0] = 255.0
            img = filtered

        return img

    @staticmethod
    def get_min_dist(cv_depth):
        return MAX_DEPTH * np.min(cv_depth) / 255


class SpotRosSubscriber:
    def __init__(self, node_name):
        rospy.init_node(node_name, disable_signals=True)

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate subscribers
        rospy.Subscriber(
            FRONT_DEPTH_TOPIC,
            Image,
            self.front_depth_callback,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.Subscriber(
            FRONT_GRAY_TOPIC,
            Image,
            self.front_gray_callback,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.Subscriber(
            ROBOT_STATE_TOPIC,
            Float32MultiArray,
            self.robot_state_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            COLLISION_TOPIC,
            Float32,
            self.collision_callback,
            queue_size=1,
            buff_size=2**24,
        )

        # Msg holders
        self.front_depth = None
        self.back_depth = None
        self.collision = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.depth_updated = False
        self.rgb_updated = False
        self.collision_updated = False
        rospy.loginfo(f"[{node_name}]: Subscribing has started.")

    def front_depth_callback(self, msg):
        self.front_depth = msg
        self.front_depth_updated = True

    def front_gray_callback(self, msg):
        self.front_gray = msg
        self.gray_updated = True

    def robot_state_callback(self, msg):
        self.x, self.y, self.yaw = msg.data[:3]

    def collision_callback(self, msg):
        self.collision = msg.data
        self.collision_updated = True

    def collided(self):
        if not self.collision_updated:
            return 0.0
        return self.collision

    def ros_to_img(self, ros_img):
        # Gather latest images
        if isinstance(ros_img, ByteMultiArray):
            return decode_ros_blosc(ros_img)
        elif isinstance(ros_img, CompressedImage):
            return self.cv_bridge.compressed_imgmsg_to_cv2(ros_img)
        elif isinstance(ros_img, Image):
            return self.cv_bridge.imgmsg_to_cv2(ros_img)

    @property
    def front_depth_img(self):
        if self.front_depth is None or not self.front_depth_updated:
            print("DEPTH IMAGE IS NONE!")
            return None
        return self.ros_to_img(self.front_depth)

    @property
    def collided(self):
        if not self.collision_updated:
            return 0.0
        self.collision_updated = False
        return self.collision

    @property
    def front_gray_img(self):
        if self.front_gray is None or not self.gray_updated:
            print("GRAY IMAGE IS NONE!")
            return None
        return self.ros_to_img(self.front_rgb)

    @property
    def collided(self):
        if not self.collision_updated:
            return 0.0
        return self.collision

    def ros_to_img(self, ros_img):
        # Gather latest images
        if isinstance(ros_img, ByteMultiArray):
            return decode_ros_blosc(ros_img)
        elif isinstance(ros_img, CompressedImage):
            return self.cv_bridge.compressed_imgmsg_to_cv2(ros_img)
        elif isinstance(ros_img, Image):
            return self.cv_bridge.imgmsg_to_cv2(ros_img)

    @property
    def front_depth_img(self):
        if self.front_depth is None or not self.depth_updated:
            print("IMAGE IS NONE!")
            return None
        return self.ros_to_img(self.front_depth)

    @property
    def front_rgb_img(self):
        if self.front_rgb is None or not self.rgb_updated:
            print("IMAGE IS NONE!")
            return None
        return self.ros_to_img(self.front_rgb)


class SpotRosProprioceptionPublisher:
    def __init__(self, spot, verbose=False):
        rospy.init_node("spot_ros_proprioception_node", disable_signals=True)
        self.spot = spot

        # Instantiate filtered image publishers
        self.pub = rospy.Publisher(ROBOT_STATE_TOPIC, Float32MultiArray, queue_size=1)
        self.last_publish = time.time()
        rospy.loginfo("[spot_ros_proprioception_node]: Publishing has started.")

    def publish_msgs(self):
        while time.time() - self.last_publish < 1 / 100:
            # Limit to 100 Hz max
            pass
        st = time.time()
        robot_state = self.spot.get_robot_state()
        if self.verbose:
            rospy.loginfo(
                f"[spot_ros_proprioception_node]: Proprioception retrieval / publish time: "
                f"{1/(time.time() - st):.4f} / {1/(time.time() - self.last_publish):.4f} Hz"
            )
        msg = Float32MultiArray()
        xy_yaw = self.spot.get_xy_yaw(robot_state=robot_state)
        msg.data = np.array(
            list(xy_yaw),
            dtype=np.float32,
        )
        self.pub.publish(msg)
        self.last_publish = time.time()


def decode_ros_blosc(msg: ByteMultiArray):
    byte_data = msg.data
    byte_data = [(i + 128).to_bytes(1, "big") for i in byte_data]
    decoded = blosc.unpack_array(b"".join(byte_data))
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--proprioception", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.proprioception:
        name = "spot_ros_proprioception_node"
        cls = SpotRosProprioceptionPublisher
    else:
        name = "spot_ros_node"
        cls = SpotRosPublisher

    spot = Spot(name)
    srn = cls(spot, args.verbose)
    while not rospy.is_shutdown():
        srn.publish_msgs()


if __name__ == "__main__":
    main()
