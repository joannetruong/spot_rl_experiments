import argparse
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from depth_map_utils import fill_in_fast, fill_in_multiscale
from sensor_msgs.msg import CompressedImage, Image
from spot_wrapper.spot import (Spot, SpotCamIds, image_response_to_cv2,
                               scale_depth_img)
from std_msgs.msg import Float32

RGB_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
ROBOT_STATE_TOPIC = "/robot_state"
COLLISION_TOPIC = "/collision"

SRC2MSG = {
    SpotCamIds.FRONTLEFT_DEPTH: Image,
    SpotCamIds.FRONTRIGHT_DEPTH: Image,
    SpotCamIds.BACK_DEPTH: Image,
    SpotCamIds.LEFT_DEPTH: Image,
    SpotCamIds.RIGHT_DEPTH: Image,
}

MAX_DEPTH = 10.0
MIN_DEPTH = 0.3
WIDTH = 320
HEIGHT = 240
FILTER_FRONT_DEPTH = True
CLAMP_DEPTH = True


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


class ExternalRosPublisher:
    def __init__(self, spot, verbose=False):
        rospy.init_node("spot_ros_node", disable_signals=True)
        self.spot = spot

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate raw image publishers
        self.sources = list(SRC2MSG.keys())

        # Instantiate filtered image publishers
        self.monitor_collision = (
            SpotCamIds.FRONTLEFT_DEPTH in self.sources
            and SpotCamIds.FRONTRIGHT_DEPTH in self.sources
            and SpotCamIds.BACK_DEPTH in self.sources
            and SpotCamIds.LEFT_DEPTH in self.sources
            and SpotCamIds.RIGHT_DEPTH in self.sources
        )
        self.collision_pub = rospy.Publisher(COLLISION_TOPIC, Float32, queue_size=1)

        self.last_publish = time.time()
        self.verbose = verbose
        rospy.loginfo("[spot_ros_node]: Publishing has started.")

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
        for src, response in zip(self.sources, image_responses):
            img = image_response_to_cv2(response)

            # Publish filtered front depth images later
            if src in depth_keys:
                depth_eyes[src] = img

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
            scaled_depth = filter_depth(scaled_depth, MAX_DEPTH)
            min_x_depths.append(self.get_min_dist(scaled_depth))
        for y in y_keys:
            scaled_depth = scale_depth_img(depth_eyes[y], max_depth=MAX_DEPTH)
            scaled_depth = np.uint8(scaled_depth * 255.0)
            scaled_depth = filter_depth(scaled_depth, MAX_DEPTH)
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
    def get_min_dist(cv_depth):
        return MAX_DEPTH * np.min(cv_depth) / 255


class ExternalRosSubscriber:
    def __init__(self, node_name):
        rospy.init_node(node_name, disable_signals=True)

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate subscribers
        rospy.Subscriber(
            DEPTH_TOPIC,
            Image,
            self.front_depth_callback,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.Subscriber(
            RGB_TOPIC,
            Image,
            self.front_rgb_callback,
            queue_size=1,
            buff_size=2**24,
        )

        # Msg holders
        self.front_depth = None
        self.front_rgb = None

        self.depth_updated = False
        self.rgb_updated = False
        self.collision_updated = False
        rospy.loginfo(f"[{node_name}]: Subscribing has started.")

    def front_depth_callback(self, msg):
        self.front_depth = msg
        self.depth_updated = True

    def front_rgb_callback(self, msg):
        self.front_rgb = msg
        self.rgb_updated = True

    def collision_callback(self, msg):
        self.collision = msg.data
        self.collision_updated = True

    @property
    def collided(self):
        if not self.collision_updated:
            return 0.0
        return self.collision

    def ros_to_img(self, ros_img):
        # Gather latest images
        if isinstance(ros_img, CompressedImage):
            return self.cv_bridge.compressed_imgmsg_to_cv2(ros_img)
        elif isinstance(ros_img, Image):
            return self.cv_bridge.imgmsg_to_cv2(ros_img)

    @property
    def front_depth_img(self):
        if self.front_depth is None or not self.depth_updated:
            print("IMAGE IS NONE!")
            return None
        depth_img = self.ros_to_img(self.front_depth)
        depth_img = scale_depth_img(depth_img, max_depth=MAX_DEPTH)
        depth_img = np.uint8(depth_img * 255.0)
        if FILTER_FRONT_DEPTH:
            depth_img = filter_depth(depth_img, MAX_DEPTH)
        depth_img = cv2.resize(depth_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        return depth_img

    @property
    def front_rgb_img(self):
        if self.front_rgb is None or not self.rgb_updated:
            print("IMAGE IS NONE!")
            return None
        return self.ros_to_img(self.front_rgb)


def main():
    # spot = Spot("external_camera_node")
    # srn = ExternalRosSubscriber(spot)

    spot = Spot("spot_ros_node")
    srn = ExternalRosPublisher(spot, False)
    while not rospy.is_shutdown():
        srn.publish_msgs()


if __name__ == "__main__":
    main()
