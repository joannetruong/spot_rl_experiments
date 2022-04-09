import argparse
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from depth_map_utils import fill_in_fast, fill_in_multiscale
from sensor_msgs.msg import CompressedImage, Image
from spot_wrapper.spot import Spot, image_response_to_cv2, scale_depth_img
from std_msgs.msg import Float32MultiArray

RGB_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
ROBOT_STATE_TOPIC = "/robot_state"

MAX_DEPTH = 10.0
MIN_DEPTH = 0.3
WIDTH = 320
HEIGHT = 240
FILTER_FRONT_DEPTH = True
CLAMP_DEPTH = True


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
        rospy.loginfo(f"[{node_name}]: Subscribing has started.")

    def front_depth_callback(self, msg):
        self.front_depth = msg
        self.depth_updated = True

    def front_rgb_callback(self, msg):
        self.front_rgb = msg
        self.rgb_updated = True

    @property
    def front_depth_img(self):
        if self.front_depth is None or not self.depth_updated:
            print("IMAGE IS NONE!")
            return None
        if isinstance(self.front_depth, CompressedImage):
            depth_img = self.cv_bridge.compressed_imgmsg_to_cv2(self.front_depth)
        elif isinstance(self.front_depth, Image):
            depth_img = self.cv_bridge.imgmsg_to_cv2(self.front_depth)
        depth_img = scale_depth_img(depth_img, max_depth=MAX_DEPTH)
        depth_img = np.uint8(depth_img * 255.0)
        if FILTER_FRONT_DEPTH:
            depth_img = self.filter_depth(depth_img, MAX_DEPTH)
        depth_img = cv2.resize(depth_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        return depth_img

    @property
    def front_rgb_img(self):
        if self.front_rgb is None or not self.rgb_updated:
            print("IMAGE IS NONE!")
            return None
        if isinstance(self.front_rgb, CompressedImage):
            return self.cv_bridge.compressed_imgmsg_to_cv2(self.front_rgb)
        elif isinstance(self.front_rgb, Image):
            return self.cv_bridge.imgmsg_to_cv2(self.front_rgb)

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


def main():
    parser = argparse.ArgumentParser()
    spot = Spot("external_camera_node")
    srn = ExternalRosSubscriber(spot)


if __name__ == "__main__":
    main()
