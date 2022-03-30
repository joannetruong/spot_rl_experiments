import argparse
import time

import blosc
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from depth_map_utils import fill_in_fast, fill_in_multiscale
from sensor_msgs.msg import CompressedImage, Image
from spot_wrapper.spot import (Spot, SpotCamIds, image_response_to_cv2,
                               scale_depth_img)
from std_msgs.msg import ByteMultiArray, Float32MultiArray

FRONT_DEPTH_TOPIC = "/spot_cams/filtered_front_depth"
FRONT_GRAY_TOPIC = "/spot_cams/front_gray"
ROBOT_STATE_TOPIC = "/robot_state"
SRC2MSG = {
    SpotCamIds.FRONTLEFT_DEPTH: Image,
    SpotCamIds.FRONTRIGHT_DEPTH: Image,
    SpotCamIds.FRONTLEFT_FISHEYE: Image,
    SpotCamIds.FRONTRIGHT_FISHEYE: Image,
}
MAX_DEPTH = 3.5
FILTER_FRONT_DEPTH = False
CLAMP_DEPTH = False


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
        self.use_front_depth = (
            SpotCamIds.FRONTLEFT_DEPTH in self.sources
            and SpotCamIds.FRONTRIGHT_DEPTH in self.sources
        )
        self.filter_front_depth = FILTER_FRONT_DEPTH
        if self.use_front_depth:
            self.front_depth_pub = rospy.Publisher(
                FRONT_DEPTH_TOPIC, Image, queue_size=1
            )
        self.use_front_gray = (
            SpotCamIds.FRONTLEFT_FISHEYE in self.sources
            and SpotCamIds.FRONTRIGHT_FISHEYE in self.sources
        )
        if self.use_front_gray:
            self.front_gray_pub = rospy.Publisher(FRONT_GRAY_TOPIC, Image, queue_size=1)
        self.last_publish = time.time()
        self.verbose = verbose
        rospy.loginfo("[spot_ros_node]: Publishing has started.")

    def publish_msgs(self):
        st = time.time()
        image_responses = self.spot.get_image_responses(self.sources, quality=None)
        retrieval_time = time.time() - st
        # Publish raw images
        depth_eyes = {}
        gray_eyes = {}
        for pub, src, response in zip(self.img_pubs, self.sources, image_responses):
            img = image_response_to_cv2(response)

            # Publish filtered front depth images later
            if (
                src in [SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH]
                and self.use_front_depth
            ):
                depth_eyes[src] = img

            if (
                src in [SpotCamIds.FRONTRIGHT_FISHEYE, SpotCamIds.FRONTLEFT_FISHEYE]
                and self.use_front_gray
            ):
                gray_eyes[src] = img
        if self.use_front_gray:
            g_keys = [SpotCamIds.FRONTRIGHT_FISHEYE, SpotCamIds.FRONTLEFT_FISHEYE]
            gray_merged = np.hstack([gray_eyes[g] for g in g_keys])

            gray_msg = self.cv_bridge.cv2_to_imgmsg(gray_merged)
            self.front_gray_pub.publish(gray_msg)

        # Filter and publish
        if self.use_front_depth:
            # Merge
            d_keys = [SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH]

            depth_merged = np.hstack([depth_eyes[d] for d in d_keys])
            # Filter
            depth_merged = scale_depth_img(depth_merged, max_depth=MAX_DEPTH)
            depth_merged = np.uint8(depth_merged * 255.0)
            if self.filter_front_depth:
                depth_merged = self.filter_depth(depth_merged, MAX_DEPTH)
            depth_merged = cv2.resize(
                depth_merged, (256, 256), interpolation=cv2.INTER_AREA
            )
            depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_merged, encoding="mono8")

            self.front_depth_pub.publish(depth_msg)
        if self.verbose:
            rospy.loginfo(
                f"[spot_ros_node]: Image retrieval / publish time: "
                f"{1 / retrieval_time:.4f} / {1 / (time.time() - self.last_publish):.4f} Hz"
            )
        self.last_publish = time.time()

    @staticmethod
    def filter_depth(img, max_depth):
        img = (
            fill_in_multiscale(img.copy().astype(np.float32) * (max_depth / 255.0))[0]
            * (255.0 / max_depth)
        ).astype(np.uint8)
        if CLAMP_DEPTH:
            img[img < 3.0] = 255.0
        return img

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

        # Msg holders
        self.front_depth = None
        self.front_gray = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.depth_updated = False
        self.gray_updated = False
        rospy.loginfo(f"[{node_name}]: Subscribing has started.")

    def front_depth_callback(self, msg):
        self.front_depth = msg
        self.depth_updated = True

    def front_gray_callback(self, msg):
        self.front_gray = msg
        self.gray_updated = True

    def robot_state_callback(self, msg):
        self.x, self.y, self.yaw = msg.data[:3]

    @property
    def front_depth_img(self):
        if self.front_depth is None or not self.depth_updated:
            print("IMAGE IS NONE!")
            return None
        # Gather latest images
        if isinstance(self.front_depth, ByteMultiArray):
            return decode_ros_blosc(self.front_depth)
        elif isinstance(self.front_depth, CompressedImage):
            return self.cv_bridge.compressed_imgmsg_to_cv2(self.front_depth)
        elif isinstance(self.front_depth, Image):
            return self.cv_bridge.imgmsg_to_cv2(self.front_depth)

    @property
    def front_gray_img(self):
        if self.front_gray is None or not self.gray_updated:
            print("IMAGE IS NONE!")
            return None
        # Gather latest images
        if isinstance(self.front_gray, ByteMultiArray):
            return decode_ros_blosc(self.front_gray)
        elif isinstance(self.front_gray, CompressedImage):
            return self.cv_bridge.compressed_imgmsg_to_cv2(self.front_gray)
        elif isinstance(self.front_gray, Image):
            return self.cv_bridge.imgmsg_to_cv2(self.front_gray)


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
