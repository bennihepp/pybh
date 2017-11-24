import os
import time
import sys
try:
    from cStringIO import StringIO
except:
    from io import BytesIO
import numpy as np
import cv2
from unrealcv import Client
from .engine import BaseEngine
from pybh import math_utils
from pybh.contrib import transformations


class UnrealCVWrapper(BaseEngine):

    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_CUBIC = cv2.INTER_CUBIC

    class Exception(RuntimeError):
        pass

    def __init__(self,
                 address=None,
                 port=None,
                 image_scale_factor=1.0,
                 max_depth_distance=np.finfo(np.float).max,
                 max_depth_viewing_angle=math_utils.degrees_to_radians(90.),
                 max_request_trials=5,
                 request_timeout=5.0,
                 location_tolerance=1e-3,
                 orientation_tolerance=1e-2,
                 connect=True,
                 connect_wait_time=5.0,
                 connect_timeout=5.0):
        super(UnrealCVWrapper, self).__init__(max_depth_distance, max_depth_viewing_angle)
        self._width = None
        self._height = None
        self._image_scale_factor = image_scale_factor
        self._max_request_trials = max_request_trials
        self._request_timeout = request_timeout
        self._connect_wait_time = connect_wait_time
        self._connect_timeout = connect_timeout
        self._location_tolerance = location_tolerance
        self._orientation_tolerance = orientation_tolerance
        self._request_trials = 0
        if address is None:
            address = '127.0.0.1'
        if port is None:
            port = 9000
        self._cv_client = Client((address, port))
        if connect:
            self.connect()

    def _get_location_from_unreal_location(self, unreal_location):
        """Convert location in Unreal Engine coordinate frame to right-handed coordinate frame in meters"""
        # Convert left-handed Unreal system to right-handed system
        location = math_utils.convert_xyz_from_left_to_right_handed(unreal_location)
        # Convert location from Unreal (cm) to meters
        location *= 0.01
        return location

    def _get_unreal_location_from_location(self, location):
        """Convert location in right-handed coordinate frame in meters to Unreal Engine location"""
        unreal_location = math_utils.convert_xyz_from_right_to_left_handed(location)
        # Convert meters to Unreal (cm)
        unreal_location *= 100
        return unreal_location

    def _get_euler_rpy_from_unreal_pyr(self, unreal_pyr):
        """Convert rotation in Unreal Engine angles to euler angles in radians"""
        pitch, yaw, roll = [v * np.pi / 180. for v in unreal_pyr]
        euler_rpy = np.array([roll, pitch, yaw])
        roll, pitch, yaw = math_utils.convert_rpy_from_left_to_right_handed(euler_rpy)
        if pitch <= -np.pi:
            pitch += 2 * np.pi
        elif pitch > np.pi:
            pitch -= 2 * np.pi
        euler_rpy = np.array([roll, pitch, yaw])
        return euler_rpy

    def _get_unreal_pyr_from_euler_rpy(self, euler_rpy):
        """Convert rotation in euler angles in radians to Unreal Engine angles"""
        euler_rpy = [v * 180. / np.pi for v in euler_rpy]
        unreal_roll, unreal_pitch, unreal_yaw = math_utils.convert_rpy_from_right_to_left_handed(euler_rpy)
        return [unreal_pitch, unreal_yaw, unreal_roll]

    def _get_quaternion_from_euler_rpy(self, euler_rpy):
        """Convert euler angles in radians to quaternion"""
        roll, pitch, yaw = euler_rpy
        quat = transformations.quaternion_from_euler(yaw, pitch, roll, 'rzyx')
        return quat

    def _is_location_set(self, location):
        current_location = self.get_location()
        if np.all(np.abs(current_location - location) < self._location_tolerance):
            return True
        return False

    def _is_orientation_rpy_set(self, roll, pitch, yaw):
        current_roll, current_pitch, current_yaw = self.get_orientation_rpy()
        if math_utils.is_angle_equal(current_roll, roll, self._orientation_tolerance) \
                and math_utils.is_angle_equal(current_pitch, pitch, self._orientation_tolerance) \
                and math_utils.is_angle_equal(current_yaw, yaw, self._orientation_tolerance):
            return True
        return False

    def _is_pose_rpy_set(self, pose):
        location = pose[0]
        roll, pitch, yaw = pose[1]
        current_location, current_euler_rpy = self.get_pose_rpy()
        current_roll, current_pitch, current_yaw = current_euler_rpy
        # print("Desired pose:", location, roll, pitch, yaw)
        # print("Current pose:", current_location, current_roll, current_pitch, current_yaw)
        if np.all(np.abs(current_location - location) < self._location_tolerance) \
                and math_utils.is_angle_equal(current_roll, roll, self._orientation_tolerance) \
                and math_utils.is_angle_equal(current_pitch, pitch, self._orientation_tolerance) \
                and math_utils.is_angle_equal(current_yaw, yaw, self._orientation_tolerance):
            return True
        return False

    def _read_npy(self, npy_io):
        return np.load(npy_io)

    def _read_npy_from_bytes(self, npy_bytes):
        if sys.version_info.major == 3:
            npy_io = BytesIO(npy_bytes)
        else:
            npy_io = StringIO(npy_bytes)
        return self._read_npy(npy_io)

    def _ray_distance_to_depth_image(self, ray_distance_image, focal_length):
        """Convert a ray-distance image to a plane depth image"""
        height = ray_distance_image.shape[0]
        width = ray_distance_image.shape[1]
        x_c = np.float(width) / 2 - 1
        y_c = np.float(height) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, width - 1, num=width), np.linspace(0, height - 1, num=height))
        dist_from_center = ((rows - y_c) ** 2 + (columns - x_c) ** 2) ** (0.5)
        depth_image = ray_distance_image / (1 + (dist_from_center / focal_length) ** 2) ** (0.5)
        return depth_image

    def _unrealcv_request(self, request):
        """Send a request to UnrealCV. Automatically retry in case of timeout."""
        if type(request) is str:
            request = request.encode("utf-8", "ignore")
        result = None
        while result is None:
            self._request_trials += 1
            if self._request_trials > self._max_request_trials:
                raise self.Exception("UnrealCV request failed")
            result = self._cv_client.request(request, self._request_timeout)
            if result is None:
                print("UnrealCV request timed out. Retrying.")
                self._cv_client.disconnect()
                time.sleep(self._connect_wait_time)
                self._cv_client.connect()
        self._request_trials = 0
        return result

    def connect(self):
        """Open connection to UnrealCV"""
        if self._cv_client.isconnected():
            print("WARNING: Already connected to UnrealCV")
        else:
            self._cv_client.connect(self._connect_timeout)
            if not self._cv_client.isconnected():
                raise(self.Exception("Unable to connect to UnrealCV"))

    def close(self):
        """Close connection to UnrealCV"""
        self._cv_client.disconnect()

    def unrealcv_client(self):
        """Return underlying UnrealCV client"""
        return self._cv_client

    def scale_image(self, image, scale_factor=None, interpolation_mode=INTER_CUBIC):
        """Scale an image to the desired size"""
        if scale_factor is None:
            scale_factor = self._image_scale_factor
        if scale_factor == 1:
            return image
        dsize = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        scaled_image = cv2.resize(image, dsize=dsize, interpolation=interpolation_mode)
        return scaled_image

    def scale_image_with_nearest_interpolation(self, image, scale_factor=None):
        """Scale an image to the desired size using 'nearest' interpolation"""
        return self.scale_image(image, scale_factor=scale_factor, interpolation_mode=self.INTER_NEAREST)

    def get_width(self):
        """Return width of image plane"""
        if self._width is None:
            image = self.get_ray_distance_image()
            self._height = image.shape[0]
            self._width = image.shape[1]
        return self._width

    def get_height(self):
        """Return height of image plane"""
        if self._height is None:
            self.get_width()
        return self._height

    def get_horizontal_field_of_view_degrees(self):
        """Return the horizontal field of view of the camera in degrees"""
        horz_fov_bytes = self._unrealcv_request(b'vget /camera/0/horizontal_fieldofview')
        horz_fov_degrees = float(horz_fov_bytes)
        return horz_fov_degrees

    def get_horizontal_field_of_view(self):
        """Return the horizontal field of view of the camera in radians"""
        horz_fov_degrees = self.get_horizontal_field_of_view_degrees()
        horz_fov = math_utils.degrees_to_radians(horz_fov_degrees)
        return horz_fov

    def set_horizontal_field_of_view(self, horz_fov):
        """Set the horizontal field of view of the camera in radians"""
        horz_fov = math_utils.radians_to_degrees(horz_fov)
        response = self._unrealcv_request('vset /camera/0/horizontal_fieldofview {:f}'.format(horz_fov))
        if response != b"ok":
            raise self.Exception("UnrealCV request failed: {}".format(response))

    def get_image_scale_factor(self):
        """Return scale factor for image retrieval"""
        return self._image_scale_factor

    def get_focal_length(self):
        """Return focal length of camera"""
        try:
            horz_fov = self.get_horizontal_field_of_view()
        except UnrealCVWrapper.Exception:
            print("WARNING: UnrealCV does not support querying horizontal field of view. Assuming 90 degrees.")
            horz_fov = math_utils.degrees_to_radians(90.0)
        width = self.get_width()
        focal_length = width / (2 * np.tan(horz_fov / 2.))
        return focal_length

    def set_focal_length(self, focal_length):
        """Set the focal length of camera"""
        horz_fov = 2 * np.arctan(self.get_width() / (2 * focal_length))
        self.set_horizontal_field_of_view(horz_fov)

    def get_intrinsics(self):
        """Return intrinsics of camera"""
        intrinsics = np.zeros((3, 3))
        intrinsics[0, 0] = self.get_focal_length()
        intrinsics[1, 1] = self.get_focal_length()
        intrinsics[0, 2] = (self.get_width() - 1) / 2.0
        intrinsics[1, 2] = (self.get_height() - 1) / 2.0
        intrinsics[2, 2] = 1.0
        return intrinsics

    def get_rgb_image(self, scale_factor=None):
        """Return the current RGB image"""
        img_bytes = self._unrealcv_request(b'vget /camera/0/lit png')
        img = np.fromstring(img_bytes, np.uint8)
        rgb_image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        rgb_image = self.scale_image(rgb_image, scale_factor)
        return rgb_image

    def get_rgb_image_by_file(self, scale_factor=None):
        """Return the current RGB image (transport via filesystem)"""
        filename = self._unrealcv_request(b'vget /camera/0/lit lit.png')
        rgb_image = cv2.imread(filename)
        rgb_image = self.scale_image(rgb_image, scale_factor)
        os.remove(filename)
        return rgb_image

    def get_normal_rgb_image(self, scale_factor=None):
        """Return the current normal image in RGB encoding (i.e. 128 is 0, 0 is -1, 255 is +1)"""
        img_bytes = self._unrealcv_request(b'vget /camera/0/normal png')
        img = np.fromstring(img_bytes, np.uint8)
        normal_image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        normal_image = self.scale_image_with_nearest_interpolation(normal_image, scale_factor)
        return normal_image

    def get_normal_rgb_image_by_file(self, scale_factor=None):
        """Return the current normal image in RGB encoding (i.e. 128 is 0, 0 is -1, 255 is +1)
        (transport via filesystem)
        """
        filename = self._unrealcv_request(b'vget /camera/0/normal normal.png')
        normal_image = cv2.imread(filename)
        normal_image = self.scale_image_with_nearest_interpolation(normal_image, scale_factor)
        os.remove(filename)
        return normal_image

    def get_normal_image(self, scale_factor=None):
        """Return the current normal image in vector representation"""
        normal_rgb_image = self.get_normal_rgb_image(scale_factor)
        # Image is in BGR ordering, so [z, y, x]. Convert to RGB.
        normal_rgb_image = normal_rgb_image[:, :, ::-1]
        normal_image = self._convert_normal_rgb_image_to_normal_image(normal_rgb_image, inplace=False)
        # Convert left-handed Unreal system to right-handed system
        normal_image[:, :, 1] = -normal_image[:, :, 1]
        self.filter_normal_image(normal_image)
        return normal_image

    def get_normal_image_by_file(self, scale_factor=None):
        """Return the current normal image in vector representation (transport via filesystem)"""
        normal_rgb_image = self.get_normal_rgb_image_by_file(scale_factor)
        # Image is in BGR ordering, so [z, y, x]. Convert to RGB.
        normal_rgb_image = normal_rgb_image[:, :, ::-1]
        normal_image = self._convert_normal_rgb_image_to_normal_image(normal_rgb_image, inplace=False)
        # Convert left-handed Unreal system to right-handed system
        normal_image[:, :, 1] = -normal_image[:, :, 1]
        return normal_image

    def get_ray_distance_image(self, scale_factor=None):
        """Return the current ray-distance image"""
        img_bytes = self._unrealcv_request(b'vget /camera/0/depth npy')
        ray_distance_image = self._read_npy_from_bytes(img_bytes)
        ray_distance_image = self.scale_image_with_nearest_interpolation(ray_distance_image, scale_factor)
        return ray_distance_image

    def get_ray_distance_image_by_file(self, scale_factor=None):
        """Return the current ray-distance image (transport via filesystem)"""
        filename = self._unrealcv_request(b'vget /camera/0/depth depth.exr')
        ray_distance_image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        ray_distance_image = self.scale_image_with_nearest_interpolation(ray_distance_image, scale_factor)
        os.remove(filename)
        return ray_distance_image

    def get_depth_image(self, scale_factor=None):
        """Return the current depth image"""
        # timer = utils.Timer()
        ray_distance_image = self.get_ray_distance_image(scale_factor)
        depth_image = self._ray_distance_to_depth_image(ray_distance_image, self.get_focal_length())
        # print("get_depth_image() took {}".format(timer.elapsed_seconds())
        return depth_image

    def get_depth_image_by_file(self, scale_factor=None):
        """Return the current depth image (transport via filesystem)"""
        ray_distance_image = self.get_ray_distance_image(scale_factor)
        depth_image = self._ray_distance_to_depth_image_by_file(ray_distance_image, self.get_focal_length())
        return depth_image

    def get_rgb_ray_distance_normal_images(self, scale_factor=None):
        """Return the current rgb, ray-distance and normal image"""
        resp_bytes = self._unrealcv_request(b'vget /camera/0/lit_depth_normal npy')
        if sys.version_info.major == 3:
            resp_io = BytesIO(resp_bytes)
        else:
            resp_io = StringIO(resp_bytes)
        rgb_image = self._read_npy(resp_io)
        rgb_image = self.scale_image(rgb_image, scale_factor)
        ray_distance_image = self._read_npy(resp_io)
        ray_distance_image = self.scale_image_with_nearest_interpolation(ray_distance_image, scale_factor)
        normal_image = self._read_npy(resp_io)
        normal_image = self.scale_image_with_nearest_interpolation(normal_image, scale_factor)
        return rgb_image, ray_distance_image, normal_image

    def get_rgb_depth_normal_images(self, scale_factor=None):
        """Return the current rgb, depth and normal image"""
        # timer = utils.Timer()
        rgb_image, ray_distance_image, normal_image = self.get_rgb_ray_distance_normal_images(scale_factor)
        depth_image = self._ray_distance_to_depth_image(ray_distance_image, self.get_focal_length())
        # print("get_depth_image() took {}".format(timer.elapsed_seconds())
        return rgb_image, depth_image, normal_image

    def get_vis_depth_image(self, scale_factor=None):
        """Return the current vis-depth image"""
        img_bytes = self._unrealcv_request(b'vget /camera/0/vis_depth npy')
        depth_image = self._read_npy_from_bytes(img_bytes)
        depth_image = self.scale_image_with_nearest_interpolation(depth_image, scale_factor)
        return depth_image

    def get_plane_depth_image(self, scale_factor=None):
        """Return the current plane-depth image"""
        img_bytes = self._unrealcv_request(b'vget /camera/0/plane_depth npy')
        depth_image = self._read_npy_from_bytes(img_bytes)
        depth_image = self.scale_image_with_nearest_interpolation(depth_image, scale_factor)
        return depth_image

    def get_location(self):
        """Return the current location in meters as [x, y, z]"""
        location_bytes = self._unrealcv_request(b'vget /camera/0/location')
        unreal_location = np.array([float(v) for v in location_bytes.split()])
        assert(len(unreal_location) == 3)
        location = self._get_location_from_unreal_location(unreal_location)
        return location

    def get_orientation_rpy(self):
        """Return the current orientation in radians as [roll, pitch, yaw]"""
        orientation_bytes = self._unrealcv_request(b'vget /camera/0/rotation')
        unreal_pyr = [float(v) for v in orientation_bytes.split()]
        assert(len(unreal_pyr) == 3)
        euler_rpy = self._get_euler_rpy_from_unreal_pyr(unreal_pyr)
        return euler_rpy

    def get_orientation_quat(self):
        """Return the current orientation quaterion quat = [w, x, y, z]"""
        euler_rpy = self.get_orientation_rpy()
        quat = self._get_quaternion_from_euler_rpy(euler_rpy)

        # # Transformation test for debugging
        # transform_mat = transformations.quaternion_matrix(quat)
        # sensor_x = transform_mat[:3, :3].dot(np.array([1, 0, 0]))
        # sensor_y = transform_mat[:3, :3].dot(np.array([0, 1, 0]))
        # sensor_z = transform_mat[:3, :3].dot(np.array([0, 0, 1]))
        # rospy.loginfo("sensor x axis: {} {} {}".format(sensor_x[0], sensor_x[1], sensor_x[2]))
        # rospy.loginfo("sensor y axis: {} {} {}".format(sensor_y[0], sensor_y[1], sensor_y[2]))
        # rospy.loginfo("sensor z axis: {} {} {}".format(sensor_z[0], sensor_z[1], sensor_z[2]))

        return quat

    def get_pose_rpy(self):
        """Return the current pose as a tuple of location and orientation rpy"""
        # return self.get_location(), self.get_orientation_quat()
        pose_bytes = self._unrealcv_request(b'vget /camera/0/pose')
        pose_unreal = np.array([float(v) for v in pose_bytes.split()])
        assert(len(pose_unreal) == 6)

        unreal_location = pose_unreal[:3]
        location = self._get_location_from_unreal_location(unreal_location)

        unreal_pyr = pose_unreal[3:]
        euler_rpy = self._get_euler_rpy_from_unreal_pyr(unreal_pyr)

        return location, euler_rpy

    def get_pose_quat(self):
        """Return the current pose as a tuple of location and orientation quaternion"""
        location, euler_rpy = self.get_pose_rpy()
        quat = self._get_quaternion_from_euler_rpy(euler_rpy)
        return location, quat

    def set_location(self, location, wait_until_set=False):
        """Set new location in meters as [x, y, z]"""
        unreal_location = self._get_unreal_location_from_location(location)
        # UnrealCV cannot handle scientific notation so we use :f format specifier
        request_bytes = 'vset /camera/0/location {:f} {:f} {:f}'.format(
            unreal_location[0], unreal_location[1], unreal_location[2])
        # print("Sending location request: {}".format(request_bytes))
        response = self._unrealcv_request(request_bytes)
        if response != b"ok":
            raise self.Exception("UnrealCV request failed: {}".format(response))
        if wait_until_set:
            start_time = time.time()
            while time.time() - start_time < self._request_timeout:
                if self._is_location_set(location):
                    time.sleep(0.1)
                    return
            raise self.Exception("UnrealCV: New orientation was not set within time limit")

    def set_orientation_rpy(self, roll, pitch, yaw, wait_until_set=False):
        """Set new orientation in radians"""
        unreal_pitch, unreal_yaw, unreal_roll = self._get_unreal_pyr_from_euler_rpy([roll, pitch, yaw])
        # UnrealCV cannot handle scientific notation so we use :f format specifier
        request_bytes = 'vset /camera/0/rotation {:f} {:f} {:f}'.format(
            unreal_pitch, unreal_yaw, unreal_roll)
        # print("Sending orientation request: {}".format(request_bytes))
        response = self._unrealcv_request(request_bytes)
        if response != b"ok":
            raise self.Exception("UnrealCV request failed: {}".format(response))
        if wait_until_set:
            start_time = time.time()
            while time.time() - start_time < self._request_timeout:
                if self._is_orientation_rpy_set(roll, pitch, yaw):
                    time.sleep(0.1)
                    return
            raise self.Exception("UnrealCV: New orientation was not set within time limit")

    def set_orientation_quat(self, quat):
        """Set new orientation quaterion quat = [w, x, y, z]"""
        # yaw, pitch, roll = transformations.euler_from_quaternion(quat, 'rxyz')
        yaw, pitch, roll = transformations.euler_from_quaternion(quat, 'rzyx')
        self.set_orientation_rpy(roll, pitch, yaw)

    def set_pose_rpy(self, pose, wait_until_set=False):
        """Set new pose in meters as [x, y, z] and radians [roll, pitch, yaw]"""
        location = pose[0]
        euler_rpy = pose[1]
        unreal_location = self._get_unreal_location_from_location(location)
        unreal_pyr = self._get_unreal_pyr_from_euler_rpy(euler_rpy)
        # UnrealCV cannot handle scientific notation so we use :f format specifier
        request_bytes = 'vset /camera/0/pose {:f} {:f} {:f} {:f} {:f} {:f}'.format(
            unreal_location[0], unreal_location[1], unreal_location[2],
            unreal_pyr[0], unreal_pyr[1], unreal_pyr[2])
        response = self._unrealcv_request(request_bytes)
        if response != b"ok":
            raise self.Exception("UnrealCV request failed: {}".format(response))
        if wait_until_set:
            start_time = time.time()
            while time.time() - start_time < self._request_timeout:
                if self._is_pose_rpy_set((location, euler_rpy)):
                    time.sleep(0.1)
                    return
            raise self.Exception("UnrealCV: New pose was not set within time limit")

    def set_pose_quat(self, pose, wait_until_set=False):
        """Set new pose as a tuple of location and orientation quaternion"""
        yaw, pitch, roll = transformations.euler_from_quaternion(pose[1], 'rzyx')
        euler_rpy = [roll, pitch, yaw]
        self.set_pose_rpy((pose[0], euler_rpy), wait_until_set)

    def enable_input(self):
        """Enable input in Unreal Engine"""
        self._unrealcv_request(b"vset /action/input/enable")

    def disable_input(self):
        """Disable input in Unreal Engine"""
        self._unrealcv_request(b"vset /action/input/disable")

    def get_objects(self):
        """List object names in Unreal Engine"""
        objects = self._unrealcv_request(b"vget /objects")
        objects = [object_name.strip() for object_name in objects.split()]
        return objects

    def show_object(self, object_name):
        """Show object in Unreal Engine"""
        response = self._unrealcv_request("vset /object/{}/show".format(object_name))
        if response != b"ok":
            raise self.Exception("UnrealCV request failed: {}".format(response))

    def hide_object(self, object_name):
        """Hide object in Unreal Engine"""
        response = self._unrealcv_request("vset /object/{}/hide".format(object_name))
        if response != b"ok":
            raise self.Exception("UnrealCV request failed: {}".format(response))

    def test(self):
        """Perform some tests"""
        print("Performing some tests on UnrealCV")
        import time
        prev_depth_image = None
        location1 = self.get_location()
        location2 = location1 + [2, 2, 0]
        self.set_location(location1, wait_until_set=True)
        for i in range(100):
            self.set_location(location2, wait_until_set=True)
            _ = self.get_depth_image()
            self.set_location(location1, wait_until_set=True)
            depth_image = self.get_depth_image()
            if prev_depth_image is not None:
                assert(np.all(depth_image == prev_depth_image))
            prev_depth_image = depth_image
            time.sleep(0.1)
