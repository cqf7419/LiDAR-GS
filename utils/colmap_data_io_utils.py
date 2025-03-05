import os
import struct
import numpy as np
import collections
from plyfile import PlyData
from utils.graphics_utils import BasicPointCloud

random_sample_points_number = 100000

BaseImageParaLane = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "name"])
CameraParaLane = collections.namedtuple(
    "Camera", ["model", "width", "height", "params"])

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera3DRealCar = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage3DRealCar = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


class ImageDataParaLane(BaseImageParaLane):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


class ImageData3DRealCar(BaseImage3DRealCar):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def fetchPly(path):
    plydata = PlyData.read(os.path.join(path, "pcd_rescale/sparse/0/points3D.ply"))
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T

    if positions.shape[0] > random_sample_points_number:
        indices = np.random.choice(
            positions.shape[0],
            random_sample_points_number,
            replace=True,
        )
        positions = positions[indices]

    colors = np.zeros_like(positions)
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def fetchMultiPly(path, track_list):
    positions = None
    for track_name in track_list:
        plydata = PlyData.read(os.path.join(path, track_name, "sparse/0/points3D.ply"))
        vertices = plydata["vertex"]
        curr_positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
        if positions is None:
            positions = curr_positions
        else:
            positions = np.vstack((curr_positions, positions))

    if positions.shape[0] > random_sample_points_number:
        indices = np.random.choice(
            positions.shape[0],
            random_sample_points_number,
            replace=True,
        )
        positions = positions[indices]

    colors = np.zeros_like(positions)
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def readExtrinsicsText(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                image_name = elems[9]
                images[image_id] = ImageDataParaLane(
                    id=image_id, qvec=qvec, tvec=tvec,
                    name=image_name)
    return images


def readIntrinsicsText(path):
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_channel = str(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_channel] = CameraParaLane(model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def readNextBytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def readExtrinsicsBinary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = readNextBytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = readNextBytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = readNextBytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = readNextBytes(fid, 1, "c")[0]
            num_points2D = readNextBytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = readNextBytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = ImageData3DRealCar(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def readIntrinsicsBinary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = readNextBytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = readNextBytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = readNextBytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera3DRealCar(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def debugProjections(cam_infos, pcd):
    id = 0
    for cam_info in cam_infos:
        world_to_camera_r = cam_info.R
        world_to_camera_t = cam_info.T
        intrinsic = np.eye(3)
        intrinsic[0, 0] = cam_info.K[0]
        intrinsic[1, 1] = cam_info.K[1]
        intrinsic[0, 2] = cam_info.K[2]
        intrinsic[1, 2] = cam_info.K[3]

        projections, _ = cv2.projectPoints(pcd.points, world_to_camera_r, world_to_camera_t, intrinsic, None)
        test_image = cam_info.img_mask

        for point in projections:
            cv2.circle(
                test_image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1
            )
        cv2.imwrite("test_image_" + str(id) + ".png", test_image)
        id += 1
    return    