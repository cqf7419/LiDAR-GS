from scene.GT_Dynamic_dataloader import GT_Dataloader
import numpy as np
import open3d as o3d
import os

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs2.log")) # outputs0
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

class Datas:
    def __init__(self, casename, caseid):
        self.source_path = "/mnt_gx/ziqian_data/"+casename #"/mnt_gx/ziqian_data/recon_cases_91115_default"
        self.caseid = caseid #"GT2-00007_20240402105310_20240402105440_128279484"
        self.sensorid = 0 #随便给一个


# logger = get_logger("/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/exp_bash")
lidar = ["TOP", "BACK", "LEFT", "RIGHT"]
output_path = "/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/lidar-gs/output1"
case_names = os.listdir(output_path)
print(case_names)
frame_total = 0
cd_total = []
# for casename in case_names:
#     # if casename == "recon_cases_91115_default" or casename == "recon_cases_91505_default" or casename == "recon_cases_91490_default": continue
#     # if casename == "recon_cases_91492_default" or casename == "recon_cases_91505_default" or casename == "recon_cases_91490_default": continue
#     if casename == "recon_cases_91115_default" or casename == "recon_cases_91492_default" or casename == "recon_cases_91490_default": continue
#     case_ids = os.listdir(output_path + "/" + casename)
#     for caseid in case_ids:
#         print("[ eva debug ] ", casename, caseid)

#         args = Datas(casename, caseid)
#         if os.path.exists(args.source_path + "/meta_infos/" + caseid + ".pkl") == False: continue # meta info 缺失
#         lidar_num = len(os.listdir(output_path + "/" + casename + "/" + caseid)) # 没有重建成功
#         if lidar_num !=4:
#             continue
#         if len(os.listdir(output_path + "/" + casename + "/" + caseid + "/TOP"))<4 or len(os.listdir(output_path + "/" + casename + "/" + caseid + "/BACK"))<4 \
#             or len(os.listdir(output_path + "/" + casename + "/" + caseid + "/LEFT"))<4 or len(os.listdir(output_path + "/" + casename + "/" + caseid + "/RIGHT"))<4: continue # 没有重建成功
#         GT_DATA = GT_Dataloader(args, train=False)

#         gt_pcd = []
#         cd_eachcase = []
#         cd2_eachcase = []
#         for i in range(50):
#             base_path = "/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/lidar-gs/output1/"

#             pcd_path = base_path + casename + "/" + caseid + "/" + lidar[0] + "/render_point_3000/" + "{}_gt__points.txt".format(i)
#             if os.path.exists(pcd_path):
#                 TOP_gt_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3]
#             else:
#                 print("[ Error ]",pcd_path)
#                 continue
#             pcd_path = base_path + casename + "/" + caseid + "/" + lidar[1] + "/render_point_3000/" + "{}_gt__points.txt".format(i)
#             if os.path.exists(pcd_path):
#                 BACK_gt_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3]
#             else:
#                 print("[ Error ]",pcd_path)
#                 continue
#             pcd_path = base_path + casename + "/" + caseid + "/" + lidar[2] + "/render_point_3000/" + "{}_gt__points.txt".format(i)
#             if os.path.exists(pcd_path):
#                 LEFT_gt_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3]
#             else:
#                 print("[ Error ]",pcd_path)
#                 continue
#             pcd_path = base_path + casename + "/" + caseid + "/" + lidar[3] + "/render_point_3000/" + "{}_gt__points.txt".format(i)
#             if os.path.exists(pcd_path):
#                 RIGHT_gt_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3] 
#             else:
#                 print("[ Error ]",pcd_path)
#                 continue
#             gt_pcd = [TOP_gt_pcd, BACK_gt_pcd, LEFT_gt_pcd, RIGHT_gt_pcd]

#             pcd_path = base_path + casename + "/" + caseid + "/" + lidar[0] + "/render_point_3000/" + "{}_render_points.txt".format(i)
#             TOP_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3]
#             pcd_path = base_path + casename + "/" + caseid + "/" + lidar[1] + "/render_point_3000/" + "{}_render_points.txt".format(i)
#             BACK_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3] 
#             pcd_path = base_path + casename + "/" + caseid + "/" + lidar[2] + "/render_point_3000/" + "{}_render_points.txt".format(i)
#             LEFT_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3]
#             pcd_path = base_path + casename + "/" + caseid + "/" + lidar[3] + "/render_point_3000/" + "{}_render_points.txt".format(i)
#             RIGHT_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3]

#             cd,cd2,cd3 = GT_DATA.evaluate_with_rawpcd(i, TOP_pcd, BACK_pcd, LEFT_pcd, RIGHT_pcd, rangeview_pcd = gt_pcd)
#             # print(cd,cd2,cd3) # cd1(for gt pcd) /  cd2(for gt rangeview) / diff_cd
#             frame_total = frame_total + 1
#             cd_total.append(cd.detach().cpu().numpy())
#             cd_eachcase.append(cd.detach().cpu().numpy())
#             cd2_eachcase.append(cd2.detach().cpu().numpy())
#         logger.info(f'{casename},{caseid},{np.mean(np.array(cd_eachcase))},{np.mean(np.array(cd2_eachcase))}')
# final_cd = np.mean(np.array(cd_total))
# print(final_cd)
# logger.info(f'final cd: {final_cd}, total frame: {frame_total}')

casename = "recon_cases_91115_default"
caseid = "GT2-00007_20240402105310_20240402105440_128279484"
args = Datas(casename, caseid)
GT_DATA = GT_Dataloader(args, train=False)

gt_pcd = []
cd_eachcase = []
cd2_eachcase = []
# for i in range(50):
i = 0
base_path = "/mnt_gx/usr/lansheng/workspace/LiDAR-GS-dynamic/lidar-gs/output1/"
# GT_DATA.test_pcd(i)

pcd_path = base_path + casename + "/" + caseid + "/" + lidar[0] + "/render_point_3000/" + "{}_gt__points.txt".format(i)
if os.path.exists(pcd_path):
    TOP_gt_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3]
else:
    print("[ Error ]",pcd_path)
    exit(0)
pcd_path = base_path + casename + "/" + caseid + "/" + lidar[1] + "/render_point_3000/" + "{}_gt__points.txt".format(i)
if os.path.exists(pcd_path):
    BACK_gt_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3]
else:
    print("[ Error ]",pcd_path)
    exit(0)
pcd_path = base_path + casename + "/" + caseid + "/" + lidar[2] + "/render_point_3000/" + "{}_gt__points.txt".format(i)
if os.path.exists(pcd_path):
    LEFT_gt_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3]
else:
    print("[ Error ]",pcd_path)
    exit(0)
pcd_path = base_path + casename + "/" + caseid + "/" + lidar[3] + "/render_point_3000/" + "{}_gt__points.txt".format(i)
if os.path.exists(pcd_path):
    RIGHT_gt_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3] 
else:
    print("[ Error ]",pcd_path)
    exit(0)
gt_pcd = [TOP_gt_pcd, BACK_gt_pcd, LEFT_gt_pcd, RIGHT_gt_pcd]

pcd_path = base_path + casename + "/" + caseid + "/" + lidar[0] + "/render_point_3000/" + "{}_render_points.txt".format(i)
TOP_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3]
pcd_path = base_path + casename + "/" + caseid + "/" + lidar[1] + "/render_point_3000/" + "{}_render_points.txt".format(i)
BACK_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3] 
pcd_path = base_path + casename + "/" + caseid + "/" + lidar[2] + "/render_point_3000/" + "{}_render_points.txt".format(i)
LEFT_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3]
pcd_path = base_path + casename + "/" + caseid + "/" + lidar[3] + "/render_point_3000/" + "{}_render_points.txt".format(i)
RIGHT_pcd = np.loadtxt(pcd_path, delimiter=" ", skiprows=2)[:,:3]

cd,cd2,cd3 = GT_DATA.evaluate_with_rawpcd(i, TOP_pcd, BACK_pcd, LEFT_pcd, RIGHT_pcd, rangeview_pcd = gt_pcd)
print(cd,cd2,cd3) # cd1(for gt pcd) /  cd2(for gt rangeview) / diff_cd
frame_total = frame_total + 1
# cd_total.append(cd.detach().cpu().numpy())
# cd_eachcase.append(cd.detach().cpu().numpy())
# cd2_eachcase.append(cd2.detach().cpu().numpy())
# logger.info(f'{casename},{caseid},frame_{i},{np.mean(np.array(cd_eachcase))},{np.mean(np.array(cd2_eachcase))}')

# final_cd = np.mean(np.array(cd_total))
# print(final_cd)
# logger.info(f'final cd: {final_cd}, total frame: {frame_total}')
