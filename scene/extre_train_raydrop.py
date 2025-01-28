import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np
import os
from tqdm import tqdm
import imageio
import cv2

def cosine_scheduler(base_value,
                     final_value,
                     globel_step,
                     warmup_iters=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(globel_step - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == globel_step
    return schedule

class raydrop(nn.Module):
    def __init__(self):
        super(raydrop, self).__init__()
        self.enc_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "degree": 4,
            },
        ).cuda()
        # self.enc_pos = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={
        #         "otype": "Frequency",
        #         "degree": 4,
        #     },
        # ).cuda()
        self.enc_i_d = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "Frequency",
                "degree": 6,
            },
        ).cuda()
        # self.unet = RayDrop(D=4, W=128, input_ch=self.enc_dir.n_output_dims+self.enc_i_d.n_output_dims, output_ch=1).cuda()#UNet(in_channels=3, out_channels=1).cuda()
        self.unet = tcnn.Network(
            n_input_dims=self.enc_dir.n_output_dims+self.enc_i_d.n_output_dims, #+self.enc_pos.n_output_dims
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 128,
                "n_hidden_layers": 4,
            },
        )
    def forward(self, dir, intensity, depth):
        enc_i_d = self.enc_i_d(torch.cat((intensity, depth),dim=1))
        enc_dir = self.enc_dir(dir)
        render_raydrop = self.unet(torch.cat((enc_dir,enc_i_d),dim=1))
        return render_raydrop
        # def forward(self,x):
            # enc_i_d_input = gaussians.get_enc_i_d(torch.cat((image[0].reshape(-1,1).detach(), depth[0].reshape(-1,1).detach()),dim=1))
            # enc_dir_input = gaussians.get_enc_dir(viewpoint_cam.ray_dir.cuda())
            # enc_pos_input = gaussians.get_enc_pos(viewpoint_cam.lidar_center.cuda()).repeat(viewpoint_cam.image_height*viewpoint_cam.image_width,1)
            # render_raydrop = gaussians.unet(torch.cat((enc_dir_input,enc_i_d_input,enc_pos_input),dim=1)).reshape(viewpoint_cam.image_height,viewpoint_cam.image_width)

def load(path):

    dir = np.load(path+"/train/"+"dir.npy")
    render_depth_train = []
    render_intensity_train = []
    gt_raydrop_train = []
    for i in range(47):
        render_depth = np.load(path+"/train/"+"depth_{}".format(i)+".npy")
        render_depth_train.append(render_depth)
        render_intensity =  np.load(path+"/train/"+"intensity_{}".format(i)+".npy")
        render_intensity_train.append(render_intensity)
        gt_raydrop =  np.load(path+"/train/"+"gt_raydrop_{}".format(i)+".npy")
        gt_raydrop_train.append(gt_raydrop)

    render_depth_test = []
    render_intensity_test = []
    gt_raydrop_test = []
    for i in range(3):
        render_depth = np.load(path+"/test/"+"depth_{}".format(i)+".npy")
        render_depth_test.append(render_depth)
        render_intensity =  np.load(path+"/test/"+"intensity_{}".format(i)+".npy")
        render_intensity_test.append(render_intensity)
        gt_raydrop =  np.load(path+"/test/"+"gt_raydrop_{}".format(i)+".npy")
        gt_raydrop_test.append(gt_raydrop)

    return dir, render_depth_train, render_intensity_train, gt_raydrop_train, render_depth_test, render_intensity_test, gt_raydrop_test

def train():
    dataset_path = "/data/usr/lansheng/workspace/lidar-gs/render_kitti_train_drop/1908"
    (
        dir, 
        render_depth_train, 
        render_intensity_train, 
        gt_raydrop_train, 
        render_depth_test, 
        render_intensity_test, 
        gt_raydrop_test
    ) = load(dataset_path)
    print("[prepare]: load data.")
    print(dir.shape)
    model = raydrop().cuda()
    grad_vars = list(model.parameters())
    optimizer = torch.optim.Adam(params=grad_vars,
                                 lr=5e-4,
                                 betas=(0.9, 0.999))

    N_iters = 10000
    epochs = 100
    lr_schedule = cosine_scheduler(
        base_value=5e-4,
        final_value=5e-5,
        globel_step=N_iters - 1,
        warmup_iters=1000,
    )
    mse_loss = torch.nn.MSELoss()
    print("[prepare]: set opt and sch.")

    steps = 0
    progress_bar = tqdm(range(0, N_iters), desc="Training progress")
    ema_loss = 0
    for epoch in range(1,epochs):
        dir_train = torch.tensor(dir, dtype=torch.float32).cuda()
        for i in range(len(render_depth_train)):
            steps +=1
            render_depth = torch.tensor(render_depth_train[i],  dtype=torch.float32).reshape(-1,1).cuda()
            render_intensity = torch.tensor(render_intensity_train[i], dtype=torch.float32).reshape(-1,1).cuda()
            gt_raydrop = torch.tensor(gt_raydrop_train[i], dtype=torch.float32).reshape(-1,1).cuda()

            #forward
            render_raydrop = model(dir_train, render_intensity, render_depth)

            #loss
            loss = mse_loss(render_raydrop, gt_raydrop)

            loss.backward()
            optimizer.step()
            decay_rate = 0.1
            new_lrate = 5e-4 * (decay_rate**(steps / N_iters))
            for param_group in optimizer.param_groups:
                if False:
                    param_group["lr"] = lr_schedule[steps]
                else:
                    param_group['lr'] = new_lrate
            optimizer.zero_grad(set_to_none = True)

            ema_loss = 0.6*ema_loss + 0.4*loss.item()
            if steps % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss:.{7}f}",
                    })
                progress_bar.update(10)


        if epoch % 5 == 0:  ## val  
            val_loss = 0
            for i in range(len(render_intensity_test)):
                render_depth = torch.tensor(render_depth_test[i],  dtype=torch.float32).reshape(-1,1).cuda()
                render_intensity = torch.tensor(render_intensity_test[i], dtype=torch.float32).reshape(-1,1).cuda()
                gt_raydrop = torch.tensor(gt_raydrop_test[i], dtype=torch.float32).reshape(-1,1).cuda()

                #forward
                with torch.no_grad():
                    render_raydrop = model(dir_train, render_intensity, render_depth)
                    val_loss += mse_loss(render_raydrop, gt_raydrop)
                    render_raydrop = render_raydrop.reshape(66,1030)
                    render_raydrop_mask = torch.where(render_raydrop > 0.5, 1, 0)*255.0
                    imageio.imwrite('/data/usr/lansheng/workspace/lidar-gs/outputs/kitti1908_raydrop/test_raydrop_{}.png'.format(i),render_raydrop_mask.detach().cpu().numpy().astype(np.uint8) )
            
            val_loss /= len(render_intensity_test)
            print("{} epoch, val loss = ".format(epoch), val_loss)
        if epoch % 20 == 0:
            basedir = "/data/usr/lansheng/workspace/lidar-gs/outputs/kitti1908_raydrop"
            os.makedirs(basedir, exist_ok=True)
            path = os.path.join(basedir, '{}.pth'.format(epoch))
            ckpt = {
                'global_step': steps,
                'network_fn_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(ckpt, path)
            print('Saved checkpoints at', path)
        
def load_model(ckpt_path, optimizer, model):
    print("Reloading from", ckpt_path)
    ckpt = torch.load(ckpt_path)

    start = ckpt["global_step"]
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # Load model
    model.load_state_dict(ckpt["network_fn_state_dict"])



if __name__ == "__main__":
    train()