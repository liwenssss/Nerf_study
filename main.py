import numpy as np
import torch
import torch.backends.cudnn
from tqdm import tqdm
import matplotlib.pyplot as plt

from net import NeRF
from utils import sample_rays_np
from utils import render_rays

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(999)
    np.random.seed(666)
    torch.backends.cudnn.benchmark = True

    #############################
    # data preprocess
    #############################
    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    H, W = images.shape[1:3]
    print("images.shape:", images.shape)
    print("poses.shape:", poses.shape)
    print("focal:", focal)

    n_train = 100
    test_img, test_pose = images[101], poses[101]
    images = images[:n_train]
    poses = poses[:n_train]

    plt.imshow(test_img)
    plt.show()

    print("Process rays data!")
    rays_o_list = list()
    rays_d_list = list()
    rays_rgb_list = list()

    for i in range(n_train):
        img = images[i]
        pose = poses[i]
        rays_o, rays_d = sample_rays_np(H, W, focal, pose)
        print(img.shape, pose.shape, rays_o.shape, rays_d.shape)

        rays_o_list.append(rays_o.reshape(-1, 3))
        rays_d_list.append(rays_d.reshape(-1, 3))
        rays_rgb_list.append(img.reshape(-1, 3))

    rays_o_npy = np.concatenate(rays_o_list, axis=0)
    rays_d_npy = np.concatenate(rays_d_list, axis=0)
    rays_rgb_npy = np.concatenate(rays_rgb_list, axis=0)
    rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy, rays_rgb_npy], axis=1), device=device)

    #############################
    # training parameters
    #############################
    N = rays.shape[0]
    Batch_size = 4096
    iterations = N // Batch_size
    print(f"There are {iterations} batches of rays and each batch contains {Batch_size} rays")

    bound = (2., 6.)
    N_samples = (64, None)
    use_view = True
    epoch = 10
    psnr_list = []
    e_nums = []

    #############################
    # test data
    #############################
    test_rays_o, test_rays_d = sample_rays_np(H, W, focal, test_pose)
    test_rays_o = torch.tensor(test_rays_o, device=device)
    test_rays_d = torch.tensor(test_rays_d, device=device)
    test_rgb = torch.tensor(test_img, device=device)

    #############################
    # training
    #############################
    net = NeRF(use_view_dirs=use_view).to(device)
    optimizer = torch.optim.Adam(net.parameters(), 5e-4)
    mse = torch.nn.MSELoss()

    for e in range(epoch):
        # create iteration for training
        rays = rays[torch.randperm(N), :]
        train_iter = iter(torch.split(rays, Batch_size, dim=0))

        # render + mse
        with tqdm(total=iterations, desc=f"Epoch {e + 1}", ncols=100) as p_bar:
            for i in range(iterations):
                train_rays = next(train_iter)
                assert train_rays.shape == (Batch_size, 9)

                rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
                rays_od = (rays_o, rays_d)
                rgb, _, __ = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device,
                                         use_view=use_view)

                loss = mse(rgb, target_rgb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                p_bar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
                p_bar.update(1)

        with torch.no_grad():
            rgb_list = list()
            for j in range(test_rays_o.shape[0]):
                rays_od = (test_rays_o[j], test_rays_d[j])
                rgb, _, __ = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device,
                                         use_view=use_view)
                rgb_list.append(rgb.unsqueeze(0))
            rgb = torch.cat(rgb_list, dim=0)
            loss = mse(rgb, torch.tensor(test_img, device=device)).cpu()
            psnr = -10. * torch.log(loss).item() / torch.log(torch.tensor([10.]))
            print(f"PSNR={psnr.item()}")
            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(rgb.cpu().detach().numpy())
            plt.title(f'Epoch: {e + 1}')
            plt.subplot(122)

            e_nums.append(e + 1)
            psnr_list.append(psnr.numpy())
            plt.plot(e_nums, psnr_list)
            plt.title('PSNR')
            plt.show()

    print('Done')

if __name__ == '__main__':
    main()