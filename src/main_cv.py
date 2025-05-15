import argparse
import os
from math import sqrt

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from src.pytorch_fid.fid_score import calculate_fid_given_paths
from src.score_sde.models.ncsnpp_generator_adagn import NCSNpp


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():

    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, m, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise * sqrt(m)

    return x_t


class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)  # .to(x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x


def sample_from_model_gibbs(diff_coef, post_coef, generator, k, y, T, opt, x_prev):
    t_time = torch.full((y.size(0),), opt.fixed_t * T, dtype=torch.int64).to(y.device)

    with torch.no_grad():
        y_mean = torch.mean(y, dim=1)
        latent_z = torch.randn(y_mean.size(0), opt.nz, device=y_mean.device)  # .to(y_mean.device)
        x = generator(y_mean, t_time, latent_z)
        x = sample_posterior(post_coef, x, y_mean, t_time)
        y[:, k] = q_sample(diff_coef, x, t_time, opt.m)

        t_time -= 1
        while t_time[0] >= 0:
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)  # .to(x.device)
            x_new = generator(x, t_time, latent_z)
            x = sample_posterior(post_coef, x_new, x, t_time)
            t_time -= 1
        x = x.detach()

    return x, y


def main():
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                        help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int, default=1000)
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy',
                        help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')

    # Gibbs
    parser.add_argument('--fixed_t', type=float, default=0.1, help='Gibbs fixed noise level')
    parser.add_argument('--n', type=int, default=10, help='Gibbs rounds')
    parser.add_argument('--m', type=int, default=100, help='Gibbs ensemble size')
    parser.add_argument('--sample_freq', type=int, default=10, help='Gibbs sampling frequency')

    args = parser.parse_args()

    os.chdir("..")
    torch.manual_seed(42)
    device = 'cuda:0'

    netG = NCSNpp(args).to(device)
    ckpt = torch.load('checkpoints/dd_gan/{}/{}/netG_{}.pth'.format(args.dataset, args.exp, args.epoch_id),
                      map_location=device)

    # loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()

    to_range_0_1 = lambda x: (x + 1.) / 2.

    if args.dataset == 'cifar10':
        real_img_dir = 'src/pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'src/pytorch_fid/celebahq_stat.npy'
    else:
        real_img_dir = args.real_img_dir

    diff_coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    save_dir = "./generated_samples/{}".format(args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("./generated_chains/{}".format(args.dataset), exist_ok=True)

    with torch.no_grad():
        chain_length = args.n * args.m // args.sample_freq
        samples_needed = 50000 // chain_length
        samples_left = samples_needed
        chain_idx = 0
        img_idx = 0
        while samples_left > 0:
            print("Samples left: {}".format(samples_left * chain_length))
            samples_to_return = min(args.batch_size, samples_left)
            y = sqrt(args.m) * torch.randn(samples_to_return, args.m, args.num_channels, args.image_size,
                                           args.image_size).to(device)

            sample_batches = []
            fake_sample = None
            for i in range(args.n):
                for k in range(args.m):
                    fake_sample, y = sample_from_model_gibbs(diff_coeff, pos_coeff, netG, k, y, args.num_timesteps,
                                                             args, fake_sample)

                    if (k + 1) % args.sample_freq == 0:
                        tqdm.write(f'[{i + 1} / {args.n}]: {k + 1}/{args.m}')
                        with torch.no_grad():
                            t_time = torch.full((y.size(0),),
                                                args.fixed_t * args.num_timesteps, dtype=torch.int64).to(y.device)
                            y_mean = torch.mean(y, dim=1)
                            latent_z = torch.randn(y_mean.size(0), args.nz, device=y_mean.device)  # .to(y_mean.device)
                            fake_sample = netG(y_mean, t_time, latent_z)
                            fake_sample = sample_posterior(pos_coeff, fake_sample, y_mean, t_time)
                            t_time -= 1
                            while t_time[0] >= 0:
                                latent_z = torch.randn(fake_sample.size(0), args.nz,
                                                       device=fake_sample.device)  # .to(fake_sample.device)
                                fake_sample_new = netG(fake_sample, t_time, latent_z)
                                fake_sample = sample_posterior(pos_coeff, fake_sample_new, fake_sample, t_time)
                                t_time -= 1
                            fake_sample = fake_sample.detach()
                        sample_batches.append(to_range_0_1(fake_sample))
            sample_batches = torch.stack(sample_batches, dim=1)
            samples_left -= samples_to_return
            for i in range(samples_to_return):
                torchvision.utils.save_image(sample_batches[i],
                                             './generated_chains/{}/{}.jpg'.format(args.dataset, chain_idx), nrow=10)
                for j in range(chain_length):
                    torchvision.utils.save_image(sample_batches[i, j],
                                                 './generated_samples/{}/{}.jpg'.format(args.dataset, img_idx))
                    img_idx += 1
                chain_idx += 1
        paths = [save_dir, real_img_dir]

        kwargs = {'batch_size': 100 if args.dataset == 'cifar10' else 10, 'device': device, 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))


if __name__ == '__main__':
    main()
