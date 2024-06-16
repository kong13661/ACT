import torch.nn as nn
import torch


class ConsistencyWrapper(nn.Module):
    def __init__(self,
                 model,
                 discriminator,
                 t_rescale=False,
                 sigma_data=0.5,
                 sigma_min=0.002,
                 sigma_max=80.,
                 rho = 7,
                 ) -> None:
        super().__init__()
        self.unet = model
        self.t_rescale = t_rescale
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.rho = rho
        self.sigma_max = sigma_max
        self._discriminator = discriminator

    def generator_training(self, training):
        ...

    def forward(self, xt, t, labels=None):
        rescaled_t = t
        if self.t_rescale:
            rescaled_t = 1000 * 0.25 * torch.log(t + 1e-44)

        c_skip, c_out, c_in, _ = self.get_scalings_for_boundary_condition(t[:, None, None, None])

        out = self.unet(c_in * xt, rescaled_t, labels)
        sample = c_out * out.sample + c_skip * xt

        return sample

    def diffusion(self, x0, t, noise):
        return x0 + noise * t[:, None, None, None]

    def discriminator(self, xt, t, labels=None):
        rescaled_t = t
        if self.t_rescale:
            rescaled_t = 1000 * 0.25 * torch.log(t + 1e-44)
        return self._discriminator(xt, rescaled_t, labels)

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        loss_weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return c_skip, c_out, c_in, loss_weight

    @torch.no_grad()
    def heun_solver(self, samples, t, next_t):
        x = samples
        t = t[:, None, None, None]
        next_t = next_t[:, None, None, None]
        denoiser = self(samples, t.reshape([-1]), output='score_match')
        d = (x - denoiser) / t
        samples = x + d * (next_t - t)
        denoiser = self(samples, next_t.reshape(-1), output='score_match')
        next_d = (samples - denoiser) / next_t
        samples = x + (d + next_d) * ((next_t - t) / 2)
        return samples

    @torch.no_grad()
    def euler_solver(self, samples, t, next_t):
        denoiser = self(samples, t.reshape([-1]), output='score_match')
        return self.euler_solver_input_score_match(samples, t, next_t, denoiser)

    @torch.no_grad()
    def euler_solver_input_score_match(self, samples, t, next_t, denoiser):
        x = samples
        t = t[:, None, None, None]
        next_t = next_t[:, None, None, None]
        d = (x - denoiser) / t
        samples = x + d * (next_t - t)
        return samples

    @torch.no_grad()
    def euler_solver_small_bin_noise_input_score_match(
            self, samples, t, next_t, x0, noise, score_match, large_bin_index):
        x = samples[large_bin_index]
        t_large_bin = t[large_bin_index]
        next_t_large_bin = next_t[large_bin_index]
        score_match_large_bin = score_match[large_bin_index]

        r = torch.empty_like(samples)
        r[~large_bin_index] = self.diffusion(
            x0[~large_bin_index], next_t[~large_bin_index], noise[~large_bin_index])

        if large_bin_index.any():
            r[large_bin_index] = self.euler_solver_input_score_match(
                x, t_large_bin, next_t_large_bin, score_match_large_bin)

        return r

    @torch.no_grad()
    def heun_solver_x0(self, samples, t, next_t):
        x = samples
        t = t[:, None, None, None]
        next_t = next_t[:, None, None, None]
        denoiser = self(samples, t.squeeze(), output='sample')
        d = (x - denoiser) / t
        samples = x + d * (next_t - t)
        denoiser = self(samples, next_t.squeeze(), output='sample')
        next_d = (samples - denoiser) / next_t
        samples = x + (d + next_d) * ((next_t - t) / 2)
        return samples

    def timesteps_to_time(self, timesteps, bins):
        timesteps = nn.functional.relu(timesteps)
        return (self.sigma_min ** (1 / self.rho) + timesteps / (bins - 1) * (
            self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho)
        )) ** self.rho

    def time_to_timesteps(self, time, bins):
        timesteps = time ** (1 / self.rho)
        timesteps = (timesteps - self.sigma_min ** (1 / self.rho)) * (bins - 1) / (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))
        return timesteps

    def prev_time(self, time, bins):
        timesteps = self.time_to_timesteps(time, bins)
        prev_timesteps = nn.functional.relu(timesteps - 1)
        return self.timesteps_to_time(prev_timesteps, bins)

    def next_time(self, time, bins):
        timesteps = self.time_to_timesteps(time, bins)
        next_timesteps = (timesteps + 1).clip(0, (bins - 1))
        return self.timesteps_to_time(next_timesteps, bins)

    def train_time_sampler(self, bins, batch_size):
        t = torch.randint(1, bins, [batch_size], device=self.device)
        t = self.timesteps_to_time(t, bins)
        return t

    def train_time_sampler_savetime(self, bins, batch_size):
        t = torch.rand(batch_size, device=self.device) * ((bins - 1) - 1)
        t = self.timesteps_to_time(t, bins)
        return t

    @property
    def device(self):
        return next(self.parameters()).device
