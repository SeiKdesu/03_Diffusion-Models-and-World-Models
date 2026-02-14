from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from coroutines import coroutine
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from models.rew_end_model import RewEndModel
from data import Batch

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
InitialCondition = Tuple[Tensor, Optional[Tensor], Tuple[Optional[Tensor], Optional[Tensor]]]


@dataclass
class WorldModelEnvConfig:
    horizon: int
    num_batches_to_preload: int
    diffusion_sampler_next_obs: DiffusionSamplerConfig
    diffusion_sampler_upsampling: Optional[DiffusionSamplerConfig] = None


class WorldModelEnv:
    def __init__(
        self,
        denoiser: Denoiser,
        upsampler: Optional[Denoiser],
        rew_end_model: Optional[RewEndModel],
        init_source: Union[Path, Iterable[Batch]],
        num_envs: int,
        seq_length: int,
        cfg: WorldModelEnvConfig,
        return_denoising_trajectory: bool = False,
    ) -> None:
        self.sampler_next_obs = DiffusionSampler(denoiser, cfg.diffusion_sampler_next_obs)
        self.sampler_upsampling = None if upsampler is None else DiffusionSampler(upsampler, cfg.diffusion_sampler_upsampling)
        self.rew_end_model = rew_end_model
        self.horizon = cfg.horizon
        self.return_denoising_trajectory = return_denoising_trajectory
        self.num_envs = num_envs
        self.num_actions = denoiser.cfg.inner_model.num_actions
        if isinstance(init_source, (str, Path)):
            self.generator_init = self.make_generator_init_from_spawn(Path(init_source), cfg.num_batches_to_preload)
        else:
            self.generator_init = self.make_generator_init_from_dataloader(init_source, cfg.num_batches_to_preload)
        
        self.n_skip_next_obs = seq_length - self.sampler_next_obs.denoiser.cfg.inner_model.num_steps_conditioning
        self.n_skip_upsampling = None if upsampler is None else seq_length - self.sampler_upsampling.denoiser.cfg.inner_model.num_steps_conditioning

    @property
    def device(self) -> torch.device:
        return self.sampler_next_obs.denoiser.device

    @torch.no_grad()
    def reset(self, **kwargs) -> ResetOutput:
        obs, obs_full_res, act, next_act, (hx, cx) = self.generator_init.send(self.num_envs)
        self.obs_buffer = obs
        self.act_buffer = act
        self.next_act = next_act[0]
        self.obs_full_res_buffer = obs_full_res
        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=obs.device)
        self.hx_rew_end = hx
        self.cx_rew_end = cx
        obs_to_return = self.obs_buffer[:, -1] if self.sampler_upsampling is None else self.obs_full_res_buffer[:, -1]
        return obs_to_return, {}

    @torch.no_grad()
    def step(self, act: torch.LongTensor) -> StepOutput:
        self.act_buffer[:, -1] = act

        next_obs, denoising_trajectory = self.predict_next_obs()

        if self.sampler_upsampling is not None:
            next_obs_full, denoising_trajectory_upsampling = self.upsample_next_obs(next_obs)

        if self.rew_end_model is not None:
            rew, end = self.predict_rew_end(next_obs.unsqueeze(1))
        else:
            rew = torch.zeros(next_obs.size(0), dtype=torch.float32, device=self.device)
            end = torch.zeros(next_obs.size(0), dtype=torch.int64, device=self.device)
        
        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()

        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs

        if self.sampler_upsampling is not None:
            self.obs_full_res_buffer = self.obs_full_res_buffer.roll(-1, dims=1)
            self.obs_full_res_buffer[:, -1] = next_obs_full

        info = {}
        dead = torch.logical_or(end.bool(), trunc.bool())
        if dead.any():
            if self.sampler_upsampling is None:
                final_obs = next_obs[dead]
            else:
                final_obs = next_obs_full[dead]

            obs_new, obs_full_res_new, act_new, next_act_new, (hx_new, cx_new) = self.generator_init.send(int(dead.sum()))

            self.obs_buffer[dead] = obs_new
            self.act_buffer[dead] = act_new
            if self.sampler_upsampling is not None:
                self.obs_full_res_buffer[dead] = obs_full_res_new

            self.ep_len[dead] = 0
            if self.rew_end_model is not None:
                self.hx_rew_end[:, dead] = hx_new
                self.cx_rew_end[:, dead] = cx_new

            info["final_observation"] = final_obs
            if self.sampler_upsampling is None:
                next_obs[dead] = obs_new[:, -1]
                info["burnin_obs"] = obs_new[:, :-1]
            else:
                next_obs_full[dead] = obs_full_res_new[:, -1]
                next_obs[dead] = obs_new[:, -1]
                info["burnin_obs"] = obs_full_res_new[:, :-1]

        if self.return_denoising_trajectory:
            info["denoising_trajectory"] = torch.stack(denoising_trajectory, dim=1)
            
        if self.sampler_upsampling is not None:
            info["obs_low_res"] = next_obs
            if self.return_denoising_trajectory:
                info["denoising_trajectory_upsampling"] = torch.stack(denoising_trajectory_upsampling, dim=1)

        obs_to_return = self.obs_buffer[:, -1] if self.sampler_upsampling is None else self.obs_full_res_buffer[:, -1]
        return obs_to_return, rew, end, trunc, info

    @torch.no_grad()
    def predict_next_obs(self) -> Tuple[Tensor, List[Tensor]]:
        return self.sampler_next_obs.sample(self.obs_buffer[:, self.n_skip_next_obs:], self.act_buffer[:, self.n_skip_next_obs:])

    @torch.no_grad()
    def upsample_next_obs(self, next_obs: Tensor) -> Tuple[Tensor, List[Tensor]]:
        low_res = F.interpolate(next_obs, scale_factor=self.sampler_upsampling.denoiser.cfg.upsampling_factor, mode="bicubic").unsqueeze(1)
        return self.sampler_upsampling.sample(torch.cat((self.obs_full_res_buffer[:, self.n_skip_upsampling:], low_res), dim=1), None)

    @torch.no_grad()
    def predict_rew_end(self, next_obs: Tensor) -> Tuple[Tensor, Tensor]:
        logits_rew, logits_end, (self.hx_rew_end, self.cx_rew_end) = self.rew_end_model.predict_rew_end(
            self.obs_buffer[:, -1:],
            self.act_buffer[:, -1:],
            next_obs,
            (self.hx_rew_end, self.cx_rew_end),
        )
        rew = Categorical(logits=logits_rew).sample().squeeze(1) - 1.0  # in {-1, 0, 1}
        end = Categorical(logits=logits_end).sample().squeeze(1)
        return rew, end

    @coroutine
    def make_generator_init_from_spawn(
        self,
        spawn_dir: Path,
        num_batches_to_preload: int,
    ) -> Generator[InitialCondition, None, None]:
        num_dead = yield

        spawn_dirs = cycle(sorted(list(spawn_dir.iterdir())))

        while True:
            # Preload on device and burnin rew/end model
            obs_, obs_full_res_, act_, next_act_, hx_, cx_ = [], [], [], [], [], []
            for _ in range(num_batches_to_preload):
                d = next(spawn_dirs)
                obs = torch.tensor(np.load(d / "low_res.npy"), device=self.device).div(255).mul(2).sub(1).unsqueeze(0)
                obs_full_res = torch.tensor(np.load(d / "full_res.npy"), device=self.device).div(255).mul(2).sub(1).unsqueeze(0)
                act = torch.tensor(np.load(d / "act.npy"), dtype=torch.long, device=self.device).unsqueeze(0)
                next_act = torch.tensor(np.load(d / "next_act.npy"), dtype=torch.long, device=self.device).unsqueeze(0)

                obs_.extend(list(obs))
                obs_full_res_.extend(list(obs_full_res))
                act_.extend(list(act))
                next_act_.extend(list(next_act))

                if self.rew_end_model is not None:
                    with torch.no_grad():
                        *_, (hx, cx) = self.rew_end_model.predict_rew_end(obs_[:, :-1], act[:, :-1], obs[:, 1:])  # Burn-in of rew/end model
                    assert hx.size(0) == cx.size(0) == 1
                    hx_.extend(list(hx[0]))
                    cx_.extend(list(cx[0]))

            # Yield new initial conditions for dead envs
            c = 0
            while c + num_dead <= len(obs_):
                obs = torch.stack(obs_[c : c + num_dead])
                act = torch.stack(act_[c : c + num_dead])
                next_act = next_act_[c : c + num_dead]
                obs_full_res = torch.stack(obs_full_res_[c : c + num_dead]) if self.sampler_upsampling is not None else None
                hx = torch.stack(hx_[c : c + num_dead]).unsqueeze(0) if self.rew_end_model is not None else None
                cx = torch.stack(cx_[c : c + num_dead]).unsqueeze(0) if self.rew_end_model is not None else None
                c += num_dead
                num_dead = yield obs, obs_full_res, act, next_act, (hx, cx)

    @coroutine
    def make_generator_init_from_dataloader(
        self,
        data_loader: Iterable[Batch],
        num_batches_to_preload: int,
    ) -> Generator[InitialCondition, None, None]:
        num_dead = yield

        data_iter = iter(data_loader)
        obs_pool: List[Tensor] = []
        obs_full_res_pool: List[Tensor] = []
        act_pool: List[Tensor] = []
        next_act_pool: List[Tensor] = []
        hx_pool: List[Tensor] = []
        cx_pool: List[Tensor] = []

        while True:
            while len(obs_pool) < num_dead:
                for _ in range(num_batches_to_preload):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(data_loader)
                        batch = next(data_iter)

                    batch = batch.to(self.device)
                    obs = batch.obs
                    act = batch.act

                    if self.sampler_upsampling is not None:
                        if len(batch.info) == 0 or "full_res" not in batch.info[0]:
                            raise ValueError("upsampler requires batch.info[*]['full_res'] to be present")
                        obs_full_res = torch.stack([info["full_res"] for info in batch.info]).to(self.device)
                    else:
                        obs_full_res = None

                    if self.rew_end_model is not None:
                        with torch.no_grad():
                            _, _, (hx, cx) = self.rew_end_model.predict_rew_end(obs[:, :-1], act[:, :-1], obs[:, 1:])
                        hx = hx[0]
                        cx = cx[0]
                    else:
                        hx = cx = None

                    for i in range(obs.size(0)):
                        obs_pool.append(obs[i])
                        act_pool.append(act[i])
                        next_act_pool.append(act[i, -1])
                        if obs_full_res is not None:
                            obs_full_res_pool.append(obs_full_res[i])
                        if hx is not None and cx is not None:
                            hx_pool.append(hx[i])
                            cx_pool.append(cx[i])

            obs = torch.stack(obs_pool[:num_dead])
            act = torch.stack(act_pool[:num_dead])
            next_act = torch.stack(next_act_pool[:num_dead])
            obs_full_res = (
                torch.stack(obs_full_res_pool[:num_dead]) if self.sampler_upsampling is not None else None
            )
            if self.rew_end_model is not None:
                hx = torch.stack(hx_pool[:num_dead]).unsqueeze(0)
                cx = torch.stack(cx_pool[:num_dead]).unsqueeze(0)
            else:
                hx = cx = None

            obs_pool = obs_pool[num_dead:]
            act_pool = act_pool[num_dead:]
            next_act_pool = next_act_pool[num_dead:]
            if self.sampler_upsampling is not None:
                obs_full_res_pool = obs_full_res_pool[num_dead:]
            if self.rew_end_model is not None:
                hx_pool = hx_pool[num_dead:]
                cx_pool = cx_pool[num_dead:]

            num_dead = yield obs, obs_full_res, act, next_act, (hx, cx)
