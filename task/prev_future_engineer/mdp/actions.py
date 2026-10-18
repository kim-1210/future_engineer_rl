from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

class CarDriveAction(ActionTerm):
    cfg: "CarDriveActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "CarDriveActionCfg", env) -> None:
        super().__init__(cfg, env)

        # 조향 joint 객체설정
        left_steering_joint_ids, left_steering_joint_names = self._asset.find_joints([cfg.left_steering_joint_name])
        right_steering_joint_ids, right_steering_joint_names = self._asset.find_joints([cfg.right_steering_joint_name])
        # 바퀴 joint 객체설정
        self._wheel_joint_ids, self._wheel_joint_names = self._asset.find_joints(cfg.wheel_joint_names)

        # joint 오류 체크
        if len(left_steering_joint_ids) != 1:
            raise ValueError(f"Left steering joint did not match exactly one joint: {cfg.left_steering_joint_name}")
        if len(right_steering_joint_ids) != 1:
            raise ValueError(f"Right steering joint did not match exactly one joint: {cfg.right_steering_joint_name}")
        if len(self._wheel_joint_ids) == 0:
            raise ValueError(f"No wheel joints matched: {cfg.wheel_joint_names}")
        if cfg.use_ackermann and len(self._wheel_joint_ids) != 2:
            raise ValueError(
                "Ackermann rear wheel velocity split requires exactly two rear wheel joints: "
                f"{cfg.wheel_joint_names} matched {self._wheel_joint_names}"
            )
        if cfg.use_ackermann:
            left_columns = [index for index, name in enumerate(self._wheel_joint_names) if "left" in name]
            right_columns = [index for index, name in enumerate(self._wheel_joint_names) if "right" in name]
            if len(left_columns) != 1 or len(right_columns) != 1:
                raise ValueError(
                    "Ackermann rear wheel velocity split requires one left and one right rear wheel joint: "
                    f"{self._wheel_joint_names}"
                )
            self._left_wheel_column = left_columns[0]
            self._right_wheel_column = right_columns[0]

        self._left_steering_joint_id = left_steering_joint_ids[0]
        self._right_steering_joint_id = right_steering_joint_ids[0]
        self._left_steering_joint_name = left_steering_joint_names[0]
        self._right_steering_joint_name = right_steering_joint_names[0]

        # 기본 행동 변수 설정
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._left_steering_target = torch.zeros(self.num_envs, 1, device=self.device)
        self._right_steering_target = torch.zeros(self.num_envs, 1, device=self.device)
        self._wheel_velocity_targets = torch.zeros(self.num_envs, len(self._wheel_joint_ids), device=self.device)
        self._wheel_velocity_signs = torch.tensor(
            cfg.wheel_velocity_signs,
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        if self._wheel_velocity_signs.shape[1] != len(self._wheel_joint_ids):
            raise ValueError(
                "wheel_velocity_signs must have the same length as wheel_joint_names: "
                f"{cfg.wheel_velocity_signs} vs {self._wheel_joint_names}"
            )

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        # action최대값 설정
        clipped_actions = torch.clamp(actions, -1.0, 1.0)
        # 조향 raw로 넣지않고 scale 적용 후
        steering = clipped_actions[:, 0:1] * self.cfg.steering_scale
        speed_alpha = (clipped_actions[:, 1:2] + 1.0) * 0.5
        # speed도 
        speed = self.cfg.min_speed + speed_alpha * (
            self.cfg.speed_scale - self.cfg.min_speed
        )

        left_steering = steering
        right_steering = steering
        wheel_velocity_targets = speed.expand(-1, len(self._wheel_joint_ids))

        # use_ackermann -> 좌우 앞바퀴 조향각과 좌우 뒷바퀴 속도를 회전반경에 맞춘다.
        if self.cfg.use_ackermann:
            abs_steering = torch.abs(steering)
            straight = abs_steering < 1.0e-4
            radius = self.cfg.wheel_base / torch.clamp(torch.tan(abs_steering), min=1.0e-4)
            inside = torch.atan2(
                torch.full_like(radius, self.cfg.wheel_base),
                torch.clamp(radius - 0.5 * self.cfg.track_width, min=1.0e-4),
            )
            outside = torch.atan2(
                torch.full_like(radius, self.cfg.wheel_base),
                radius + 0.5 * self.cfg.track_width,
            )
            inside = torch.clamp(inside, max=self.cfg.max_steering_joint_angle)
            outside = torch.clamp(outside, max=self.cfg.max_steering_joint_angle)
            sign = torch.sign(steering)
            left_steering = torch.where(steering >= 0.0, sign * inside, sign * outside)
            right_steering = torch.where(steering >= 0.0, sign * outside, sign * inside)
            left_steering = torch.where(straight, steering, left_steering)
            right_steering = torch.where(straight, steering, right_steering)

            rear_inside_ratio = torch.clamp(
                (radius - 0.5 * self.cfg.track_width) / torch.clamp(radius, min=1.0e-4),
                min=0.0,
            )
            rear_outside_ratio = (
                radius + 0.5 * self.cfg.track_width
            ) / torch.clamp(radius, min=1.0e-4)

            left_rear_speed = torch.where(
                steering >= 0.0,
                speed * rear_inside_ratio,
                speed * rear_outside_ratio,
            )
            right_rear_speed = torch.where(
                steering >= 0.0,
                speed * rear_outside_ratio,
                speed * rear_inside_ratio,
            )
            split_wheel_velocity_targets = torch.zeros_like(wheel_velocity_targets)
            split_wheel_velocity_targets[:, self._left_wheel_column : self._left_wheel_column + 1] = left_rear_speed
            split_wheel_velocity_targets[:, self._right_wheel_column : self._right_wheel_column + 1] = right_rear_speed
            wheel_velocity_targets = split_wheel_velocity_targets
            wheel_velocity_targets = torch.where(
                straight.expand_as(wheel_velocity_targets),
                speed.expand_as(wheel_velocity_targets),
                wheel_velocity_targets,
            )
        
        self._processed_actions[:, 0:1] = steering
        self._processed_actions[:, 1:2] = speed
        self._left_steering_target[:] = left_steering * self.cfg.left_steering_sign
        self._right_steering_target[:] = right_steering * self.cfg.right_steering_sign
        self._wheel_velocity_targets[:] = wheel_velocity_targets * self._wheel_velocity_signs

    # 행동 적용함수
    def apply_actions(self):
        # 양쪽 joint의 위치를 얻은 target으로 이동
        self._asset.set_joint_position_target(
            self._left_steering_target,
            joint_ids=[self._left_steering_joint_id],
        )
        self._asset.set_joint_position_target(
            self._right_steering_target,
            joint_ids=[self._right_steering_joint_id],
        )
        # 회전시 바퀴 속도설정
        self._asset.set_joint_velocity_target(
            self._wheel_velocity_targets,
            joint_ids=self._wheel_joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
            self._left_steering_target[:] = 0.0
            self._right_steering_target[:] = 0.0
            self._wheel_velocity_targets[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._left_steering_target[env_ids] = 0.0
            self._right_steering_target[env_ids] = 0.0
            self._wheel_velocity_targets[env_ids] = 0.0


@configclass
class CarDriveActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = CarDriveAction
    left_steering_joint_name: str = MISSING
    right_steering_joint_name: str = MISSING
    wheel_joint_names: list[str] = MISSING
    wheel_velocity_signs: list[float] = MISSING
    steering_scale: float = 0.5
    max_steering_joint_angle: float = 0.5236
    left_steering_sign: float = 1.0
    right_steering_sign: float = 1.0
    speed_scale: float = 12.0
    min_speed: float = 2.0
    use_ackermann: bool = True
    wheel_base: float = 0.16
    track_width: float = 0.13
