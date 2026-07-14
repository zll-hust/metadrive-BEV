#!/usr/bin/env python
"""Generate LeWM-style fixed-camera BEV clips from diverse MetaDrive maps.

The ego vehicle and traffic vehicles are controlled by MetaDrive's IDM policy.
The BEV camera is fixed at a road-valid midpoint of the ego route, so vehicles
move across the video instead of remaining centered in the image. A strict map
bounding-box center mode is also available. All vehicles are assigned a
non-zero lane-aligned velocity before the first simulation step. The defaults
produce 100 RGB uint8 frames at 224x224 and 10 Hz (a 10-second episode).

Example:
    python -m metadrive.examples.generate_diverse_bev_videos \
        --output-dir bev_dataset --variants 2
"""

import argparse
import json
from pathlib import Path

import mediapy
import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.envs import MetaDriveEnv
from metadrive.manager.object_manager import TrafficObjectManager
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.traffic_manager import PGTrafficManager, TrafficMode
from metadrive.policy.idm_policy import IDMPolicy


# A first straight block is automatically prepended by MetaDrive.  Keeping the
# first block short makes the distinguishing road geometry visible early.
SCENES = {
    "highway": {
        "map": "SSS",
        "lane_num": 4,
        "exit_length": 25,
        "traffic_density": 0.16,
    },
    "curve": {
        "map": "CCS",
        "lane_num": 2,
        "exit_length": 18,
        "traffic_density": 0.18,
    },
    "intersection": {
        "map": "XSS",
        "lane_num": 2,
        "exit_length": 18,
        "traffic_density": 0.16,
    },
    "ramp": {
        "map": "rSS",
        "lane_num": 3,
        "exit_length": 22,
        "traffic_density": 0.18,
    },
    "roundabout": {
        "map": "OSS",
        "lane_num": 2,
        "exit_length": 18,
        "traffic_density": 0.16,
    },
}

PHYSICS_DT = 0.02
EGO_SEMANTIC_COLOR = (0, 255, 0)
TRAFFIC_SEMANTIC_COLOR = (255, 255, 0)
TRAFFIC_SPAWN_GAP_M = 20


class DatasetTrafficManager(PGTrafficManager):
    """Generate traffic at safer spacing without changing MetaDrive globally."""

    VEHICLE_GAP = TRAFFIC_SPAWN_GAP_M


class FixedLengthMetaDriveEnv(MetaDriveEnv):
    """Keep the ego alive so every generated episode has exactly N steps."""

    def setup_engine(self):
        # MetaDriveEnv hard-codes PGTrafficManager, so register our dataset
        # variant here to give initially spawned vehicles a safer headway.
        super(MetaDriveEnv, self).setup_engine()
        self.engine.register_manager("map_manager", PGMapManager())
        self.engine.register_manager("traffic_manager", DatasetTrafficManager())
        if abs(self.config["accident_prob"]) > 1e-2:
            self.engine.register_manager("object_manager", TrafficObjectManager())

    def done_function(self, vehicle_id):
        natural_done, done_info = super().done_function(vehicle_id)
        done_info["natural_termination"] = natural_done
        return False, done_info


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("bev_dataset"))
    parser.add_argument("--frames", type=int, default=100, help="Frames per video (default: 100).")
    parser.add_argument("--fps", type=int, default=10, help="Environment steps and MP4 frames per second.")
    parser.add_argument("--variants", type=int, default=1, help="Seeds per scene type.")
    parser.add_argument("--seed", type=int, default=1000, help="First random seed.")
    parser.add_argument("--size", type=int, default=224, help="Square RGB video resolution in pixels.")
    parser.add_argument(
        "--camera-mode",
        choices=("scene_core", "route_midpoint", "map_center"),
        default="scene_core",
        help="Fixed camera placement. scene_core focuses the intersection/roundabout/target road block.",
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=2.0,
        help="BEV pixels per meter (default: 2).",
    )
    parser.add_argument("--ego-initial-speed", type=float, default=25.0, help="Initial ego speed in km/h.")
    parser.add_argument(
        "--traffic-min-speed", type=float, default=22.0, help="Minimum initial traffic speed in km/h."
    )
    parser.add_argument(
        "--traffic-max-speed", type=float, default=28.0, help="Maximum initial traffic speed in km/h."
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=tuple(SCENES),
        default=list(SCENES),
        help="Scene types to generate. The default is all scene types.",
    )
    parser.add_argument("--no-semantic", dest="semantic", action="store_false", help="Disable semantic map colors.")
    parser.set_defaults(semantic=True)
    return parser.parse_args()


def validate_args(args):
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.variants <= 0:
        raise ValueError("--variants must be positive")
    if args.fps <= 0 or args.size <= 0 or args.scaling <= 0:
        raise ValueError("--fps, --size and --scaling must be positive")
    if args.ego_initial_speed <= 0 or args.traffic_min_speed <= 0:
        raise ValueError("Initial vehicle speeds must be positive")
    if args.traffic_max_speed < args.traffic_min_speed:
        raise ValueError("--traffic-max-speed must be >= --traffic-min-speed")


def make_env(scene, seed, fps):
    decision_repeat = round(1.0 / (fps * PHYSICS_DT))
    actual_fps = 1.0 / (decision_repeat * PHYSICS_DT)
    if not np.isclose(actual_fps, fps):
        raise ValueError(
            f"Requested {fps} Hz can not be represented exactly with physics_dt={PHYSICS_DT}; "
            f"nearest rate is {actual_fps:.6g} Hz."
        )
    return FixedLengthMetaDriveEnv(
        {
            "start_seed": seed,
            "num_scenarios": 1,
            "map_config": {
                "type": "block_sequence",
                "config": scene["map"],
                "lane_num": scene["lane_num"],
                "exit_length": scene["exit_length"],
            },
            "traffic_density": scene["traffic_density"],
            # Trigger creates each traffic vehicle once. We immediately activate
            # them below, but never respawn replacements at arbitrary locations.
            "traffic_mode": TrafficMode.Trigger,
            "random_traffic": True,
            "need_inverse_traffic": True,
            "agent_policy": IDMPolicy,
            "random_agent_model": True,
            "traffic_vehicle_config": {
                "show_navi_mark": False,
                "show_dest_mark": False,
                "show_lidar": False,
                "show_lane_line_detector": False,
                "show_side_detector": False,
            },
            "vehicle_config": {
                "show_navi_mark": False,
                "show_dest_mark": False,
                "show_navigation_arrow": False,
                "show_lidar": False,
                "top_down_width": 2.5,
                "top_down_length": 5.0,
            },
            "physics_world_step_size": PHYSICS_DT,
            "decision_repeat": decision_repeat,
            "use_render": False,
            "show_terrain": False,
            "horizon": None,
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "out_of_road_done": False,
            "on_continuous_line_done": False,
            "on_broken_line_done": False,
            "log_level": 30,
        }
    )


def activate_all_traffic(env):
    """Activate all one-shot traffic immediately so IDM controls it from step zero."""
    manager = env.engine.traffic_manager
    active_ids = {vehicle.id for vehicle in manager._traffic_vehicles}
    for block_vehicles in manager.block_triggered_vehicles:
        for vehicle in manager.get_objects(block_vehicles.vehicles).values():
            if vehicle.id not in active_ids:
                manager._traffic_vehicles.append(vehicle)
                active_ids.add(vehicle.id)
    manager.block_triggered_vehicles.clear()


def initialize_vehicle_speeds(env, seed, ego_speed_km_h, traffic_speed_range):
    """Give every existing vehicle a lane-aligned velocity before frame zero."""
    activate_all_traffic(env)
    rng = np.random.RandomState(seed + 7919)
    assigned_speeds = {}
    for obj in env.engine.get_objects().values():
        if not isinstance(obj, BaseVehicle):
            continue
        speed_km_h = (
            ego_speed_km_h if obj is env.agent else rng.uniform(traffic_speed_range[0], traffic_speed_range[1])
        )
        obj.set_velocity_km_h(obj.heading, value=speed_km_h, in_local_frame=False)
        assigned_speeds[obj.name] = float(speed_km_h)
    return assigned_speeds


def traffic_stats(env, radius_m, initial_traffic_ids):
    ego_position = np.asarray(env.agent.position)
    traffic = []
    nearby_speeds = []
    for obj in env.engine.get_objects().values():
        if isinstance(obj, BaseVehicle) and obj is not env.agent:
            traffic.append(obj)
            if np.linalg.norm(np.asarray(obj.position) - ego_position) <= radius_m:
                nearby_speeds.append(float(obj.speed_km_h))
    current_ids = {vehicle.id for vehicle in traffic}
    return {
        "traffic_vehicle_count": len(traffic),
        "new_traffic_vehicle_count": len(current_ids - initial_traffic_ids),
        "crashed_traffic_vehicle_count": sum(vehicle.crash_vehicle for vehicle in traffic),
        "stopped_traffic_vehicle_count": sum(
            vehicle.speed_km_h <= 1.0 and not vehicle.crash_vehicle for vehicle in traffic
        ),
        "nearby_vehicle_count": len(nearby_speeds),
        "moving_nearby_vehicle_count": sum(speed > 1.0 for speed in nearby_speeds),
        "mean_nearby_speed_km_h": float(np.mean(nearby_speeds)) if nearby_speeds else 0.0,
    }


def plan_fixed_camera(env, args, seed, scene_name):
    """Preview the route and choose a fixed scene-core or route-midpoint camera."""
    positions = [np.asarray(env.agent.position, dtype=float)]
    for _ in range(args.frames):
        env.step([0.0, 0.0])
        positions.append(np.asarray(env.agent.position, dtype=float))

    positions = np.stack(positions)
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative_distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    midpoint_index = int(np.searchsorted(cumulative_distance, cumulative_distance[-1] / 2))
    route_midpoint = positions[min(midpoint_index, len(positions) - 1)]

    target_block = env.current_map.blocks[1]
    block_bbox = target_block.block_network.get_bounding_box()
    scene_core = np.asarray(
        [(block_bbox[0] + block_bbox[1]) / 2, (block_bbox[2] + block_bbox[3]) / 2], dtype=float
    )
    if args.camera_mode == "route_midpoint":
        camera_position = route_midpoint
    elif scene_name in {"intersection", "roundabout"}:
        # Their geometric center is the semantic core that should dominate the crop.
        camera_position = scene_core
    else:
        # Curves and ramps can have an empty bounding-box center. Project it to
        # the closest point actually traversed by the ego to keep road in view.
        nearest_index = int(np.argmin(np.linalg.norm(positions - scene_core, axis=1)))
        camera_position = positions[nearest_index]

    half_view = args.size / args.scaling / 2
    visibility_margin = 4.0
    effective_half_view = half_view - visibility_margin
    if args.camera_mode == "scene_core":
        # Shift away from the scene core only as much as needed to fit the full
        # ten-second ego trajectory. This keeps an intersection prominent while
        # avoiding a few missing ego frames near the start or end of the clip.
        lower_camera_bound = positions[1:].max(axis=0) - effective_half_view
        upper_camera_bound = positions[1:].min(axis=0) + effective_half_view
        if np.all(lower_camera_bound <= upper_camera_bound):
            camera_position = np.clip(camera_position, lower_camera_bound, upper_camera_bound)

    visible = np.all(np.abs(positions[1:] - camera_position) <= effective_half_view, axis=1)
    predicted_visibility_ratio = float(np.mean(visible))

    env.reset(seed=seed)
    env.engine.force_fps.disable()
    assigned_speeds = initialize_vehicle_speeds(
        env,
        seed,
        ego_speed_km_h=args.ego_initial_speed,
        traffic_speed_range=(args.traffic_min_speed, args.traffic_max_speed),
    )
    return {
        "camera_position": camera_position,
        "scene_core": scene_core,
        "target_block_id": target_block.ID,
        "route_distance": float(cumulative_distance[-1]),
        "predicted_ego_visibility_ratio": predicted_visibility_ratio,
        "assigned_speeds": assigned_speeds,
    }


def generate_clip(args, scene_name, variant_index, seed):
    scene = SCENES[scene_name]
    env = make_env(scene, seed, args.fps)
    frames = []
    frame_metadata = []
    view_meters = args.size / args.scaling
    step_dt = PHYSICS_DT * round(1.0 / (args.fps * PHYSICS_DT))

    # The large backing canvas must contain the whole procedural map.  Only the
    # small screen_size crop is encoded into the video.
    film_side = max(3000, args.size * 4)
    try:
        env.reset(seed=seed)
        env.engine.force_fps.disable()

        assigned_speeds = initialize_vehicle_speeds(
            env,
            seed,
            ego_speed_km_h=args.ego_initial_speed,
            traffic_speed_range=(args.traffic_min_speed, args.traffic_max_speed),
        )
        camera_position = None
        camera_plan = None
        if args.camera_mode != "map_center":
            camera_plan = plan_fixed_camera(env, args, seed, scene_name)
            camera_position = camera_plan["camera_position"]
            assigned_speeds = camera_plan["assigned_speeds"]

        initial_ego_position = np.asarray(env.agent.position, dtype=float)
        initial_traffic_ids = {
            obj.id
            for obj in env.engine.get_objects().values()
            if isinstance(obj, BaseVehicle) and obj is not env.agent
        }
        previous_velocity = np.asarray(env.agent.velocity, dtype=float)
        previous_speed = float(env.agent.speed)
        previous_heading = float(env.agent.heading_theta)

        render_kwargs = {
            "film_size": (film_side, film_side),
            "screen_size": (args.size, args.size),
            "scaling": args.scaling,
            "camera_position": None if camera_position is None else tuple(float(v) for v in camera_position),
            "center_on_map": args.camera_mode == "map_center",
            "target_agent_heading_up": False,
            "num_stack": 1,
            "draw_target_vehicle_trajectory": False,
            "draw_contour": True,
            "contour_width": 1,
            "semantic_map": args.semantic,
            "semantic_target_vehicle_color": EGO_SEMANTIC_COLOR,
            "semantic_traffic_vehicle_color": TRAFFIC_SEMANTIC_COLOR,
            "window": False,
        }

        for frame_index in range(args.frames):
            _, reward, terminated, truncated, info = env.step([0.0, 0.0])
            frame = env.render(mode="topdown", **render_kwargs)
            frames.append(np.ascontiguousarray(frame, dtype=np.uint8))

            position = np.asarray(env.agent.position, dtype=float)
            velocity = np.asarray(env.agent.velocity, dtype=float)
            speed = float(env.agent.speed)
            heading = float(env.agent.heading_theta)
            acceleration_world = (velocity - previous_velocity) / step_dt
            longitudinal_acceleration = (speed - previous_speed) / step_dt
            heading_delta = np.arctan2(np.sin(heading - previous_heading), np.cos(heading - previous_heading))
            yaw_rate = float(heading_delta / step_dt)
            action = [float(v) for v in env.agent.current_action]
            steering = float(env.agent.steering)
            throttle_brake = float(env.agent.throttle_brake)
            current_traffic_stats = traffic_stats(env, view_meters / 2, initial_traffic_ids)

            frame_metadata.append(
                {
                    "frame": frame_index,
                    "simulation_time_s": round((frame_index + 1) / args.fps, 6),
                    "ego_position_m": [float(v) for v in position],
                    "ego_heading_rad": heading,
                    "ego_velocity_m_s": [float(v) for v in velocity],
                    "ego_speed_m_s": speed,
                    "ego_speed_km_h": float(env.agent.speed_km_h),
                    "ego_acceleration_world_m_s2": [float(v) for v in acceleration_world],
                    "ego_longitudinal_acceleration_m_s2": float(longitudinal_acceleration),
                    "ego_yaw_rate_rad_s": yaw_rate,
                    "action": action,
                    "steering": steering,
                    "steering_angle_deg": steering * float(env.agent.max_steering),
                    "throttle_brake": throttle_brake,
                    "throttle": max(throttle_brake, 0.0),
                    "brake": max(-throttle_brake, 0.0),
                    "acceleration_command": float(info.get("acceleration", throttle_brake)),
                    "reward": float(reward),
                    "cost": float(info.get("cost", 0.0)),
                    "step_energy": float(info.get("step_energy", 0.0)),
                    "on_lane": bool(env.agent.on_lane),
                    "out_of_route": bool(env.agent.out_of_route),
                    "crash_vehicle": bool(env.agent.crash_vehicle),
                    "crash_object": bool(env.agent.crash_object),
                    **current_traffic_stats,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "natural_termination": bool(info.get("natural_termination", False)),
                    "route_completion": float(info.get("route_completion", 0.0)),
                }
            )

            previous_velocity = velocity
            previous_speed = speed
            previous_heading = heading

            if terminated or truncated:
                raise RuntimeError(
                    "Episode ended before all frames were collected "
                    f"({scene_name=}, {seed=}, frame={frame_index})."
                )
    finally:
        env.close()

    stem = f"{scene_name}_v{variant_index:02d}_seed{seed}"
    video_path = args.output_dir / f"{stem}.mp4"
    frames_path = args.output_dir / f"{stem}_bev.npy"
    metadata_path = args.output_dir / f"{stem}.json"
    frame_array = np.stack(frames)
    np.save(frames_path, frame_array, allow_pickle=False)
    mediapy.write_video(video_path, frame_array, fps=args.fps)

    metadata = {
        "scene": scene_name,
        "variant": variant_index,
        "seed": seed,
        "map_block_sequence": scene["map"],
        "lane_num": scene["lane_num"],
        "traffic_density": scene["traffic_density"],
        "frame_count": args.frames,
        "fps": args.fps,
        "env_step_hz": args.fps,
        "preview_mp4_fps": args.fps,
        "duration_s": args.frames / args.fps,
        "physics_dt_s": PHYSICS_DT,
        "decision_repeat": round(1.0 / (args.fps * PHYSICS_DT)),
        "resolution": [args.size, args.size],
        "bev_frames_file": frames_path.name,
        "bev_frames_shape": list(frame_array.shape),
        "image_dtype": "uint8",
        "image_color_order": "RGB",
        "scaling_px_per_m": args.scaling,
        "view_size_m": [view_meters, view_meters],
        "camera_fixed_in_world": True,
        "camera_mode": args.camera_mode,
        "center_on_map": args.camera_mode == "map_center",
        "camera_position_m": None if camera_position is None else [float(v) for v in camera_position],
        "scene_core_position_m": (
            None if camera_plan is None else [float(v) for v in camera_plan["scene_core"]]
        ),
        "target_block_id": None if camera_plan is None else camera_plan["target_block_id"],
        "preview_route_distance_m": None if camera_plan is None else camera_plan["route_distance"],
        "predicted_ego_visibility_ratio": (
            None if camera_plan is None else camera_plan["predicted_ego_visibility_ratio"]
        ),
        "initial_ego_position_m": [float(v) for v in initial_ego_position],
        "ego_centered": False,
        "ego_heading_up": False,
        "ego_semantic_rgb": list(EGO_SEMANTIC_COLOR),
        "traffic_semantic_rgb": list(TRAFFIC_SEMANTIC_COLOR),
        "ego_initial_speed_km_h": args.ego_initial_speed,
        "traffic_initial_speed_range_km_h": [args.traffic_min_speed, args.traffic_max_speed],
        "traffic_mode": TrafficMode.Trigger,
        "traffic_respawn_enabled": False,
        "traffic_spawn_gap_m": TRAFFIC_SPAWN_GAP_M,
        "initial_traffic_vehicle_count": len(initial_traffic_ids),
        "max_new_traffic_vehicle_count": max(
            frame["new_traffic_vehicle_count"] for frame in frame_metadata
        ),
        "frames_with_traffic_collision": sum(
            frame["crashed_traffic_vehicle_count"] > 0 for frame in frame_metadata
        ),
        "max_stopped_traffic_vehicle_count": max(
            frame["stopped_traffic_vehicle_count"] for frame in frame_metadata
        ),
        "assigned_initial_speeds_km_h": assigned_speeds,
        "semantic": args.semantic,
        "frames": frame_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return video_path, frames_path, metadata_path


def main():
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    clip_index = 0
    for scene_name in args.scenes:
        for variant_index in range(args.variants):
            seed = args.seed + clip_index
            print(f"Generating {scene_name}, variant {variant_index}, seed {seed} ...")
            generated.append(generate_clip(args, scene_name, variant_index, seed))
            clip_index += 1

    print(f"Generated {len(generated)} clips in: {args.output_dir.resolve()}")
    for video_path, frames_path, metadata_path in generated:
        print(f"  {video_path}  ({frames_path.name}, {metadata_path.name})")


if __name__ == "__main__":
    main()
