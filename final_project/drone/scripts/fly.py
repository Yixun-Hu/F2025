import argparse
import json
import os
import time
from datetime import datetime
from types import SimpleNamespace

import cv2
import torch
import yaml

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

from drone.controls.crazyflie_control import CrazyflieControl
from drone.datasets.dataloader import CrazyflieILDataset
from drone.models.continuous_action_model import ContinuousActionModel
from drone.models.discrete_action_model import DiscreteActionModel


def _resolve_path(base_dir: str, maybe_path: str) -> str:
    """Return an absolute path, preserving already-absolute inputs."""
    if maybe_path is None:
        return None
    if os.path.isabs(maybe_path):
        return maybe_path
    return os.path.abspath(os.path.join(base_dir, maybe_path))


def load_config(config_path: str) -> SimpleNamespace:
    """Load YAML config files without Hydra and mimic the previous defaults behavior."""
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve model configuration from the first `defaults` entry matching "models".
    model_cfg_name = None
    for entry in cfg.get("defaults", []):
        if isinstance(entry, dict) and "models" in entry:
            model_cfg_name = entry["models"]
            break
    if model_cfg_name is None:
        raise ValueError("Config must specify a model under `defaults`.")

    model_cfg_path = os.path.join(config_dir, "models", f"{model_cfg_name}.yaml")
    with open(model_cfg_path, "r") as f:
        model_cfg = yaml.safe_load(f) or {}

    cfg["models"] = model_cfg
    cfg.pop("defaults", None)

    # Resolve key paths relative to the config directory.
    for key in ("ckpt_path", "data_dir", "log_dir"):
        if key in cfg:
            cfg[key] = _resolve_path(config_dir, cfg[key])

    # Normalize shape-sensitive values.
    if isinstance(cfg.get("image_size"), list):
        cfg["image_size"] = tuple(cfg["image_size"])

    return SimpleNamespace(**cfg)


def build_model(model_cfg: dict) -> torch.nn.Module:
    """Instantiate the requested model without Hydra."""
    target = model_cfg.get("_target_", "")
    kwargs = {k: v for k, v in model_cfg.items() if k != "_target_"}

    if target.endswith("DiscreteActionModel"):
        return DiscreteActionModel(**kwargs)
    if target.endswith("ContinuousActionModel"):
        return ContinuousActionModel(**kwargs)

    raise ValueError(f"Unsupported model target: {target}")


def main(cfg: SimpleNamespace) -> None:
    # ----------------------------------------------------------------------- #
    #  Load and initialize model
    # ----------------------------------------------------------------------- #
    model = build_model(cfg.models)

    ckpt_path = cfg.ckpt_path
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # ----------------------------------------------------------------------- #
    #  Set up image transforms
    # ----------------------------------------------------------------------- #
    dataset = CrazyflieILDataset(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        augment=cfg.augment,
    )
    transform = dataset.transform  # match dataset transforms

    # ----------------------------------------------------------------------- #
    #  Set up logging
    # ----------------------------------------------------------------------- #
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.log_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    actions_path = os.path.join(save_dir, "actions.json")

    print(f"[LOG] Saving data to: {save_dir}")

    action_records = []

    # ----------------------------------------------------------------------- #
    #  Set up drone
    # ----------------------------------------------------------------------- #
    group_number = cfg.group_number
    URI = f"radio://0/{group_number}/2M"
    cflib.crtp.init_drivers(enable_debug_driver=False)

    camera_index = getattr(cfg, "camera_index", None)
    if camera_index is None:
        camera_index = getattr(cfg, "camera_id", 0)

    with SyncCrazyflie(URI) as scf:
        print("Connected!")
        mc = MotionCommander(scf)
        drone = CrazyflieControl(mc, camera_index=camera_index)

        print("Hovering...")
        drone.hover(height=cfg.hover_height)

        for i in range(cfg.fly_steps):
            # ========== 1) Get image from drone ==========
            image = drone.get_image(
                crop_top=cfg.crop_top
            )  # H x W x C (uint8)

            # Save raw frame
            img_filename = f"frame_{i:04d}.png"
            img_path = os.path.join(save_dir, img_filename)
            cv2.imwrite(img_path, image)

            # ========== 2) Preprocess ==========
            processed_image = transform(image).unsqueeze(0).to(device)

            # ========== 3) Model inference ==========
            # Create state tensor (zeros as placeholder since state not available during flight)
            # State format: [x, y, z, vx, vy, vz, roll, pitch, yaw]
            state = None
            if hasattr(model, 'state_dim') and model.state_dim > 0:
                state = torch.zeros(1, model.state_dim, device=device)

            with torch.no_grad():
                output = model(processed_image, state=state)

            actions = model.output_to_executable_actions(output)

            print(f"Step {i:03d} actions:", actions)

            # Append record to list
            action_records.append({
                "step": i,
                "image_file": img_filename,
                "actions": actions.tolist(),
            })

            # ========== 4) Send commands ==========
            drone.send_control_commands(actions)
            time.sleep(0.1)

        # Land the drone
        drone.mc.land()

    # ================== SAVE FINAL JSON ==================
    with open(actions_path, "w") as f:
        json.dump(action_records, f, indent=2)

    print("[LOG] Finished run. Data saved to:", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fly a Crazyflie with a trained policy (no Hydra).")
    default_config = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/example_fly_config.yaml"))
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to the YAML config file (defaults to example_fly_config.yaml).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg=cfg)