"""Run the moving-tray stacking demo. Use MUJOCO_GL=egl for headless video."""

import argparse
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import mujoco as mj
from PIL import Image, ImageDraw, ImageFont

from sim.stack import BLOCK_COUNT, TRAVEL, StackRun, build_model


FPS = 25
WIDTH, HEIGHT = 1280, 720


class VideoFrames:
    def __init__(self, renderer, stream):
        self.renderer = renderer
        self.stream = stream
        self.next_frame = 0.0
        self.title = ImageFont.load_default(size=26)
        self.label = ImageFont.load_default(size=18)
        self.small = ImageFont.load_default(size=15)

    def frame(self, run, stage):
        self.renderer.update_scene(run.data, camera="overview")
        frame = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(frame, "RGBA")
        panel = (17, 26, 38, 231)
        white = (241, 247, 253)
        mist = (179, 200, 220)
        draw.rounded_rectangle((24, 24, 384, 112), radius=13, fill=panel)
        draw.text((43, 41), "Stacking on a moving tray", font=self.title, fill=white)
        draw.text((43, 79), stage, font=self.label, fill=mist)

        draw.rounded_rectangle((1000, 24, 1256, 112), radius=13, fill=panel)
        draw.text((1018, 43), f"{run.data.time:05.2f} s   |   {run.tray_velocity*1000:+.0f} mm/s",
                  font=self.label, fill=white)
        draw.line((1019, 85, 1237, 85), fill=(104, 138, 159), width=3)
        shift = float(run.data.qpos[run.slide_q])
        fraction = max(0.0, min(1.0, (shift + TRAVEL) / (2 * TRAVEL)))
        marker = 1019 + 218 * fraction
        draw.ellipse((marker-5, 80, marker+5, 90), fill=(105, 225, 205))
        draw.text((WIDTH - 28, HEIGHT - 20), f"Tray period {run.period:.2f} s",
                  font=self.small, anchor="rb", fill=white)
        return frame

    def capture(self, run, stage):
        if run.data.time + 1e-8 >= self.next_frame:
            self.stream.write(self.frame(run, stage).tobytes())
            self.next_frame += 1 / FPS


def render_rollout(run, output):
    with TemporaryDirectory(dir=output) as work:
        movie = Path(work) / "demo.mp4"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
            "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "20", "-threads", "2", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(movie),
        ]
        with mj.Renderer(run.model, width=WIDTH, height=HEIGHT) as renderer:
            with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
                frames = VideoFrames(renderer, encoder.stdin)
                run.on_step = frames.capture
                frames.capture(run, "Four loose blocks")
                result = run.run()
                if result["success"]:
                    label = "4 layers stable through one cycle"
                elif all(layer["supported_after_retract"] for layer in result["layers"]):
                    label = "Stack lost support after release"
                else:
                    label = "Incomplete stack"
                poster = frames.frame(run, label)
                # Hold the measured final frame for one second.
                frozen_frame = poster.tobytes()
                for _ in range(FPS):
                    encoder.stdin.write(frozen_frame)
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError("ffmpeg could not encode the demo")
        movie.replace(output / "demo.mp4")
        poster.save(output / "preview.png")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/stack_demo"))
    parser.add_argument("--scene-smoke", action="store_true")
    parser.add_argument("--render", action="store_true", help="Export an actual rollout MP4 and final frame")
    parser.add_argument("--period", type=float, default=2.0, help="Tray motion period in seconds")
    args = parser.parse_args()
    if not math.isfinite(args.period) or args.period < 0.5:
        parser.error("--period must be finite and at least 0.5 s")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.scene_smoke:
        scene_smoke(args.output)
        return
    run = StackRun(args.period)
    result = render_rollout(run, args.output) if args.render else run.run()
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit(1)


def scene_smoke(output):
    model = build_model()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    for index in range(1, BLOCK_COUNT + 1):
        block = model.body(f"block{index}")
        assert model.jnt_type[block.jntadr[0]] == mj.mjtJoint.mjJNT_FREE
    assert model.joint("tray_slide").type[0] == mj.mjtJoint.mjJNT_SLIDE
    (output / "scene_smoke.json").write_text(json.dumps({
        "free_blocks": BLOCK_COUNT, "physics_dt_s": model.opt.timestep,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
