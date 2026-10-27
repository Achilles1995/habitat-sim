# EVK4 event camera in Habitat 3.0

`evk4_habitat` puts the physics-based Prophesee EVK4 of `evk4_mujoco` into Habitat-Sim. Use it on
Spot or any other robot, on walls, or on several of each at once. The pixel array is the original
model: the same Warp `pixel_kernel`, the same readout/ESP/ERC code and the same configuration keys
and bias mapping. There is no v2e-style frame differencing.

## How the photons get to the pixels

1. **Linear HDR rendering.** Habitat renders an HDR colour image for the camera (new
   `CameraSensorSpec.hdr`: an RGBA16F target with float32 readback).
   - `make_simulator()` switches the PBR shaders to linear output: textures are converted to
     linear reflectance, and there is no tonemapping and no sRGB encoding.
   - The pixel values are therefore proportional to scene radiance.
   - Without this, 8-bit tonemapped frames would turn every dark gradient into fake events.
2. **Radiometry.** Each rendered value becomes luminance:
   `L = lux_per_unit · (w·rgb) / π` [cd/m²]. Without an explicit `lux_per_unit`, the scale is set
   once per scene so that its mean illuminance is `scene_lux` (default 300 lux, assuming 18 % grey).
   Then:
   - cos⁴ vignetting;
   - pixel-area sampling and lens distortion by resampling (`spp`, `distortion`, `fx fy cx cy`);
   - optional mains flicker.
   This gives the photocurrent `k_rate · L`, with k_rate from f-number, pitch and QE exactly as in
   MuJoCo. The PSF blur comes next.
3. **Continuous time.** Between two `update(t)` calls, the camera re-renders the scene at
   interpolated states. It interpolates camera, robot base and joints, humanoid root and every
   SMPL-X joint (slerp), and objects. There are as many sub-steps as needed to keep the image
   motion under `max_px_motion` (1 px by default, up to `max_substeps`), as in the MuJoCo backend.
   The pixel circuit then integrates between the samples, so events have µs timestamps.
4. **Pixel and readout.** These are the original `evk4_mujoco` code:
   - log photoreceptor with light-dependent bandwidth;
   - Rice-hazard shot noise;
   - threshold mismatch, refractory period, leak and hot pixels;
   - latency and jitter;
   - arbiter, AFK, STC/trail and ERC.

## Ground truth on every event

`labeled_events()` returns these fields:
- `x, y, p, t[µs]`
- `geom`: the instance id.
  - Humans: 60000+.
  - Robot links: 61000 + 100·robot + link + 1.
  - Objects: 63000+.
  - Scene: 0 (or its semantic id).
  - Background: −1.
- `depth`: Euclidean range [m].
- `flow_x, flow_y` [px/s]. Flow is exact for the static scene, robot links and rigid objects.
  On skinned humans it uses the nearest bone, a rigid-skinning approximation.

`SceneMotion.instances()` maps ids to names, links and kinds.

## Synchronised RGB

`cam.frame()` returns the RGB frame at the sensor time, plus luminance, depth, instance ids, K and
the camera pose. It models a global-shutter colour camera that shares the EVK4's optics, like a
beam splitter or a DAVIS-style APS:
- The frame is pixel-aligned with the events in the default configuration (no distortion, spp 1).
- Its timestamp is the end of the exposure, on the same microsecond clock as the events.
- Motion blur is real: the linear renders of the sub-steps inside the exposure (`cam.rgb_exposure_s`,
  default 8 ms) are averaged before the image is exposed and tonemapped.

`hrc_generate.py` records this for every camera by default, at 30 fps, into `frames_<cam>.h5`
(`/rgb`, `/t_us`, `/depth`, `/instance`, `/T_world_cam`).
- `--rgb-every N`: record every N-th frame.
- `--rgb-exposure-ms`: exposure time.
- `--rgb-every 0`: events only.

`hrc_inspect.py <episode> --cam <cam> --out sync.png` overlays the events on the RGB frames and
measures the alignment. `hrc_train.py --modality events|rgb|both` trains on events, on the
synchronised RGB frames, or on both fused.

## Quick start

```python
import sys; sys.path.insert(0, "evk4/python")
import evk4_habitat as evkh
sim = evkh.make_simulator("data/scene_datasets/habitat-test-scenes/van-gogh-room.glb")
motion = evkh.SceneMotion(sim)
human = evkh.Humanoid(sim, motion, *evkh.humanoid_files("data"), feet_pos=evkh.snap(sim, [0, 0, 0]))
spot = evkh.Robot(sim, motion, "spot", evkh.snap(sim, [2, 0, 0]))
cams = [evkh.EVK4Habitat(sim, evkh.spot_mount(spot.ao, "front"), motion, name="spot_front",
                         config=dict(resolution="1280 720", binning=2, bias_diff_on=10)),
        evkh.EVK4Habitat(sim, evkh.wall_mount(eye=[-2, 2.2, -2], target=[0, 0.8, 0]), motion, name="wall")]
human.walk_to(evkh.snap(sim, [3, 0, 1]))
t = 0.0
while human.path:
    human.follow_path(); t += 1 / 30
    for c in cams:
        c.update(t)
        ev = c.labeled_events()
```

Mounts:
- `wall_mount(eye, target)`.
- `link_mount(ao, link, pos, look_dir, pitch_deg)`: any robot, any link.
- `spot_mount(ao, "front" | "head" | "top" | "rear")`.
- `Mount(T_parent_cam, (ao, link))`: a hand-eye calibration in OpenCV convention, via `pose_cv`.

## Datasets and baselines for the two HRC applications

```
# 1) human-aware robot safety: proximity, intent / trajectory, collision risk
python evk4/tools/hrc_generate.py safety  --data data --episodes 50 --seconds 12 --out datasets --preview
# 2) collaborative action recognition: approach, handover, point, wave, pick, place, sit, walk past, idle, fall
python evk4/tools/hrc_generate.py actions --data data --episodes 50 --out datasets
#    in an HSSD house:  --dataset data/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json --scene 102816009

python evk4/tools/hrc_train.py safety  --root datasets/safety  --epochs 15 --out runs/safety
python evk4/tools/hrc_train.py actions --root datasets/actions --epochs 25 --out runs/actions --augment
python evk4/tools/hrc_train.py actions --root datasets/actions --epochs 25 --out runs/actions_rgb --modality rgb
python evk4/tools/hrc_train.py actions --root datasets/actions --epochs 25 --out runs/actions_both --modality both
```

`hrc_generate.py --help` describes the episode format: per camera `events_<cam>.h5` and the
synchronised `frames_<cam>.h5`, plus `gt.npz` and `meta.json`.

Speed depends on the GPU: on an RTX 4070, Warp runs on CUDA and Habitat renders on the same GPU.
Lower `--binning`, the number of cameras or `max_substeps` to trade fidelity for speed.

## Tests

- `python evk4/tests/test_habitat_chain.py`: an analytic scene, no renderer needed. It checks:
  - radiometry;
  - the ×2 light step against the MuJoCo expectation;
  - flow, range and instance labels of a moving object;
  - adaptive sub-steps;
  - noise against light level;
  - the distortion/area-sampling path.
- `python evk4/tests/test_habitat_scene.py --data data`: humanoid, Spot camera and wall camera in
  the van Gogh room. It checks:
  - HDR input;
  - static noise level;
  - events on the human with correct range;
  - flow direction;
  - ego-motion flow.
- The original MuJoCo tests are in `evk4/tests/test_*.py` (unchanged).
