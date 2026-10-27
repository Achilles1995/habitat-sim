# evk4_mujoco: physics-based Prophesee EVK4 event camera for MuJoCo (handoff for a new chat)

**To the assistant reading this:** the user built this package in an earlier chat. Use it as-is. Do NOT write a new event-camera simulator or replace the pixel model. Build new applications on the API below. Before changing anything, run the tests. If you change core files, keep every test passing.

## 1. What it is

- **Native MuJoCo sensor plugin** `prophesee.sensor.evk4` (C++, `plugin/evk4_event_camera.cc`): a physics-based event camera, not an RGB-to-events converter.
  - Its own radiometric ray caster: lights, shadows, textures, smooth-normal meshes, lens distortion, PSF, vignetting.
  - Log photoreceptor with light-dependent bandwidth (1–2 poles).
  - Rice-hazard shot noise, which makes noise independent of the time step.
  - ON/OFF thresholds with mismatch, refractory period, leak, hot pixels, latency/jitter.
  - EVK4 readout: ROI, arbiter, AFK, STC/trail, ERC.
  - Metavision bias mapping (`bias_diff_on/off`, `bias_fo`, `bias_hpf`, `bias_refr`, `.bias` files).
  - Adaptive sub-stepping of the time step.
  - All state lives in `mjData.plugin_state`: rollouts and checkpoints are bit-exact.
- **GPU backend** (NVIDIA Warp, `python/evk4_mujoco/gpu.py`): same physics, same MJCF config, cross-checked against the C++ plugin. Measured at full 1280×720 on an RTX 4070 laptop: about 5.6 ms per 1 ms step with a static camera, about 12 ms with a moving camera. At 320×320 it is about 9× fewer pixels.
- **Ground truth on every event:** geom id, depth [m] (Euclidean range) and exact optical flow [px/s].
- **Default sensor:** EVK4 HD / Sony IMX636, 1280×720, 4.86 µm pixels. Any resolution, lens or calibration can be configured. A DAVIS346 preset exists in `esd/rig.py`.

## 2. Setup (user's machine: Windows, RTX 4070, MuJoCo 3.x pip, Python venv)

```
cmake -S plugin -B build -DMUJOCO_DIR=<folder with include/ and lib/>
cmake --build build --config Release
copy build\Release\evk4_event_camera.dll build\
pip install mujoco numpy warp-lang scipy h5py pillow opencv-python imageio imageio-ffmpeg
pip install torch --index-url https://download.pytorch.org/whl/cu124      (training only)
python tests\test_physics.py && python tests\test_phase1.py && python tests\test_esd.py --backend warp --device cuda:0
```

- **Linux / macOS:** run `./build.sh` instead of the cmake lines.
- **Plugin location:** `evk4_mujoco.plugin_path()` finds the right library (.dll/.so/.dylib) in `build/`, or uses the `EVK4_PLUGIN` environment variable.
- **Profiling:** `set EVK4_PROFILE=` disables the timing printout. `EVK4_PROFILE=1` prints totals; `=2` prints a per-stage breakdown.

## 3. Core API (`sys.path.insert(0, "python"); import evk4_mujoco as evk`)

```python
m  = evk.load_model("scenes/scene_ur5e_evk4.xml", backend="warp", resolution="320 320", focal_length_mm=1.78)
d  = mujoco.MjData(m)
cam = evk.Sensor(m, "evk4", backend="warp", device="cuda:0")    # or backend="cpu" (C++ plugin)
cam.update(d)                       # after mj_forward/mj_step, every step
ev  = cam.events()                  # Metavision EventCD layout: x u2, y u2, p i2 (1=ON, 0=OFF), t i8 [us]
gt  = cam.labeled_events()          # + geom i4, depth f4 [m range], flow_x, flow_y f4 [px/s]
cam.intensity(); cam.geometry(); cam.intrinsics(); cam.stats(); cam.time()  # time() = sensor time of current poses
```

- **Model overrides:** `load_model(xml, **config)` overrides any `<config key=...>` of the plugin.
- **GPU backend:** load with `backend="warp"`. The C++ plugin then only reserves a slot and the Warp engine renders.
- **CPU-only accessor:** `evk.EVK4(m, "evk4").events(d)`.
- **Time convention:** the CPU plugin samples at `d.time` *inside* `mj_step`, i.e. before the time increment. The GPU backend uses `d.time` after the step. Each backend is internally consistent; use `Sensor.time()` to timestamp anything paired with events.
- **Event display:** `evk.EventFrame(W, H, "dark")` (Metavision palette). `evk4_mujoco.io` provides `EVT3Writer`, `write_evt3`, `read_evt3` (Metavision `.raw`), `read_bias_file`, `write_bias_file`, `histogram`, `time_surface` and `voxel_grid`.

**Mount on any robot / body** (`mount.py`):

```python
m = evk.attach_evk4("robots/menagerie/franka_emika_panda/scene.xml", body="hand", pos=(0.065,0,0.02),
                    quat=(1,0,0,0), name="evk4", config=dict(resolution="640 480"), backend="warp")
```

- **Mount frame = OpenCV camera frame:** origin at the optical centre, +z = viewing direction, +x = image right, +y = image down. Hand–eye results `T_parent_cam` can be used directly.
- **Helpers:** `add_evk4(spec, ...)` works on an MjSpec; `look_at_quat(eye, target)`, `euler_to_quat`, `add_props(spec, center)`, `start_props`.
- **Multiple cameras:** use different `name=`.
- **Robots:** `python tools/fetch_robots.py panda gen3 x2 ur5e` downloads MuJoCo Menagerie robots to `robots/menagerie`.

**Paired frames, pixel-aligned with the event pixel grid** (`frames.py`):

```python
pair = evk.PairedCamera(m, "evk4")          # uses the sensor's own intrinsics + distortion
fr = pair.capture(d, t=cam.time())          # fr.rgb, fr.depth (z, m), fr.seg_geom, fr.seg_body, fr.T_world_cam, fr.K
flow, valid = pair.flow(fr_prev, fr)        # dense GT forward flow [px]
```

## 4. Important config keys (MJCF `<config key value>` or `load_model` kwargs)

| Group | Keys |
|---|---|
| Sensor / optics | `resolution` "W H", `binning`, `pixel_pitch_um`, `focal_length_mm` or `fx fy cx cy`, `distortion` "k1 k2 p1 p2 k3", `f_number`, `qe`, `psf_sigma_px`, `spp` |
| Scene radiometry | `lux_per_unit` (MuJoCo light units → lux), `background_lux`, `shadows` |
| Pixel | `contrast_on/off`, `contrast_sigma`, `cutoff_hz`, `bw_half_lux`, `sf_cutoff_hz`, `refractory_us`, `hpf_hz`, `leak_rate_hz`, `shot_noise`, `noise_gain`, `hot_pixel_fraction`, `temperature_c` |
| Biases | `bias_diff_on/off`, `bias_fo`, `bias_hpf`, `bias_refr`, `bias_file` |
| Readout | `latency_us`, `latency_jitter_us`, `readout_mevps`, `erc_mevps`, `afk`, `stc_us`, `trail_us`, `roi`, `max_events` |
| Timing | `max_px_motion` (sub-step criterion, default 1 px), `max_substeps` |
| Other | `geomgroup`, `seed`, `backend` (cpu / warp) |

## 5. Examples and tools

| File | Purpose |
|---|---|
| `examples/live_view.py --backend warp --device cuda:0 [--resolution 320 320]` | Live viewer with event overlay |
| `examples/any_robot.py --robot panda\|gen3\|x2\|ur5e\|ur5e_scene [--record out.mp4] [--frames-dir DIR]` | Camera on any robot; records events + RGB/depth/seg/flow |
| `examples/benchmark.py --backend warp --device cuda:0` | Speed |
| `tools/calibrate_evk4.py experiments.json` | Fit pixel parameters to real EVK4 recordings |

**ESD tabletop-segmentation simulation** (`python/evk4_mujoco/esd/`; Huang et al., *Sci. Data* 11:127, 2024):
- **Setup replicated:**
  - UR5e (kinematic IK), stereo DAVIS346 + D435 rig with the authors' calibration;
  - ESD-1/ESD-2 object replicas plus procedural random objects;
  - all ESD scenarios: objects O2–O10, speed 0.15/0.3/1 m/s, trajectory linear/rotation/linearrotation, normal/low light, 0.62/0.82 m height, occlusion.
- **Output:** ESD format (`events/left.mat` `event_labeled` = [x y p(±1) t(ns) depth(mm) label]; RGB images/masks/depth).
- **Tools:**
  - `tools/esd_generate.py` (`--grid esd1|esd2`, `--random N`, `--one`);
  - `esd_measure_objects.py`;
  - `esd_match_rate.py` (fits the contrast threshold to real event rates);
  - `esd_events_to_frames.py` (sim + real → `data.npz` of frames + labels; 255 = ignore);
  - `esd_train_unet.py` (PyTorch U-Net: train on sim, test on real; Acc / mIoU per condition).

## 6. Verified behaviour (tests/)

| Test file | What it checks |
|---|---|
| `test_physics.py`, `test_v2_features.py` | Pixel physics, noise invariance to dt, latency, biases, ESP filters, state save/restore |
| `test_gpu_backend.py` | GPU == CPU plugin (radiometry, light step, noise, flow, dirty tiles, run-time colour edits) |
| `test_phase1.py` | `attach_evk4` == hand-written XML (bit-identical); frame alignment < 0.35 px; paired seg/depth == event engine; event flow == dense GT flow |
| `test_esd.py` | Rig calibration, IK < 0.05 mm with arm out of view, speeds, ESD format, labels vs masks, loader on sim + real formats |
| `test_io.py`, `test_calibration.py` | EVT3 round trip; calibration recovers parameters |

## 7. Known limitations

- **Rendering:** local illumination only; no transparency, inter-reflections, height fields or SDF geoms in the event renderer.
- **Calibration:** DAVIS346 and bias-to-physics constants are placeholders until fitted to real recordings.
- **On-chip filters:** AFK/STC/ERC follow documented behaviour, not the exact on-chip logic.
- **Python cost:** per sub-step Python overhead is about 0.3 ms with the GPU backend.
