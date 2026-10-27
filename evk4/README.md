# prophesee.sensor.evk4 (v2): a physics-based event camera for MuJoCo

A native **MuJoCo sensor plugin** that models a **Prophesee EVK4 HD** event camera (Sony IMX636, 1280×720, 4.86 µm pixels). It comes with a **GPU backend** (NVIDIA Warp), a **UR5e scene with the camera on the tool flange**, a live viewer, EVT 3.0 export, a calibration tool for real EVK4 recordings, and test suites.

You declare it like an RGB camera:

```xml
<extension> <plugin plugin="prophesee.sensor.evk4"/> </extension>
<camera name="evk4_cam" pos="0 0 0.069" quat="0 1 0 0" fovy="47.3"/>
<sensor>
  <plugin name="evk4" plugin="prophesee.sensor.evk4" objtype="camera" objname="evk4_cam">
    <config key="resolution" value="1280 720"/>
    <config key="focal_length_mm" value="4.0"/>
    <config key="bias_file" value="my_evk4.bias"/>     <!-- optional: your Metavision biases -->
  </plugin>
</sensor>
```

## What's new in v2

| Area | v2 feature | Why it matters |
|---|---|---|
| **MuJoCo integration** | All sensor state lives in `mjData.plugin_state`. `mj_copyData`, resets, rollouts and checkpoints reproduce the event stream **bit-exactly**. | Needed for RL, MPC and reproducible datasets. |
| **Ground truth** | Every event carries the **object (geom) id, depth and exact optical flow** from MuJoCo kinematics. The flow is not estimated. | Free labels for learning flow, segmentation and depth from events. |
| **Noise physics** | Shot noise is band-limited by the photoreceptor poles. Noise-driven threshold crossings come from **Rice's crossing rate** of that smooth process, integrated in closed form over each segment. | Noise statistics **do not depend on the simulation step size** (tested). Simple per-step noise sampling fails this: in v1 the 10-lux noise rate varied ×4.5 with dt. |
| **Sensor fidelity** | Optional second photoreceptor pole (source follower); **Metavision bias integers** (`bias_diff_on/off`, `bias_fo`, `bias_hpf`, `bias_refr`) or a `.bias` file; temperature-dependent leak; reset noise. | Real camera settings can be copied into simulation. |
| **EVK4 readout & ESP** | ROI; arbiter/bus throughput with **load-dependent latency**; **AFK** anti-flicker; **STC / trail** filters; windowed **ERC**. | These shape real EVK4 data and no other simulator models them. |
| **Optics** | Calibrated intrinsics `fx fy cx cy` + OpenCV **Brown-Conrady distortion**; **stratified pixel-area sampling** (`spp`) against aliasing; smooth-normal and **UV-textured meshes** (own BVH traversal). | Match your real lens calibration. Removes faceting and edge-aliasing false events. |
| **Speed (CPU)** | Tiles whose objects, shadows and lights did not move are **not re-rendered** (same output, tested). | Measured **2.4× faster** with a static camera (530 vs 1250 ms/step at 1280×720 on 2 vCPU); no gain while the camera moves. |
| **GPU backend** | Same physics as Warp kernels (CUDA), driven by the same MJCF config. Cross-checked against the C++ plugin. | For the RTX 4070. |
| **Formats** | **EVT 3.0 `.raw` writer** (open in Metavision Studio / OpenEB), `.bias` I/O, voxel grids, time surfaces, histograms. | Direct drop-in for Metavision tools. |
| **Calibration** | `tools/calibrate_evk4.py` fits thresholds, mismatch, bandwidth, noise and leak to real EVK4 recordings. Checked in a closed loop. | For your validation stage. |

## GPU renderer (v2.5 / v2.6)

With `backend="warp"`, rendering runs as short passes instead of one very large kernel.
The output is **bit-identical** to the old fused kernel (tested: same radiance, labels and events).

| Pass | Work done | Registers (sm_89) |
|---|---|---|
| `hit_kernel` | primary rays only | 147 |
| `tilebound_kernel` | world-space box of each 16×16 tile's hits | 45 |
| `shadowcull_kernel` | per tile and light, keeps only geoms whose bounding sphere touches the tile→light capsule (UR5e scene: **3.8 of 34**) | 33 |
| `surf_kernel` | smooth normal, UV, texture, albedo | 108 |
| `shadow_kernel` | any-hit shadow rays against the culled list | 121 |
| `light_kernel` | diffuse, specular and emission | 63 |
| `flow_kernel` | exact optical-flow labels | 40 |
| *old fused `render_kernel`* | *all of the above* | *255 (hardware limit)* |

- **Thread layout:** threads are tile-major, so one warp covers a 16×2 pixel block rather than a 32×1 row. Rays in a warp then stay coherent.
- **No atomics:** the per-pixel tile-bound atomics are gone.
- **Faster first run:** kernels are compiled forward-only (no adjoint code), which roughly halves the first JIT.
- **Lower per-step overhead (v2.6):**
  - Recorded kernel launches, so only the camera, time and a few flags are re-sent each sub-step.
  - One asynchronous pinned-memory upload per render, replacing 4–6 small copies.
  - Pinned-memory event downloads.
  - Python-side cost per sub-step fell by about 45%.
- **Run-time appearance edits:** changes to `geom_rgba`, `geom_size` or materials now reach the GPU renderer, as they already did in the C++ plugin (test 6).
- **Tuning knobs** (benchmark flags or environment variables):
  - `--render split|fused` (`EVK4_RENDER`)
  - `--block 128` (`EVK4_BLOCK`)
  - `--maxreg 0|128` (`EVK4_MAXREG`)

## Any robot + paired frames (Phase 1)

### Mount the EVK4 on any robot

```python
import evk4_mujoco as evk
m = evk.attach_evk4("robots/menagerie/franka_emika_panda/scene.xml", body="hand",
                    pos=(0.065, 0, 0.02), euler_deg=(0, 0, 0),           # optical centre, OpenCV axes
                    config=dict(resolution="640 480", focal_length_mm=3.0),
                    backend="warp", save_xml="panda_evk4.xml")          # optional: keep the combined MJCF
```

- **Mount frame = OpenCV camera frame:**
  - origin at the lens's optical centre;
  - +z is the viewing direction, +x is image right, +y is image down.
  - A hand–eye result `T_flange_cam` therefore drops straight into `pos` / `quat`.
- **Housing:** the lens, body and adapter are drawn behind the optical centre. They are invisible to their own sensor.
- **Mass and gravity compensation:** the camera weighs 75 g and follows the parent's gravity-compensation setting.
- **Helpers and multiple cameras:**
  - `add_evk4(spec, ...)` works on an `MjSpec` you are building.
  - `look_at_quat(eye, target)` points a fixed camera at a target.
  - `add_props(spec, center)` adds a texture-rich workspace that needs no texture files.
  - Several cameras (e.g. a stereo pair) can be added via `name=`.
- **Robots:** `python tools/fetch_robots.py panda gen3 x2 ur5e` downloads only those MuJoCo Menagerie robots. It needs no git and no GitHub API.

### Paired RGB / depth / segmentation / optical flow

```python
cam  = evk.Sensor(m, "evk4", backend="warp")
pair = evk.PairedCamera(m, "evk4")
mujoco.mj_step(m, d); cam.update(d)
fr = pair.capture(d, t=cam.time())       # fr.rgb, fr.depth (z, m), fr.seg_geom, fr.seg_body, fr.T_world_cam, fr.K
flow, valid = pair.flow(prev_fr, fr)      # dense forward flow [px], exact from MuJoCo poses
```

- **Same pixel grid as the events:** frames use the event camera's own pixel grid, including principal point, fx ≠ fy and Brown–Conrady distortion. Frame pixel (u, v) and event pixel (u, v) are the same ray.
- **Rendering:** MuJoCo's renderer draws an ideal pinhole image, and each sensor pixel is looked up through the sensor's own lens model.
- **Mirror reflections are off by default,** because the event sensor has none.

`tests/test_phase1.py` checks the following:

| Check | Result |
|---|---|
| `attach_evk4` on the UR5e vs the hand-written XML | bit-identical events |
| Frame pixel alignment (pinhole and distorted lens) | < 0.35 px |
| Paired segmentation vs the event engine's own labels | 99.9 % agreement |
| Paired range vs the event engine's per-pixel range | median difference < 0.01 mm |
| Per-event flow labels vs dense ground-truth flow | 1.3 % median difference at 1 ms, 0.3 % at 0.25 ms (finite differencing) |

**Examples:**
- `examples/any_robot.py --robot panda|gen3|x2|ur5e|ur5e_scene` gives a live viewer.
- Add `--record out.mp4` for a 4-panel video: events | RGB / depth | flow.
- Add `--frames-dir DIR` to save `events.npy` plus `frame_XXXXX.npz` for training data.

## ESD tabletop-segmentation simulation (train on sim, test on real ESD)

This replicates the ESD dataset setup of Huang et al., *Scientific Data* 11:127 (2024): the same objects and scenarios, rendered on the UR5e, written in ESD's own format.

**Rig:** two DAVIS346 event cameras plus an Intel D435 RGB-D camera on the flange.
- **Calibration:** intrinsics, distortion and TCP→camera transforms are copied verbatim from the authors' `ESD_labeling_tool`.
- **Geometry:** both event cameras are toed in about 5–7° toward the middle; stereo baseline 0.21 m.

**Objects** (`esd/catalog.py`):
- **ESD-1 replicas (training, 10 objects):** book, white box, blue pack, tennis-ball tube, milk carton, tea box, green bottle, glass, pencil case, small cap.
- **ESD-2 replicas (testing, 5 objects).**
- **Procedural random objects** for domain randomisation.
- **Label ids and mask colours** match the authors' `read_label.m`.
- **Sizes:** measured from the authors' example frame. `tools/esd_measure_objects.py` re-measures them and extracts real top-face textures from your ESD frames.

**Scenarios** (`esd/motion.py`, `esd/generate.py`):

| Factor | Values |
|---|---|
| Number of objects | O2, O4, O6, O8, O10 (ESD-2: 2, 5) |
| Speed | 0.15, 0.3, 1 m/s |
| Trajectory | linear, rotary, linear + rotary in x-y-r |
| Lighting | normal, low |
| Camera height | 0.62, 0.82 m |
| Occlusion | with, without (physically stacked objects) |

- The UR5e follows the camera paths through IK to within 0.01 mm, and the arm never appears in the images.
- **Grid sizes:** the full ESD-1 grid is 324 sequences; the ESD-2 grid is 32.

**Output** (per sequence; same folder layout and variable names as the real ESD):
- `RGB/images|masks|labels|depth`;
- `events/left.mat` and `right.mat` with `event_labeled` = [x y p t(ns) depth(mm) label];
- `events_frame.mat` and `mask_events_frame.mat`;
- `meta.json`.

The labels come straight from the simulator, with no ICP step. `esd/data.py` reads simulated and real ESD folders the same way and provides event representations and the paper's metrics (Acc, mIoU).

```bash
python tools/esd_measure_objects.py --rgb <ESD>/RGB/images/20.png --mask <ESD>/RGB/masks/20.png --out esd_measured
python tools/esd_match_rate.py --real <ESD-1 sequence> --real <another> --backend warp   # fit the DAVIS threshold
python tools/esd_generate.py --grid esd1 --measured esd_measured/objects.json --sensor-json esd_sensor_fit.json \
       --repeat 2 --backend warp --device cuda:0 --out sim_esd --split train_esd1
python tools/esd_generate.py --random 500 --object-set random --randomize-sensor --randomize-scene \
       --sensor-json esd_sensor_fit.json --backend warp --device cuda:0 --out sim_esd --split train_random
python tests/test_esd.py
```

**Which data for which test:**
- **Known objects** (test on real ESD-1): train on the `esd1` grid.
- **Unseen objects** (test on real ESD-2): train on `random` / `mixed` objects, class-agnostic (`data.torch_dataset(..., class_agnostic=True)`). Keep the `esd2` replicas out of training.

## Pipeline (per `mj_step`)

1. **Radiometry.** Own ray caster (no OpenGL) over primitives and meshes with lights, textures, specular highlights and cast shadows. Output is linear luminance, converted to photo-electrons per second through f-number, cos⁴ vignetting, PSF, pixel area and QE.
2. **Time.** Adaptive sub-steps at interpolated MuJoCo poses until image motion is at most `max_px_motion` px. Re-rendering is occlusion-correct; there is no image warping.
3. **Pixel.** Log photoreceptor with light-dependent bandwidth (one or two poles), solved exactly for ramp inputs. Rice-hazard shot noise. ON/OFF comparators with mismatch, refractory reset with reset noise, reference high-pass, leak, hot pixels.
4. **Readout.** Latency and jitter, ROI, arbiter queue, AFK, STC/trail, ERC, 1 µs timestamps, time-ordered stream, finite buffer.

## Quick start

```bash
pip install mujoco numpy opencv-python imageio imageio-ffmpeg warp-lang
./build.sh                                          # C++ plugin against pip's MuJoCo
python tests/test_physics.py && python tests/test_v2_features.py
python examples/live_view.py --binning 2            # live: MuJoCo viewer + events (CPU)
python examples/live_view.py --backend warp         # live on the GPU (RTX 4070), full 1280x720
python examples/live_view.py --backend warp --resolution 320 320   # 320x320 array, lens scaled to keep the FOV
python tools/fetch_robots.py panda gen3 x2 ur5e                      # Menagerie robots for the next line
python examples/any_robot.py --robot panda --backend warp            # EVK4 on a Franka hand, live
python examples/benchmark.py --backend warp --device cuda:0
MUJOCO_GL=egl python examples/record_demo.py --panel flow --raw demo.raw --out demo.mp4
```

- **macOS:** use `mjpython` for the live viewer.
- **Windows:** build with `cmake -S plugin -B build -DMUJOCO_DIR=<mujoco>`.
- **GPU backend:** load the model with `backend="warp"`. The C++ plugin then only reserves the sensor slot, and the Warp engine reads the same `<config>`.

```python
import mujoco, evk4_mujoco as evk
m = evk.load_model("scenes/scene_ur5e_evk4.xml", backend="warp", bias_diff_on=20)   # kwargs override <config>
d = mujoco.MjData(m)
cam = evk.Sensor(m, "evk4", backend="warp", device="cuda:0")    # or backend="cpu" (C++ plugin)
mujoco.mj_step(m, d); cam.update(d)
ev  = cam.events()          # Metavision EventCD layout: x u2, y u2, p i2, t i8 [us], time-ordered
gt  = cam.labeled_events()  # + geom (object id), depth [m], flow_x, flow_y [px/s]
from evk4_mujoco.io import EVT3Writer, voxel_grid       # .raw for Metavision, learning representations
```

## Configuration

**Sensor and optics**

| key | default | meaning |
|---|---|---|
| `resolution` | `1280 720` | pixel array |
| `binning` | 1 | virtual binning for fast previews |
| `pixel_pitch_um` | 4.86 | pixel pitch |
| `focal_length_mm` | 0 | lens focal length; 0 = use the camera's `fovy` |
| `f_number` | 2.0 | aperture |
| `qe` | 0.5 | quantum efficiency × fill factor |
| `fx fy cx cy` | derived | calibrated intrinsics (full-resolution pixels, OpenCV) |
| `distortion` | `0 0 0 0 0` | k1 k2 p1 p2 k3 (OpenCV) |
| `spp` | 1 | 1 / 4 / 9 / 16 stratified samples per pixel |
| `psf_sigma_px` | 0.5 | optical blur |
| `rgb_weights` | Rec.709 | spectral response on R, G, B |

**Scene radiometry**

| key | default | meaning |
|---|---|---|
| `lux_per_unit` | 1000 | lux from a MuJoCo light with `diffuse=1` at normal incidence |
| `background_lux` | 50 | luminance × π for rays that hit nothing |
| `dark_current` | 30 | e⁻/s |
| `shadows` | 1 | cast shadows |
| `specular_scale` | 1 | scale of specular highlights |
| `flicker_hz`, `flicker_depth` | 0, 0 | light modulation |

**Pixel**

| key | default | meaning |
|---|---|---|
| `cutoff_hz` | 2000 | max photoreceptor bandwidth (~`bias_fo`) |
| `bw_half_lux` | 20 | illuminance at half bandwidth |
| `sf_cutoff_hz` | 0 | 2nd pole; 0 = off (3·f_c used for the noise spectrum) |
| `shot_noise`, `noise_gain` | 1, 0.05 | shot noise and its calibration factor |
| `contrast_on`, `contrast_off` | 0.22, 0.22 | thresholds in ln units |
| `contrast_sigma` | 0.02 | threshold mismatch |
| `refractory_us` | 50 | refractory period |
| `hpf_hz` | 0 | reference high-pass |
| `leak_rate_hz`, `leak_sigma` | 0.05, 0.5 | leak events |
| `temperature_c`, `leak_doubling_c` | 25, 7 | leak doubles every 7 °C |
| `hot_pixel_fraction`, `hot_pixel_rate_hz` | 5e-5, 20 | hot pixels |

**Metavision biases**

| key | default | meaning |
|---|---|---|
| `bias_file` | – | Metavision `.bias` file |
| `bias_diff_on/off`, `bias_fo`, `bias_hpf`, `bias_refr` | 0 | bias integers |
| `bias_*_scale` | 100 / 40 / 0.5 / 80 | bias → physics mapping (fit with the calibration tool) |

**Readout and ESP**

| key | default | meaning |
|---|---|---|
| `latency_us`, `latency_jitter_us` | 20, 5 | pixel/readout latency |
| `readout_mevps` | 0 | arbiter throughput; 0 = unlimited |
| `roi` | full | `x0 y0 x1 y1` |
| `afk`, `afk_min_hz`, `afk_max_hz` | 0, 40, 500 | anti-flicker |
| `stc_us`, `trail_us` | 0, 0 | STC noise filter / trail filter |
| `erc_mevps`, `erc_window_us` | 0, 200 | Event Rate Controller |
| `max_events` | 400000 | per-step output buffer |

**Engine**

| key | default | meaning |
|---|---|---|
| `max_px_motion`, `max_substeps` | 1.0, 12 | adaptive time sampling |
| `skip_static_tiles` | 1 | incremental rendering |
| `backend` | `cpu` | `cpu` = this plugin renders; `warp` = the GPU engine does |
| `geomgroup`, `nthread`, `seed`, `export_intensity`, `show_frustum` | – | misc |

## Verified behaviour

All suites pass. The numbers below are from the final run in this package.

**`tests/test_physics.py`**
- A static scene with noise off gives **0 events**.
- A ×2 light step gives 2.62 ON events/px against a theory of 2.61, and 0 OFF events.
- A ×0.5 step gives 2.60 OFF events/px against a theory of 2.61, and 0 ON events.
- A 100 µs refractory period absorbs part of a step.
- Latency after a step falls with light: 105 → 51 → 46 µs at 10 → 100 → 1000 lux.
- Background noise: 4.3 Hz/px at 10 lux, 0.12 Hz/px at 1000 lux.
- 100 Hz flicker gives ON bursts at 100 Hz.

**`tests/test_v2_features.py`**
- **Save/restore:** identical event streams after rewinding the same `MjData` and after copying into a new one.
- **Dirty-tile skipping** gives the same events as full re-rendering.
- **Flow labels:** median error **0.15 %** against the analytic flow of a spinning object.
- **Lens model** matches OpenCV `projectPoints` to **0.03 px**.
- **Metavision biases** change the thresholds as mapped, whether set as keys or through a `.bias` file.
- **Temperature:** the leak ratio is 7.4 against an expected 7.25 for 25 → 45 °C.
- **AFK** removes 100 % of a 100 Hz flicker.
- **STC** cuts dark noise by about 40×.
- **ERC** holds the target rate within 1 % and never exceeds the per-window cap.
- **Arbiter:** latency grows ×4 under load.
- **ROI** masks the array correctly.
- **Noise vs time step:** 10 lux gives 6.53 / 6.57 / 6.50 Hz/px at dt = 1 ms / 200 µs / 50 µs.

**`tests/test_gpu_backend.py` (Warp vs C++)**
- ln I images agree to about 1e-6.
- Light-step counts are identical (2.637 vs 2.637).
- Noise rates agree within 2 %.
- Event counts on the spinning mesh agree within 2 %.
- The UR5e scene gives 8903 vs 8904 events.
- The kernels compile to CUDA PTX for sm_89 (RTX 4070).

**`tests/test_io.py`**
- EVT 3.0 files round-trip exactly with our own decoder and with the independent **evlib** decoder, including the 24-bit timestamp wrap and vector words.
- **expelliarmus** reads the same x / y / polarity. Its timestamps differ because of a TIME_LOW heuristic that departs from the spec.

**`tests/test_calibration.py`**
- Synthetic "real" recordings with known parameters are recovered to 1–4 % for thresholds, noise gain, leak and `bw_half_lux`. The cutoff is recovered to 18 % and the mismatch to about 45 %.

## Calibrating against your real EVK4 (validation stage)

1. **Record.** Point the EVK4 at a uniform matte target and measure lux at the target. Use the same lens and f-number as in the MJCF. Record three kinds of data:
   - **Slow LED ramps** up and down by a known factor. These set the thresholds.
   - **Fast LED steps** at about 10, 100 and 1000 lux. These set the bandwidth.
   - **Static scenes** at 2–3 light levels. These set noise and leak.
2. **Describe.** List the recordings in a JSON file. The format is shown in `python/evk4_mujoco/calibration.py`.
3. **Fit.** Run `python tools/calibrate_evk4.py experiments.json` and paste the printed `<config>` lines into the scene. Do this once per bias setting to learn the `bias_*_scale` mapping.
4. **Digital twin.** Log UR5e joint angles with RTDE while recording the EVK4 on the real robot. Replay the joint trajectory in `scenes/scene_ur5e_evk4.xml`, then compare real and simulated `.raw` streams: event rate, polarity balance, and E2VID / E-RAFT metrics.

## Honest limitations

- **GPU speed not measured here.** The GPU backend's numerical equivalence was tested on Warp's CPU device, and its CUDA kernels compile, but no GPU was available to time it. Please run `examples/benchmark.py --backend warp` on the RTX 4070.
  - The readout/ESP stage runs on the host in numpy.
  - Batched multi-environment rendering is not implemented yet.
- **Rendering.** Local illumination only: no inter-reflections or transparency. Height fields and SDF geoms are invisible to the sensor.
- **Bias mapping.** The bias → physics mapping and the noise gain are calibration constants. Defaults are placeholders until fitted to your camera.
- **Interpolation between samples.** The noiseless signal is interpolated linearly between samples. Sampling follows motion and flicker automatically, but abrupt light switching needs a smaller `timestep`.
- **ESP filters.** The AFK, STC and ERC models follow Prophesee's documented behaviour, not the exact on-chip logic.

## Files

```
plugin/evk4_event_camera.cc     MuJoCo sensor plugin (C++17)           build.sh, plugin/CMakeLists.txt
python/evk4_mujoco/__init__.py  loader, EVK4 accessor, Sensor (cpu|warp), EventFrame
python/evk4_mujoco/gpu.py       Warp GPU backend
python/evk4_mujoco/io.py        EVT 3.0 writer/reader, .bias I/O, voxel grid, time surface, histogram
python/evk4_mujoco/calibration.py + tools/calibrate_evk4.py   fit to real EVK4 recordings
scenes/                         UR5e + EVK4 scene (UR5e: MuJoCo Menagerie, BSD-style licence)
examples/                       live_view.py, record_demo.py, benchmark.py, minimal.py
tests/                          test_physics, test_v2_features, test_gpu_backend, test_io, test_calibration
```

"Prophesee", "EVK4", "Metavision" and "IMX636" are trademarks of their owners. This is an independent simulation model. Simulated `.raw` files are marked `SIMULATED-MUJOCO` in their header.
