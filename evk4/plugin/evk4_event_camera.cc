// =====================================================================================
//  prophesee.sensor.evk4  (v2)  —  physics-based event-camera sensor plugin for MuJoCo
// =====================================================================================
//
//  Models a Prophesee EVK4 HD (Sony IMX636, 1280x720, 4.86 um pixels) as a native MuJoCo
//  sensor plugin.  Pipeline, per mj_step:
//
//   RADIOMETRY   own ray caster (no OpenGL): primitives + meshes (own BVH traversal with
//                barycentric smooth normals and UV textures), 2-D textures, Lambert +
//                Blinn-Phong, spot/point/directional lights, cast shadows, pixel-area
//                sampling (stratified spp), calibrated intrinsics + Brown-Conrady lens
//                distortion, cos^4 vignetting, optical PSF -> photo-electron rate [e-/s].
//   TIME         physics-informed adaptive sub-stepping: MuJoCo knows the exact motion of
//                every geom and of the camera; extra renders are made at interpolated poses
//                (occlusion-correct, no image warping) until image motion <= max_px_motion.
//                Tiles whose dependencies did not move are not re-rendered.
//   PIXEL        log photoreceptor with light-dependent bandwidth (bias_fo) + optional
//                source-follower pole, exact solution for ramp inputs; shot noise as an exact
//                Ornstein-Uhlenbeck process; ON/OFF comparators with mismatch
//                (bias_diff_on/off); refractory reset (bias_refr); reference high-pass
//                (bias_hpf); temperature-dependent junction leak; hot pixels.
//   READOUT      pixel latency + jitter, ROI, arbiter throughput queue (load-dependent
//                latency), EVK4 Event Signal Processing: AFK anti-flicker, STC / trail
//                filters, windowed Event Rate Controller; 1 us timebase; finite buffer.
//   LABELS       every event carries ground truth: geom id, depth, exact optical flow.
//
//  ALL dynamic state lives in d->plugin_state, so mj_copyData / mj_resetData / rollouts /
//  checkpoints reproduce the event stream bit-exactly.
//
//  Output
//   sensordata (8): [n_events, n_on, n_off, n_dropped_step, n_substeps, rate_Mev_s,
//                    mean_ln_I, n_dropped_total]
//   plugin_state  : header(64) | events(max_events x 8) | pending | pixel state | poses |
//                   probe | ln_I image.   Offsets are published in the header (see python).
//   event record  : x, y, p, t_us, geom_id, depth_m, flow_x_px_s, flow_y_px_s
//
//  License: Apache-2.0.  Independent simulation model, not a Prophesee product.
// =====================================================================================

#include <mujoco/mujoco.h>
#include <mujoco/mjplugin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace evk4 {

// ------------------------------------------------------------------ constants / helpers
constexpr double kPi = 3.14159265358979323846;
constexpr double kPhotonsPerLuxM2 = 4.09e15;  // photons / (s m^2) per lux @ 555 nm
constexpr int kHeader = 64;
constexpr int kTile = 16;
constexpr int kNSensorData = 8;
constexpr int kEvStride = 8;                  // doubles per event record
constexpr int kLayoutVersion = 2;
constexpr int PGX = 32, PGY = 18;             // ego-motion probe grid

// pixel state fields (stored in plugin_state)
enum PixField { F_XIN = 0, F_L1, F_L2, F_NOISE, F_REF, F_REFR, F_LAST_T, F_LAST_P, F_AFK_T,
                F_AFK_PER, F_AFK_UNTIL, F_NPIX };

// header slots
enum Hdr { H_NEV = 0, H_CAP, H_W, H_H, H_DROPTOT, H_TLAST, H_FX, H_NSUB, H_INIT, H_TPREV,
           H_SAMPLE, H_NPEND, H_PENDCAP, H_FY, H_CX, H_CY, H_STRIDE, H_VERSION, H_ARB_BUSY,
           H_ERC_WSTART, H_ERC_COUNT, H_ERC_KEEP, H_ERC_ACC, H_TOKEN, H_ERC_OUT,
           H_OFF_EVENTS = 30, H_OFF_PEND, H_OFF_PIX, H_OFF_POSE, H_OFF_PROBE, H_OFF_IMG,
           H_NPIXF, H_BACKEND_EXTERNAL, H_NGEOM };

struct Event {
  int64_t t;       // microseconds (after latency)
  double te;       // physical crossing time (s) - tie-break for ordering
  float depth, fx, fy;
  int32_t geom;
  uint16_t x, y;
  int16_t p;
};

static inline uint64_t mix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static inline double u01(uint64_t key) { return ((mix64(key) >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static inline double gauss(uint64_t key) {
  double a = u01(key), b = u01(key ^ 0xda942042e4dd58b5ULL);
  return std::sqrt(-2.0 * std::log(a)) * std::cos(2.0 * kPi * b);
}
static inline uint64_t keyOf(uint64_t seed, uint64_t pix, uint64_t sample, uint64_t stream) {
  return mix64(seed * 0x632be59bd9b4e019ULL ^ mix64(pix * 0x8cb92ba72f3d8dd7ULL ^
               mix64(sample * 0x9fb21c651e98df25ULL ^ stream)));
}

static const char* cfgStr(const mjModel* m, int inst, const char* key) {
  const char* s = mj_getPluginConfig(m, inst, key);
  return (s && *s) ? s : nullptr;
}
static double cfgNum(const mjModel* m, int inst, const char* key, double def) {
  const char* s = cfgStr(m, inst, key);
  if (!s) return def;
  char* end = nullptr;
  double v = std::strtod(s, &end);
  return (end == s) ? def : v;
}
static std::vector<double> parseVec(const char* s) {
  std::vector<double> out;
  if (!s) return out;
  char* end = nullptr;
  while (*s) {
    double v = std::strtod(s, &end);
    if (end == s) break;
    out.push_back(v);
    s = end;
  }
  return out;
}

static void parallelFor(int n, int nthread, const std::function<void(int, int, int)>& fn) {
  if (nthread <= 1 || n < 64) { fn(0, n, 0); return; }
  std::vector<std::thread> th;
  int chunk = (n + nthread - 1) / nthread;
  for (int t = 0; t < nthread; t++) {
    int b = t * chunk, e = std::min(n, b + chunk);
    if (b >= e) break;
    th.emplace_back(fn, b, e, t);
  }
  for (auto& x : th) x.join();
}
static void parallelDynamic(int n, int nthread, const std::function<void(int)>& fn) {
  if (nthread <= 1) { for (int i = 0; i < n; i++) fn(i); return; }
  std::atomic<int> next(0);
  std::vector<std::thread> th;
  for (int t = 0; t < nthread; t++)
    th.emplace_back([&]() { for (int i; (i = next.fetch_add(1)) < n;) fn(i); });
  for (auto& x : th) x.join();
}

// ------------------------------------------------------------------ parameters
struct Params {
  // sensor / optics
  int width = 1280, height = 720, binning = 1;
  double pixel_pitch_um = 4.86, focal_length_mm = 0, f_number = 2.0, qe = 0.5;
  double fx = 0, fy = 0, cx = -1, cy = -1;          // calibrated intrinsics (full-res px)
  double dist[5] = {0, 0, 0, 0, 0};                // k1 k2 p1 p2 k3 (OpenCV)
  int spp = 1;
  double psf_sigma_px = 0.5, specular_scale = 1.0;
  double rgb_w[3] = {0.2126, 0.7152, 0.0722};      // sensor spectral weights on R,G,B
  // scene radiometry
  double lux_per_unit = 1000.0, background_lux = 50.0, dark_current = 30.0;
  int shadows = 1;
  // photoreceptor & noise
  double cutoff_hz = 2000.0, bw_half_lux = 20.0, sf_cutoff_hz = 0.0;
  int shot_noise = 1;
  double noise_gain = 0.05;
  // change detector
  double contrast_on = 0.22, contrast_off = 0.22, contrast_sigma = 0.02;
  double refractory_us = 50.0, hpf_hz = 0.0;
  double leak_rate_hz = 0.05, leak_sigma = 0.5, temperature_c = 25.0, leak_doubling_c = 7.0;
  double hot_pixel_fraction = 5e-5, hot_pixel_rate_hz = 20.0;
  // Metavision-style biases (integers, 0 = factory default) and their calibration scales
  double bias_diff_on = 0, bias_diff_off = 0, bias_fo = 0, bias_hpf = 0, bias_refr = 0;
  double bias_diff_scale = 100.0, bias_fo_scale = 40.0, bias_hpf_hz_per_unit = 0.5,
         bias_refr_scale = 80.0;
  // readout / ESP
  double latency_us = 20.0, latency_jitter_us = 5.0, readout_mevps = 0.0;
  double erc_mevps = 0.0, erc_window_us = 200.0;
  int afk = 0; double afk_min_hz = 40, afk_max_hz = 500;
  double stc_us = 0, trail_us = 0;
  int roi[4] = {0, 0, 0, 0};                       // x0 y0 x1 y1 (0 0 0 0 = full array)
  int max_events = 400000;
  // lighting modulation
  double flicker_hz = 0.0, flicker_depth = 0.0;
  // time sampling
  double max_px_motion = 1.0;
  int max_substeps = 12;
  // misc
  int export_intensity = 1, show_frustum = 1, nthread = 0, skip_static_tiles = 1;
  int backend_external = 0;
  uint64_t seed = 1;
  unsigned char geomgroup[mjNGROUP] = {1, 1, 1, 0, 0, 0};

  // effective (after bias mapping)
  double eff_con, eff_coff, eff_cutoff, eff_hpf, eff_refr_us, eff_leak;

  static void readBiasFile(const char* path, Params& p) {
    std::ifstream f(path);
    if (!f) { mju_warning("evk4: cannot open bias_file"); return; }
    std::string line;
    while (std::getline(f, line)) {                 // Metavision format: "<value> % <name>"
      auto pos = line.find('%');
      if (pos == std::string::npos) continue;
      double v = std::atof(line.substr(0, pos).c_str());
      std::string name = line.substr(pos + 1);
      name.erase(0, name.find_first_not_of(" \t"));
      name.erase(name.find_last_not_of(" \t\r\n") + 1);
      if (name == "bias_diff_on") p.bias_diff_on = v;
      else if (name == "bias_diff_off") p.bias_diff_off = v;
      else if (name == "bias_fo") p.bias_fo = v;
      else if (name == "bias_hpf") p.bias_hpf = v;
      else if (name == "bias_refr") p.bias_refr = v;
    }
  }

  static Params Read(const mjModel* m, int inst) {
    Params p;
    auto res = parseVec(cfgStr(m, inst, "resolution"));
    if (res.size() >= 2) { p.width = (int)res[0]; p.height = (int)res[1]; }
#define RD(name) p.name = cfgNum(m, inst, #name, p.name)
#define RDI(name) p.name = (int)cfgNum(m, inst, #name, p.name)
    RDI(binning); p.binning = std::max(1, p.binning);
    RD(pixel_pitch_um); RD(focal_length_mm); RD(f_number); RD(qe);
    RD(fx); RD(fy); RD(cx); RD(cy); RDI(spp);
    RD(psf_sigma_px); RD(specular_scale); RD(lux_per_unit); RD(background_lux); RD(dark_current);
    RDI(shadows); RD(cutoff_hz); RD(bw_half_lux); RD(sf_cutoff_hz); RDI(shot_noise); RD(noise_gain);
    RD(contrast_on); RD(contrast_off); RD(contrast_sigma); RD(refractory_us); RD(hpf_hz);
    RD(leak_rate_hz); RD(leak_sigma); RD(temperature_c); RD(leak_doubling_c);
    RD(hot_pixel_fraction); RD(hot_pixel_rate_hz);
    RD(bias_diff_scale); RD(bias_fo_scale); RD(bias_hpf_hz_per_unit); RD(bias_refr_scale);
    RD(latency_us); RD(latency_jitter_us); RD(readout_mevps); RD(erc_mevps); RD(erc_window_us);
    RDI(afk); RD(afk_min_hz); RD(afk_max_hz); RD(stc_us); RD(trail_us);
    RDI(max_events); p.max_events = std::max(1024, p.max_events);
    RD(flicker_hz); RD(flicker_depth); RD(max_px_motion); p.max_px_motion = std::max(0.05, p.max_px_motion);
    RDI(max_substeps); p.max_substeps = std::max(1, p.max_substeps);
    RDI(export_intensity); RDI(show_frustum); RDI(nthread); RDI(skip_static_tiles);
    p.seed = (uint64_t)cfgNum(m, inst, "seed", (double)p.seed);
    // biases: file first, then explicit keys override
    if (const char* bf = cfgStr(m, inst, "bias_file")) readBiasFile(bf, p);
    RD(bias_diff_on); RD(bias_diff_off); RD(bias_fo); RD(bias_hpf); RD(bias_refr);
#undef RD
#undef RDI
    auto dd = parseVec(cfgStr(m, inst, "distortion"));
    for (int i = 0; i < 5 && i < (int)dd.size(); i++) p.dist[i] = dd[i];
    auto rw = parseVec(cfgStr(m, inst, "rgb_weights"));
    if (rw.size() >= 3) for (int i = 0; i < 3; i++) p.rgb_w[i] = rw[i];
    auto ro = parseVec(cfgStr(m, inst, "roi"));
    if (ro.size() >= 4) for (int i = 0; i < 4; i++) p.roi[i] = (int)ro[i];
    auto gg = parseVec(cfgStr(m, inst, "geomgroup"));
    for (int i = 0; i < (int)gg.size() && i < mjNGROUP; i++) p.geomgroup[i] = gg[i] != 0;
    const char* be = cfgStr(m, inst, "backend");
    p.backend_external = be && std::string(be) != "cpu";
    int s = (int)std::lround(std::sqrt((double)std::max(1, p.spp)));
    p.spp = std::max(1, std::min(4, s));            // stored as n (n x n stratified samples)

    // ---- bias -> physical mapping (scales are calibration constants, see README)
    p.eff_con = p.contrast_on * std::exp(p.bias_diff_on / p.bias_diff_scale);
    p.eff_coff = p.contrast_off * std::exp(p.bias_diff_off / p.bias_diff_scale);
    p.eff_cutoff = p.cutoff_hz * std::exp(p.bias_fo / p.bias_fo_scale);
    p.eff_hpf = std::max(0.0, p.hpf_hz + p.bias_hpf_hz_per_unit * p.bias_hpf);
    p.eff_refr_us = p.refractory_us * std::exp(-p.bias_refr / p.bias_refr_scale);
    p.eff_leak = p.leak_rate_hz * std::pow(2.0, (p.temperature_c - 25.0) / p.leak_doubling_c);
    return p;
  }
  int W() const { return width / binning; }
  int H() const { return height / binning; }
  int pendCap() const { return max_events; }
};

struct Layout {
  int events, pend, pix, pose, probe, img, total;
  static Layout Make(const Params& p, int ngeom) {
    Layout L;
    int N = p.W() * p.H();
    L.events = kHeader;
    if (p.backend_external) {
      L.pend = L.pix = L.pose = L.probe = L.img = L.events + kEvStride * p.max_events;
      L.total = L.img;
      return L;
    }
    L.pend = L.events + kEvStride * p.max_events;
    L.pix = L.pend + kEvStride * p.pendCap();
    L.pose = L.pix + F_NPIX * N;
    L.probe = L.pose + 12 * ngeom + 12;
    L.img = L.probe + 3 * PGX * PGY;
    L.total = L.img + (p.export_intensity ? N : 0);
    return L;
  }
};

// ------------------------------------------------------------------ the sensor
class EventCamera {
 public:
  static EventCamera* Create(const mjModel* m, mjData* d, int instance);
  void Compute(const mjModel* m, mjData* d, int instance);
  void Visualize(const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn, int instance);

 private:
  Params P_;
  Layout Lo_;
  int W_ = 0, H_ = 0, N_ = 0, cam_ = -1, sensor_ = -1, cambody_ = -1, nthread_ = 1;
  double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0, K_rate_ = 0, r_half_ = 0;
  // geometry of the pixel array (undistorted normalised coords, OpenCV convention y down)
  std::vector<float> corner_;      // (W+1)*(H+1)*2  pixel-corner normalised coords
  std::vector<float> ctr_;         // N*2            pixel-centre normalised coords
  std::vector<float> cos4_;
  std::vector<float> tileB_;       // ntile*4        normalised bounds of each tile
  // fixed pattern
  std::vector<float> th_on_, th_off_, leak_;
  // scratch
  std::vector<float> rate_, raw_, tmp_;
  std::vector<int> lblG_;
  std::vector<float> lblD_, lblFx_, lblFy_;
  std::vector<float> tileHit_;     // ntile*6 hit AABB (min,max)
  std::vector<unsigned char> tileValid_;
  std::vector<mjtNum> rx_pos_, rx_mat_, rl_pos_, rl_dir_;  // poses at last render
  mjtNum rc_pos_[3] = {0, 0, 0}, rc_mat_[9] = {0};
  double token_ = -1;
  uint64_t uid_ = 0, tokctr_ = 0, app_hash_ = 0;                // per-object identity for cache validation
  std::vector<int> vis_geoms_;
  std::vector<std::vector<Event>> tev_;
  // profiling
  double prof_render_ = 0, prof_update_ = 0, prof_readout_ = 0;
  long prof_n_ = 0, prof_tiles_ = 0, prof_tiles_dirty_ = 0;
  bool profile_ = std::getenv("EVK4_PROFILE") != nullptr;

  bool Init(const mjModel* m, mjData* d, int instance);
  void Undistort(double xd, double yd, double* xu, double* yu) const;
  bool ProjectPx(const mjtNum* cpos, const mjtNum* cmat, const mjtNum w[3], double* u, double* v) const;
  bool ProjectN(const mjtNum* cpos, const mjtNum* cmat, const mjtNum w[3], double* x, double* y) const;
  double TraceMesh(const mjModel* m, const mjtNum* gpos, const mjtNum* gmat, int g,
                   const mjtNum pnt[3], const mjtNum vec[3], double tmax, mjtNum nrm[3],
                   float uv[2], bool* hasuv) const;
  double Albedo(const mjModel* m, const mjData* d, int g, const mjtNum P[3], const float* uv,
                bool hasuv, double* spec, double* shin, double* emis) const;
  bool Occluded(const mjModel* m, const mjData* d, const mjtNum P[3], const mjtNum dir[3],
                double maxd, const std::vector<int>& occ) const;
  void Shade(const mjModel* m, const mjData* d, int g, const mjtNum P[3], mjtNum n[3],
             const mjtNum vdir[3], const float* uv, bool hasuv, double flick,
             const std::vector<std::vector<int>>& occ, double* Lv) const;
  void Render(const mjModel* m, const mjData* d, const mjtNum* cpos, const mjtNum* cmat, double t,
              const mjtNum* pxpos, const mjtNum* pxmat, const mjtNum* pcpos, const mjtNum* pcmat,
              double dtflow, mjtNum* probe, bool force);
  void Blur();
  int MotionSubsteps(const mjModel* m, const mjData* d, const mjtNum* st, double dt) const;
  void UpdatePixels(mjtNum* st, double ta, double tb, uint64_t sample);
  void Readout(mjtNum* st, std::vector<Event>& evs, std::vector<Event>& pend, double t1, long long* dropped);
};

// ----------------------------------------------------------------- creation / init
EventCamera* EventCamera::Create(const mjModel* m, mjData* d, int instance) {
  auto* ec = new EventCamera();
  if (!ec->Init(m, d, instance)) { delete ec; return nullptr; }
  return ec;
}

void EventCamera::Undistort(double xd, double yd, double* xu, double* yu) const {
  const double* k = P_.dist;
  double x = xd, y = yd;
  if (k[0] == 0 && k[1] == 0 && k[2] == 0 && k[3] == 0 && k[4] == 0) { *xu = x; *yu = y; return; }
  for (int it = 0; it < 20; it++) {                // OpenCV-style fixed point iteration
    double r2 = x * x + y * y;
    double rad = 1 + k[0] * r2 + k[1] * r2 * r2 + k[4] * r2 * r2 * r2;
    double dx = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x);
    double dy = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y;
    x = (xd - dx) / rad; y = (yd - dy) / rad;
  }
  *xu = x; *yu = y;
}

bool EventCamera::ProjectN(const mjtNum* cpos, const mjtNum* cmat, const mjtNum w[3], double* x,
                           double* y) const {
  mjtNum d[3] = {w[0] - cpos[0], w[1] - cpos[1], w[2] - cpos[2]}, c[3];
  mju_mulMatTVec3(c, cmat, d);                     // camera: x right, y up, -z forward
  double z = -c[2];
  if (z < 1e-4) return false;
  *x = c[0] / z; *y = -c[1] / z;                   // normalised, y down
  return true;
}

bool EventCamera::ProjectPx(const mjtNum* cpos, const mjtNum* cmat, const mjtNum w[3], double* u,
                            double* v) const {
  double x, y;
  if (!ProjectN(cpos, cmat, w, &x, &y)) return false;
  const double* k = P_.dist;
  double r2 = x * x + y * y, rad = 1 + k[0] * r2 + k[1] * r2 * r2 + k[4] * r2 * r2 * r2;
  double xd = x * rad + 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x);
  double yd = y * rad + k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y;
  *u = cx_ + fx_ * xd; *v = cy_ + fy_ * yd;
  return true;
}

bool EventCamera::Init(const mjModel* m, mjData* d, int instance) {
  static std::atomic<uint64_t> g_uid(1);
  uid_ = g_uid.fetch_add(1);
  P_ = Params::Read(m, instance);
  W_ = P_.W(); H_ = P_.H(); N_ = W_ * H_;
  Lo_ = Layout::Make(P_, m->ngeom);
  for (int i = 0; i < m->nsensor; i++)
    if (m->sensor_type[i] == mjSENS_PLUGIN && m->sensor_plugin[i] == instance) { sensor_ = i; break; }
  if (sensor_ < 0) { mju_warning("evk4: no sensor bound to plugin instance"); return false; }
  if (m->sensor_objtype[sensor_] != mjOBJ_CAMERA) { mju_warning("evk4: objtype must be camera"); return false; }
  cam_ = m->sensor_objid[sensor_];
  cambody_ = m->cam_bodyid[cam_];
  if (P_.backend_external) return true;

  // ---- intrinsics (calibrated values are given in full-resolution pixels)
  double pitch = P_.pixel_pitch_um * 1e-6 * P_.binning;
  double b = P_.binning;
  if (P_.fx > 0) { fx_ = P_.fx / b; fy_ = (P_.fy > 0 ? P_.fy : P_.fx) / b; }
  else if (P_.focal_length_mm > 0) { fx_ = fy_ = P_.focal_length_mm * 1e-3 / pitch; }
  else { fx_ = fy_ = 0.5 * H_ / std::tan(0.5 * m->cam_fovy[cam_] * kPi / 180.0); }
  cx_ = P_.cx >= 0 ? (P_.cx + 0.5) / b - 0.5 : 0.5 * (W_ - 1);
  cy_ = P_.cy >= 0 ? (P_.cy + 0.5) / b - 0.5 : 0.5 * (H_ - 1);
  K_rate_ = kPi / (4.0 * P_.f_number * P_.f_number) * kPhotonsPerLuxM2 * pitch * pitch * P_.qe;
  r_half_ = K_rate_ * P_.bw_half_lux / kPi;

  // ---- per-pixel ray geometry (undistorted normalised coordinates)
  corner_.resize(2 * (W_ + 1) * (H_ + 1));
  for (int v = 0; v <= H_; v++)
    for (int u = 0; u <= W_; u++) {
      double xu, yu;
      Undistort((u - 0.5 - cx_) / fx_, (v - 0.5 - cy_) / fy_, &xu, &yu);
      corner_[2 * (v * (W_ + 1) + u)] = (float)xu;
      corner_[2 * (v * (W_ + 1) + u) + 1] = (float)yu;
    }
  ctr_.resize(2 * N_); cos4_.resize(N_);
  for (int v = 0; v < H_; v++)
    for (int u = 0; u < W_; u++) {
      double xu, yu;
      Undistort((u - cx_) / fx_, (v - cy_) / fy_, &xu, &yu);
      int i = v * W_ + u;
      ctr_[2 * i] = (float)xu; ctr_[2 * i + 1] = (float)yu;
      double c2 = 1.0 / (1.0 + xu * xu + yu * yu);
      cos4_[i] = (float)(c2 * c2);
    }
  int TW = (W_ + kTile - 1) / kTile, TH = (H_ + kTile - 1) / kTile;
  tileB_.resize(4 * TW * TH);
  for (int ty = 0; ty < TH; ty++)
    for (int tx = 0; tx < TW; tx++) {
      float b4[4] = {1e9f, 1e9f, -1e9f, -1e9f};
      for (int v = ty * kTile; v <= std::min(H_, (ty + 1) * kTile); v++)
        for (int u = tx * kTile; u <= std::min(W_, (tx + 1) * kTile); u++) {
          float x = corner_[2 * (v * (W_ + 1) + u)], y = corner_[2 * (v * (W_ + 1) + u) + 1];
          b4[0] = std::min(b4[0], x); b4[1] = std::min(b4[1], y);
          b4[2] = std::max(b4[2], x); b4[3] = std::max(b4[3], y);
        }
      for (int k = 0; k < 4; k++) tileB_[4 * (ty * TW + tx) + k] = b4[k];
    }
  tileHit_.assign(6 * TW * TH, 0);
  tileValid_.assign(TW * TH, 0);

  // ---- fixed pattern noise
  th_on_.resize(N_); th_off_.resize(N_); leak_.resize(N_);
  for (int i = 0; i < N_; i++) {
    th_on_[i] = (float)std::max(0.03, P_.eff_con + P_.contrast_sigma * gauss(keyOf(P_.seed, i, 0, 11)));
    th_off_[i] = (float)std::max(0.03, P_.eff_coff + P_.contrast_sigma * gauss(keyOf(P_.seed, i, 0, 12)));
    double s = P_.leak_sigma;
    double lk = P_.eff_leak * std::exp(s * gauss(keyOf(P_.seed, i, 0, 13)) - 0.5 * s * s);
    if (u01(keyOf(P_.seed, i, 0, 14)) < P_.hot_pixel_fraction) lk = P_.hot_pixel_rate_hz;
    leak_[i] = (float)lk;
  }
  // ---- geoms visible to the sensor
  for (int g = 0; g < m->ngeom; g++) {
    int grp = m->geom_group[g];
    if (grp < 0 || grp >= mjNGROUP || !P_.geomgroup[grp]) continue;
    if (m->geom_bodyid[g] == cambody_) continue;
    int mat = m->geom_matid[g];
    float alpha = mat >= 0 ? m->mat_rgba[4 * mat + 3] : m->geom_rgba[4 * g + 3];
    if (m->geom_rgba[4 * g + 3] == 0) alpha = 0;
    if (alpha <= 0) continue;
    int t = m->geom_type[g];
    if (t == mjGEOM_HFIELD || t == mjGEOM_SDF) continue;
    if (t == mjGEOM_MESH && m->geom_dataid[g] < 0) continue;
    vis_geoms_.push_back(g);
  }
  rate_.assign(N_, 0); raw_.assign(N_, 0); tmp_.assign(N_, 0);
  lblG_.assign(N_, -1); lblD_.assign(N_, 0); lblFx_.assign(N_, 0); lblFy_.assign(N_, 0);
  rx_pos_.assign(3 * m->ngeom, 0); rx_mat_.assign(9 * m->ngeom, 0);
  rl_pos_.assign(3 * m->nlight, 0); rl_dir_.assign(3 * m->nlight, 0);
  nthread_ = P_.nthread > 0 ? P_.nthread : std::max(1u, std::thread::hardware_concurrency());
  tev_.resize(nthread_);
  if (m->vis.global.bvactive)
    mju_warning("evk4: set <visual><global bvactive=\"false\"/> (thread-safe ray casting)");
  return true;
}

// ----------------------------------------------------------------- mesh ray casting
// Own traversal of MuJoCo's mesh BVH: returns distance, interpolated (smooth) normal in world
// frame, and interpolated texture coordinates.  Prunes nodes farther than the current best.
double EventCamera::TraceMesh(const mjModel* m, const mjtNum* gpos, const mjtNum* gmat, int g,
                              const mjtNum pnt[3], const mjtNum vec[3], double tmax,
                              mjtNum nrm[3], float uv[2], bool* hasuv) const {
  *hasuv = false;
  const int meshid = m->geom_dataid[g];
  const int bvhadr = m->mesh_bvhadr[meshid];
  if (bvhadr < 0) return -1;
  const mjtNum* bvh = m->bvh_aabb + 6 * bvhadr;
  const int* child = m->bvh_child + 2 * bvhadr;
  const int* faceid = m->bvh_nodeid + bvhadr;
  mjtNum dp[3] = {pnt[0] - gpos[0], pnt[1] - gpos[1], pnt[2] - gpos[2]}, lp[3], lv[3];
  mju_mulMatTVec3(lp, gmat, dp);
  mju_mulMatTVec3(lv, gmat, vec);
  double inv[3] = {1.0 / lv[0], 1.0 / lv[1], 1.0 / lv[2]};
  int stack[mjMAXTREEDEPTH + 2];
  int ns = 0;
  stack[ns++] = 0;
  double best = tmax > 0 ? tmax : 1e30;
  int bestFace = -1; double bu = 0, bv = 0;
  const int vadr = m->mesh_vertadr[meshid], fadr = m->mesh_faceadr[meshid];
  while (ns) {
    int node = stack[--ns];
    const mjtNum* bb = bvh + 6 * node;
    double t0 = 0, t1 = best;
    bool hit = true;
    for (int k = 0; k < 3; k++) {
      double a = (bb[k] - bb[k + 3] - lp[k]) * inv[k], b = (bb[k] + bb[k + 3] - lp[k]) * inv[k];
      if (a > b) std::swap(a, b);
      t0 = std::max(t0, a); t1 = std::min(t1, b);
      if (t0 > t1) { hit = false; break; }
    }
    if (!hit) continue;
    if (faceid[node] != -1) {
      int f = faceid[node] + fadr;
      const float* A = m->mesh_vert + 3 * (vadr + m->mesh_face[3 * f]);
      const float* B = m->mesh_vert + 3 * (vadr + m->mesh_face[3 * f + 1]);
      const float* C = m->mesh_vert + 3 * (vadr + m->mesh_face[3 * f + 2]);
      double e1[3] = {B[0] - A[0], B[1] - A[1], B[2] - A[2]};
      double e2[3] = {C[0] - A[0], C[1] - A[1], C[2] - A[2]};
      double pv[3] = {lv[1] * e2[2] - lv[2] * e2[1], lv[2] * e2[0] - lv[0] * e2[2], lv[0] * e2[1] - lv[1] * e2[0]};
      double det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
      if (std::fabs(det) < 1e-14) continue;
      double id = 1.0 / det;
      double tv[3] = {lp[0] - A[0], lp[1] - A[1], lp[2] - A[2]};
      double u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * id;
      if (u < 0 || u > 1) continue;
      double qv[3] = {tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]};
      double v = (lv[0] * qv[0] + lv[1] * qv[1] + lv[2] * qv[2]) * id;
      if (v < 0 || u + v > 1) continue;
      double t = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * id;
      if (t > 1e-7 && t < best) { best = t; bestFace = f; bu = u; bv = v; }
      continue;
    }
    for (int c = 0; c < 2; c++)
      if (child[2 * node + c] != -1 && ns < mjMAXTREEDEPTH) stack[ns++] = child[2 * node + c];
  }
  if (bestFace < 0) return -1;
  if (nrm) {
    double w0 = 1 - bu - bv, ln[3];
    const int nadr = m->mesh_normaladr[meshid];
    const int* fn = m->mesh_facenormal + 3 * bestFace;
    const float* n0 = m->mesh_normal + 3 * (nadr + fn[0]);
    const float* n1 = m->mesh_normal + 3 * (nadr + fn[1]);
    const float* n2 = m->mesh_normal + 3 * (nadr + fn[2]);
    for (int k = 0; k < 3; k++) ln[k] = w0 * n0[k] + bu * n1[k] + bv * n2[k];
    double nn = std::sqrt(ln[0] * ln[0] + ln[1] * ln[1] + ln[2] * ln[2]);
    if (nn < 1e-12) {  // fall back to face normal
      const float* A = m->mesh_vert + 3 * (vadr + m->mesh_face[3 * bestFace]);
      const float* B = m->mesh_vert + 3 * (vadr + m->mesh_face[3 * bestFace + 1]);
      const float* C = m->mesh_vert + 3 * (vadr + m->mesh_face[3 * bestFace + 2]);
      double e1[3] = {B[0] - A[0], B[1] - A[1], B[2] - A[2]}, e2[3] = {C[0] - A[0], C[1] - A[1], C[2] - A[2]};
      ln[0] = e1[1] * e2[2] - e1[2] * e2[1]; ln[1] = e1[2] * e2[0] - e1[0] * e2[2]; ln[2] = e1[0] * e2[1] - e1[1] * e2[0];
      nn = std::sqrt(ln[0] * ln[0] + ln[1] * ln[1] + ln[2] * ln[2]) + 1e-30;
    }
    mjtNum lnn[3] = {ln[0] / nn, ln[1] / nn, ln[2] / nn};
    mju_mulMatVec3(nrm, gmat, lnn);
  }
  const int tadr = m->mesh_texcoordadr[meshid];
  if (uv && tadr >= 0) {
    double w0 = 1 - bu - bv;
    const int* ft = m->mesh_facetexcoord + 3 * bestFace;
    const float* t0 = m->mesh_texcoord + 2 * (tadr + ft[0]);
    const float* t1 = m->mesh_texcoord + 2 * (tadr + ft[1]);
    const float* t2 = m->mesh_texcoord + 2 * (tadr + ft[2]);
    uv[0] = (float)(w0 * t0[0] + bu * t1[0] + bv * t2[0]);
    uv[1] = (float)(w0 * t0[1] + bu * t1[1] + bv * t2[1]);
    *hasuv = true;
  }
  return best;
}

// ----------------------------------------------------------------- materials / shading
static void texSample(const mjModel* m, int tex, double u, double v, float out[3]) {
  int w = m->tex_width[tex], h = m->tex_height[tex], nc = m->tex_nchannel[tex];
  const unsigned char* data = m->tex_data + m->tex_adr[tex];
  u -= std::floor(u); v -= std::floor(v);
  double fu = u * w - 0.5, fv = v * h - 0.5;
  int u0 = (int)std::floor(fu), v0 = (int)std::floor(fv);
  double au = fu - u0, av = fv - v0;
  float acc[3] = {0, 0, 0};
  for (int k = 0; k < 4; k++) {
    int uu = u0 + (k & 1), vv = v0 + (k >> 1);
    uu = ((uu % w) + w) % w; vv = ((vv % h) + h) % h;
    double wt = ((k & 1) ? au : 1 - au) * ((k >> 1) ? av : 1 - av);
    const unsigned char* px = data + (size_t)nc * (vv * w + uu);
    for (int c = 0; c < 3; c++) acc[c] += (float)(wt * px[nc >= 3 ? c : 0] / 255.0);
  }
  out[0] = acc[0]; out[1] = acc[1]; out[2] = acc[2];
}

double EventCamera::Albedo(const mjModel* m, const mjData* d, int g, const mjtNum P[3],
                           const float* uvm, bool hasuv, double* spec, double* shin,
                           double* emis) const {
  int mat = m->geom_matid[g];
  const float* grgba = m->geom_rgba + 4 * g;
  bool geomDefault = std::fabs(grgba[0] - 0.5f) < 1e-6 && std::fabs(grgba[1] - 0.5f) < 1e-6 &&
                     std::fabs(grgba[2] - 0.5f) < 1e-6;
  float rgb[3] = {grgba[0], grgba[1], grgba[2]};
  *spec = 0.5; *shin = 0.5; *emis = 0;
  if (mat >= 0) {
    if (geomDefault) { rgb[0] = m->mat_rgba[4*mat]; rgb[1] = m->mat_rgba[4*mat+1]; rgb[2] = m->mat_rgba[4*mat+2]; }
    *spec = m->mat_specular[mat]; *shin = m->mat_shininess[mat]; *emis = m->mat_emission[mat];
    int tex = m->mat_texid[mat * mjNTEXROLE + mjTEXROLE_RGB];
    if (tex < 0) tex = m->mat_texid[mat * mjNTEXROLE + mjTEXROLE_RGBA];
    if (tex >= 0 && m->tex_type[tex] == mjTEXTURE_2D) {
      const mjtNum* sz = m->geom_size + 3 * g;
      double rep0 = m->mat_texrepeat[2*mat], rep1 = m->mat_texrepeat[2*mat+1];
      bool uni = m->mat_texuniform[mat];
      double scl0 = rep0 > 0 ? rep0 : 1, scl1 = rep1 > 0 ? rep1 : 1;
      int type = m->geom_type[g];
      double uu = 0, vv = 0; bool ok = true;
      if (type == mjGEOM_MESH) {
        if (hasuv) { uu = uvm[0]; vv = uvm[1]; } else ok = false;
      } else {
        if (uni) { if (sz[0] > 0) scl0 *= sz[0]; if (sz[1] > 0) scl1 *= sz[1]; }
        mjtNum dl[3] = {P[0] - d->geom_xpos[3*g], P[1] - d->geom_xpos[3*g+1], P[2] - d->geom_xpos[3*g+2]}, l[3];
        mju_mulMatTVec3(l, d->geom_xmat + 9 * g, dl);
        if (type == mjGEOM_PLANE) {
          uu = sz[0] > 0 ? (l[0] + sz[0]) / (2 * sz[0]) : 0.5 * l[0];
          vv = sz[1] > 0 ? 1.0 - (l[1] + sz[1]) / (2 * sz[1]) : -0.5 * l[1];
        } else if (type == mjGEOM_BOX) {
          double a[3] = {l[0] / sz[0], l[1] / sz[1], l[2] / sz[2]};
          int ax = 2;
          if (std::fabs(a[0]) >= std::fabs(a[1]) && std::fabs(a[0]) >= std::fabs(a[2])) ax = 0;
          else if (std::fabs(a[1]) >= std::fabs(a[2])) ax = 1;
          double p, q;
          if (ax == 2) { p = a[0]; q = a[1]; } else if (ax == 0) { p = a[1]; q = a[2]; } else { p = a[0]; q = a[2]; }
          uu = (p + 1) / 2; vv = 1.0 - (q + 1) / 2;
        } else if (type == mjGEOM_CYLINDER) {
          if (std::fabs(l[2]) > sz[1] * 0.999) { uu = 0.5 + 0.5 * l[0] / sz[0]; vv = 0.5 + 0.5 * l[1] / sz[0]; }
          else { uu = std::atan2(l[1], l[0]) / (2 * kPi) + 0.5; vv = 0.5 + 0.5 * l[2] / sz[1]; }
        } else if (type == mjGEOM_SPHERE || type == mjGEOM_ELLIPSOID) {
          double r = std::sqrt(l[0]*l[0] + l[1]*l[1] + l[2]*l[2]) + 1e-12;
          uu = std::atan2(l[1], l[0]) / (2 * kPi) + 0.5;
          vv = std::acos(std::max(-1.0, std::min(1.0, l[2] / r))) / kPi;
        } else ok = false;
      }
      if (ok) {
        float t3[3];
        texSample(m, tex, uu * scl0, vv * scl1, t3);
        rgb[0] *= t3[0]; rgb[1] *= t3[1]; rgb[2] *= t3[2];
      }
    }
  }
  const double wn = P_.rgb_w[0] + P_.rgb_w[1] + P_.rgb_w[2];
  return (P_.rgb_w[0] * rgb[0] + P_.rgb_w[1] * rgb[1] + P_.rgb_w[2] * rgb[2]) / wn;
}

static inline double wsum(const double* w, const float* c) { return w[0] * c[0] + w[1] * c[1] + w[2] * c[2]; }

bool EventCamera::Occluded(const mjModel* m, const mjData* d, const mjtNum P[3],
                           const mjtNum dir[3], double maxd, const std::vector<int>& occ) const {
  for (int g : occ) {
    int type = m->geom_type[g];
    const mjtNum* c = d->geom_xpos + 3 * g;
    if (type != mjGEOM_PLANE) {
      double rb = m->geom_rbound[g];
      double oc[3] = {c[0] - P[0], c[1] - P[1], c[2] - P[2]};
      double tp = oc[0]*dir[0] + oc[1]*dir[1] + oc[2]*dir[2];
      if (tp < -rb || tp - rb > maxd) continue;
      double perp2 = oc[0]*oc[0] + oc[1]*oc[1] + oc[2]*oc[2] - tp*tp;
      if (perp2 > rb * rb) continue;
    }
    double dist;
    if (type == mjGEOM_MESH) {
      bool hu;
      dist = TraceMesh(m, c, d->geom_xmat + 9 * g, g, P, dir, maxd, nullptr, nullptr, &hu);
    } else {
      dist = mju_rayGeom(c, d->geom_xmat + 9 * g, m->geom_size + 3 * g, P, dir, type, nullptr);
    }
    if (dist > 0 && dist < maxd) return true;
  }
  return false;
}

void EventCamera::Shade(const mjModel* m, const mjData* d, int g, const mjtNum P[3], mjtNum n[3],
                        const mjtNum vdir[3], const float* uv, bool hasuv, double flick,
                        const std::vector<std::vector<int>>& occ, double* Lv) const {
  double spec, shin, emis;
  double rho = Albedo(m, d, g, P, uv, hasuv, &spec, &shin, &emis);
  if (n[0]*vdir[0] + n[1]*vdir[1] + n[2]*vdir[2] > 0) { n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; }
  mjtNum Po[3] = {P[0] + 2e-4 * n[0], P[1] + 2e-4 * n[1], P[2] + 2e-4 * n[2]};
  double Ediff = 0, Espec = 0;
  const double wn = P_.rgb_w[0] + P_.rgb_w[1] + P_.rgb_w[2];
  for (int l = 0; l < m->nlight; l++) {
    if (!m->light_active[l]) continue;
    Ediff += wsum(P_.rgb_w, m->light_ambient + 3 * l) / wn * P_.lux_per_unit;
    double dif = wsum(P_.rgb_w, m->light_diffuse + 3 * l) / wn * P_.lux_per_unit * flick;
    double spl = wsum(P_.rgb_w, m->light_specular + 3 * l) / wn * P_.lux_per_unit * flick;
    if (dif <= 0 && spl <= 0) continue;
    const mjtNum* lp = d->light_xpos + 3 * l;
    const mjtNum* ld = d->light_xdir + 3 * l;
    mjtNum L[3];
    double dist = 1e9, att = 1.0, spot = 1.0;
    int type = m->light_type[l];
    if (type == mjLIGHT_DIRECTIONAL) { L[0] = -ld[0]; L[1] = -ld[1]; L[2] = -ld[2]; }
    else {
      L[0] = lp[0] - P[0]; L[1] = lp[1] - P[1]; L[2] = lp[2] - P[2];
      dist = mju_normalize3(L);
      const float* a = m->light_attenuation + 3 * l;
      att = 1.0 / std::max(1e-6, (double)a[0] + a[1] * dist + a[2] * dist * dist);
      if (type == mjLIGHT_SPOT) {
        double ca = -(L[0]*ld[0] + L[1]*ld[1] + L[2]*ld[2]);
        double cut = std::cos(m->light_cutoff[l] * kPi / 180.0);
        spot = ca < cut ? 0.0 : std::pow(std::max(0.0, ca), (double)m->light_exponent[l]);
      }
    }
    double ndl = n[0]*L[0] + n[1]*L[1] + n[2]*L[2];
    if (ndl <= 0 || spot <= 0) continue;
    if (P_.shadows && m->light_castshadow[l] && Occluded(m, d, Po, L, dist, occ[l])) continue;
    Ediff += dif * att * spot * ndl;
    mjtNum h[3] = {L[0] - vdir[0], L[1] - vdir[1], L[2] - vdir[2]};
    mju_normalize3(h);
    double ndh = std::max(0.0, n[0]*h[0] + n[1]*h[1] + n[2]*h[2]);
    Espec += spl * att * spot * std::pow(ndh, std::max(1.0, shin * 128.0)) * ndl;
  }
  *Lv = (rho * Ediff + P_.specular_scale * spec * Espec + emis * rho * P_.lux_per_unit) / kPi;
}

// ----------------------------------------------------------------- renderer
// Renders the photo-electron rate of every pixel into raw_ (then PSF -> rate_).  Tiles whose
// primary/shadow dependencies did not move since the last render are reused (unless force).
// Also produces per-pixel labels: geom id, depth and exact optical flow w.r.t. (pxpos, pcpos).
void EventCamera::Render(const mjModel* m, const mjData* d, const mjtNum* cpos, const mjtNum* cmat,
                         double t, const mjtNum* pxpos, const mjtNum* pxmat, const mjtNum* pcpos,
                         const mjtNum* pcmat, double dtflow, mjtNum* probe, bool force) {
  auto tr0 = std::chrono::steady_clock::now();
  const int TW = (W_ + kTile - 1) / kTile, TH = (H_ + kTile - 1) / kTile, NT = TW * TH;
  const double flick = (P_.flicker_hz > 0 && P_.flicker_depth > 0)
      ? 1.0 - P_.flicker_depth * 0.5 * (1.0 + std::cos(2 * kPi * P_.flicker_hz * t)) : 1.0;

  // ---- what moved since the last render?
  auto moved = [](const mjtNum* a, const mjtNum* b, int n) {
    for (int k = 0; k < n; k++) if (std::fabs(a[k] - b[k]) > 1e-12) return true;
    return false;
  };
  // runtime edits of appearance (lights, colours, sizes, materials) invalidate all tiles
  uint64_t hsh = 1469598103934665603ULL;
  auto hmix = [&](const void* p, size_t n) {
    const unsigned char* c = (const unsigned char*)p;
    for (size_t k = 0; k < n; k++) { hsh ^= c[k]; hsh *= 1099511628211ULL; }
  };
  hmix(m->light_diffuse, sizeof(float) * 3 * m->nlight); hmix(m->light_ambient, sizeof(float) * 3 * m->nlight);
  hmix(m->light_specular, sizeof(float) * 3 * m->nlight); hmix(m->light_active, sizeof(mjtBool) * m->nlight);
  hmix(m->light_cutoff, sizeof(float) * m->nlight); hmix(m->light_exponent, sizeof(float) * m->nlight);
  hmix(m->light_attenuation, sizeof(float) * 3 * m->nlight); hmix(m->light_castshadow, sizeof(mjtBool) * m->nlight);
  hmix(m->geom_rgba, sizeof(float) * 4 * m->ngeom); hmix(m->geom_size, sizeof(mjtNum) * 3 * m->ngeom);
  hmix(m->geom_matid, sizeof(int) * m->ngeom);
  hmix(m->mat_rgba, sizeof(float) * 4 * m->nmat); hmix(m->mat_specular, sizeof(float) * m->nmat);
  hmix(m->mat_emission, sizeof(float) * m->nmat);
  const bool appearanceChanged = hsh != app_hash_;
  app_hash_ = hsh;
  bool camMoved = force || appearanceChanged || moved(cpos, rc_pos_, 3) || moved(cmat, rc_mat_, 9) || flick != 1.0;
  for (int l = 0; l < m->nlight && !camMoved; l++)
    if (moved(d->light_xpos + 3 * l, rl_pos_.data() + 3 * l, 3) || moved(d->light_xdir + 3 * l, rl_dir_.data() + 3 * l, 3))
      camMoved = true;
  std::vector<int> movedG;
  for (int g : vis_geoms_)
    if (moved(d->geom_xpos + 3 * g, rx_pos_.data() + 3 * g, 3) || moved(d->geom_xmat + 9 * g, rx_mat_.data() + 9 * g, 9))
      movedG.push_back(g);

  // ---- screen-space culling (normalised coordinates) of every visible geom
  auto geomRect = [&](const mjtNum* gpos, const mjtNum* gmat, int g, double r[4]) -> int {
    // returns 0: not visible, 1: rect valid, 2: straddles camera (treat as global)
    double rb = m->geom_rbound[g];
    mjtNum dv[3] = {gpos[0] - cpos[0], gpos[1] - cpos[1], gpos[2] - cpos[2]}, cc[3];
    mju_mulMatTVec3(cc, cmat, dv);
    double z = -cc[2];
    if (z + rb < 1e-3) return 0;
    if (z - rb < 1e-3) return 2;
    r[0] = r[1] = 1e30; r[2] = r[3] = -1e30;
    const mjtNum* ab = m->geom_aabb + 6 * g;
    bool allFront = true;
    for (int c8 = 0; c8 < 8; c8++) {
      mjtNum loc[3] = {ab[0] + ((c8 & 1) ? ab[3] : -ab[3]), ab[1] + ((c8 & 2) ? ab[4] : -ab[4]),
                       ab[2] + ((c8 & 4) ? ab[5] : -ab[5])}, w[3];
      mju_mulMatVec3(w, gmat, loc); mju_addTo3(w, gpos);
      double x, y;
      if (!ProjectN(cpos, cmat, w, &x, &y)) { allFront = false; break; }
      r[0] = std::min(r[0], x); r[1] = std::min(r[1], y); r[2] = std::max(r[2], x); r[3] = std::max(r[3], y);
    }
    if (!allFront) {
      double x = cc[0] / z, y = -cc[1] / z, rr = rb / std::sqrt(z * z - rb * rb);
      rr *= 1.0 + x * x + y * y;
      r[0] = x - rr; r[1] = y - rr; r[2] = x + rr; r[3] = y + rr;
    }
    double mg = 3.0 / fx_;
    r[0] -= mg; r[1] -= mg; r[2] += mg; r[3] += mg;
    return 1;
  };
  auto overl = [&](const double r[4], int tile) {
    const float* b = &tileB_[4 * tile];
    return !(r[2] < b[0] || r[0] > b[2] || r[3] < b[1] || r[1] > b[3]);
  };
  std::vector<std::vector<int>> tiles(NT);
  std::vector<int> global;
  for (int g : vis_geoms_) {
    if (m->geom_type[g] == mjGEOM_PLANE) { global.push_back(g); continue; }
    double r[4];
    int k = geomRect(d->geom_xpos + 3 * g, d->geom_xmat + 9 * g, g, r);
    if (k == 0) continue;
    if (k == 2) { global.push_back(g); continue; }
    for (int tl = 0; tl < NT; tl++) if (overl(r, tl)) tiles[tl].push_back(g);
  }

  // ---- dirty tiles
  std::vector<unsigned char> dirty(NT, 0);
  bool all = camMoved || !P_.skip_static_tiles;
  for (int tl = 0; tl < NT; tl++) dirty[tl] = all || !tileValid_[tl];
  if (!all && !movedG.empty()) {
    for (int g : movedG) {
      if (m->geom_type[g] == mjGEOM_PLANE) { std::fill(dirty.begin(), dirty.end(), 1); break; }
      double r0[4], r1[4];
      int k1 = geomRect(d->geom_xpos + 3 * g, d->geom_xmat + 9 * g, g, r1);
      int k0 = geomRect(rx_pos_.data() + 3 * g, rx_mat_.data() + 9 * g, g, r0);
      if (k0 == 2 || k1 == 2) { std::fill(dirty.begin(), dirty.end(), 1); break; }
      for (int tl = 0; tl < NT; tl++)
        if ((k1 == 1 && overl(r1, tl)) || (k0 == 1 && overl(r0, tl))) dirty[tl] = 1;
    }
    // moving shadow casters: capsule from the tile's hit bounds to each shadow light
    if (P_.shadows)
      for (int tl = 0; tl < NT; tl++) {
        if (dirty[tl] || !tileValid_[tl]) continue;
        const float* hb = &tileHit_[6 * tl];
        if (hb[3] < hb[0]) continue;
        mjtNum C[3], R = 0;
        for (int j = 0; j < 3; j++) { C[j] = 0.5 * (hb[j] + hb[j + 3]); R += 0.25 * (hb[j + 3] - hb[j]) * (hb[j + 3] - hb[j]); }
        R = std::sqrt(R);
        for (int l = 0; l < m->nlight && !dirty[tl]; l++) {
          if (!m->light_active[l] || !m->light_castshadow[l]) continue;
          mjtNum E[3];
          if (m->light_type[l] == mjLIGHT_DIRECTIONAL) for (int j = 0; j < 3; j++) E[j] = C[j] - 20.0 * d->light_xdir[3 * l + j];
          else mju_copy3(E, d->light_xpos + 3 * l);
          mjtNum seg[3] = {E[0] - C[0], E[1] - C[1], E[2] - C[2]};
          double sl2 = mju_dot3(seg, seg);
          for (int g : movedG) {
            for (int which = 0; which < 2 && !dirty[tl]; which++) {
              const mjtNum* c = which ? d->geom_xpos + 3 * g : rx_pos_.data() + 3 * g;
              mjtNum w[3] = {c[0] - C[0], c[1] - C[1], c[2] - C[2]};
              double tt = sl2 > 0 ? std::min(1.0, std::max(0.0, mju_dot3(w, seg) / sl2)) : 0.0;
              mjtNum q[3] = {w[0] - tt * seg[0], w[1] - tt * seg[1], w[2] - tt * seg[2]};
              double rr = m->geom_rbound[g] + R;
              if (mju_dot3(q, q) <= rr * rr) dirty[tl] = 1;
            }
            if (dirty[tl]) break;
          }
        }
      }
  }
  if (probe) for (int k = 0; k < 3 * PGX * PGY; k++) probe[k] = NAN;
  const int sx = std::max(1, W_ / PGX), sy = std::max(1, H_ / PGY);
  const int ns = P_.spp;
  const double bgLv = P_.background_lux / kPi;
  int ndirty = 0;
  for (int tl = 0; tl < NT; tl++) ndirty += dirty[tl];

  parallelDynamic(NT, nthread_, [&](int tile) {
    const int tx = tile % TW, ty = tile / TW;
    const int u0 = tx * kTile, v0 = ty * kTile;
    const int u1 = std::min(W_, u0 + kTile), v1 = std::min(H_, v0 + kTile);
    if (!dirty[tile]) {                           // reuse: nothing moved in this tile
      for (int v = v0; v < v1; v++)
        for (int u = u0; u < u1; u++) { lblFx_[v * W_ + u] = 0; lblFy_[v * W_ + u] = 0; }
      if (probe) {
        for (int v = v0; v < v1; v++)
          for (int u = u0; u < u1; u++)
            if ((u % sx == sx / 2) && (v % sy == sy / 2)) {
              int i = v * W_ + u, g = lblG_[i];
              if (g >= 0 && m->body_weldid[m->geom_bodyid[g]] == 0) {
                mjtNum lc[3] = {ctr_[2 * i], -ctr_[2 * i + 1], -1.0}, dir[3];
                mju_mulMatVec3(dir, cmat, lc); mju_normalize3(dir);
                int pu = u / sx, pv = v / sy;
                if (pu < PGX && pv < PGY)
                  for (int j = 0; j < 3; j++) probe[3 * (pv * PGX + pu) + j] = cpos[j] + lblD_[i] * dir[j];
              }
            }
      }
      return;
    }
    const std::vector<int>& cand = tiles[tile];
    // pass 1: primary hits -> tile hit bounds (for shadow culling)
    mjtNum bmin[3] = {1e30, 1e30, 1e30}, bmax[3] = {-1e30, -1e30, -1e30};
    struct Hit { int g; double dist; mjtNum n[3], P[3], dir[3]; float uv[2]; bool hasuv; };
    auto trace = [&](double xn, double yn, Hit* h) {
      mjtNum lc[3] = {xn, -yn, -1.0};
      mju_mulMatVec3(h->dir, cmat, lc);
      mju_normalize3(h->dir);
      h->g = -1; h->dist = -1; h->hasuv = false;
      mjtNum nrm[3]; float uv[2] = {0, 0}; bool hu = false;
      auto test = [&](int g) {
        double dist;
        int type = m->geom_type[g];
        double tmax = h->dist > 0 ? h->dist : 1e30;
        if (type == mjGEOM_MESH) dist = TraceMesh(m, d->geom_xpos + 3 * g, d->geom_xmat + 9 * g, g, cpos, h->dir, tmax, nrm, uv, &hu);
        else { dist = mju_rayGeom(d->geom_xpos + 3 * g, d->geom_xmat + 9 * g, m->geom_size + 3 * g, cpos, h->dir, type, nrm); hu = false; }
        if (dist > 1e-4 && (h->dist < 0 || dist < h->dist)) {
          h->dist = dist; h->g = g; mju_copy3(h->n, nrm); h->hasuv = hu; h->uv[0] = uv[0]; h->uv[1] = uv[1];
        }
      };
      for (int g : global) test(g);
      for (int g : cand) test(g);
      if (h->g >= 0) for (int j = 0; j < 3; j++) h->P[j] = cpos[j] + h->dist * h->dir[j];
    };
    const int tw = u1 - u0, npx = tw * (v1 - v0);
    std::vector<Hit> hits((size_t)npx * ns * ns);
    for (int k = 0; k < npx; k++) {
      int u = u0 + k % tw, v = v0 + k / tw;
      for (int s = 0; s < ns * ns; s++) {
        double a = (s % ns + 0.5) / ns, b = (s / ns + 0.5) / ns;  // stratified sub-pixel
        double xn, yn;
        if (ns == 1) { xn = ctr_[2 * (v * W_ + u)]; yn = ctr_[2 * (v * W_ + u) + 1]; }
        else {
          const float* c00 = &corner_[2 * (v * (W_ + 1) + u)];
          const float* c10 = c00 + 2;
          const float* c01 = &corner_[2 * ((v + 1) * (W_ + 1) + u)];
          const float* c11 = c01 + 2;
          xn = (1 - a) * (1 - b) * c00[0] + a * (1 - b) * c10[0] + (1 - a) * b * c01[0] + a * b * c11[0];
          yn = (1 - a) * (1 - b) * c00[1] + a * (1 - b) * c10[1] + (1 - a) * b * c01[1] + a * b * c11[1];
        }
        Hit& h = hits[(size_t)k * ns * ns + s];
        trace(xn, yn, &h);
        if (h.g >= 0) for (int j = 0; j < 3; j++) { bmin[j] = std::min(bmin[j], h.P[j]); bmax[j] = std::max(bmax[j], h.P[j]); }
      }
    }
    for (int j = 0; j < 3; j++) { tileHit_[6 * tile + j] = (float)bmin[j]; tileHit_[6 * tile + 3 + j] = (float)bmax[j]; }
    // pass 2: shadow occluder candidates per light
    std::vector<std::vector<int>> occ(m->nlight);
    if (P_.shadows && bmax[0] >= bmin[0]) {
      mjtNum C[3], R = 0;
      for (int j = 0; j < 3; j++) { C[j] = 0.5 * (bmin[j] + bmax[j]); R += 0.25 * (bmax[j] - bmin[j]) * (bmax[j] - bmin[j]); }
      R = std::sqrt(R);
      for (int l = 0; l < m->nlight; l++) {
        if (!m->light_active[l] || !m->light_castshadow[l]) continue;
        mjtNum E[3];
        if (m->light_type[l] == mjLIGHT_DIRECTIONAL) for (int j = 0; j < 3; j++) E[j] = C[j] - 20.0 * d->light_xdir[3 * l + j];
        else mju_copy3(E, d->light_xpos + 3 * l);
        mjtNum seg[3] = {E[0] - C[0], E[1] - C[1], E[2] - C[2]};
        double sl2 = mju_dot3(seg, seg);
        for (int g : vis_geoms_) {
          if (m->geom_type[g] == mjGEOM_PLANE) { occ[l].push_back(g); continue; }
          const mjtNum* c = d->geom_xpos + 3 * g;
          mjtNum w[3] = {c[0] - C[0], c[1] - C[1], c[2] - C[2]};
          double tt = sl2 > 0 ? std::min(1.0, std::max(0.0, mju_dot3(w, seg) / sl2)) : 0.0;
          mjtNum q[3] = {w[0] - tt * seg[0], w[1] - tt * seg[1], w[2] - tt * seg[2]};
          double rr = m->geom_rbound[g] + R;
          if (mju_dot3(q, q) <= rr * rr) occ[l].push_back(g);
        }
      }
    }
    // pass 3: shading, labels, probe
    for (int k = 0; k < npx; k++) {
      int u = u0 + k % tw, v = v0 + k / tw, i = v * W_ + u;
      double acc = 0;
      for (int s = 0; s < ns * ns; s++) {
        Hit& h = hits[(size_t)k * ns * ns + s];
        double Lv = bgLv;
        if (h.g >= 0) Shade(m, d, h.g, h.P, h.n, h.dir, h.uv, h.hasuv, flick, occ, &Lv);
        acc += Lv;
      }
      raw_[i] = (float)(K_rate_ * acc / (ns * ns) * cos4_[i]);
      Hit& h = hits[(size_t)k * ns * ns + (ns * ns) / 2];   // label from central sample
      lblG_[i] = h.g;
      lblD_[i] = (float)(h.g >= 0 ? h.dist : 0);
      lblFx_[i] = lblFy_[i] = 0;
      if (h.g >= 0 && pxpos && dtflow > 0) {
        // exact flow: where was this material point at the start of the interval?
        const int g = h.g;
        mjtNum dl[3] = {h.P[0] - d->geom_xpos[3*g], h.P[1] - d->geom_xpos[3*g+1], h.P[2] - d->geom_xpos[3*g+2]}, loc[3], w0[3];
        mju_mulMatTVec3(loc, d->geom_xmat + 9 * g, dl);
        mju_mulMatVec3(w0, pxmat + 9 * g, loc); mju_addTo3(w0, pxpos + 3 * g);
        double ua, va, ub, vb;
        if (ProjectPx(pcpos, pcmat, w0, &ua, &va) && ProjectPx(cpos, cmat, h.P, &ub, &vb)) {
          lblFx_[i] = (float)((ub - ua) / dtflow); lblFy_[i] = (float)((vb - va) / dtflow);
        }
      }
      if (probe && h.g >= 0 && (u % sx == sx / 2) && (v % sy == sy / 2)) {
        int pu = u / sx, pv = v / sy;
        if (pu < PGX && pv < PGY && m->body_weldid[m->geom_bodyid[h.g]] == 0)
          for (int j = 0; j < 3; j++) probe[3 * (pv * PGX + pu) + j] = h.P[j];
      }
    }
    tileValid_[tile] = 1;
  });
  for (int g : vis_geoms_) { mju_copy3(&rx_pos_[3 * g], d->geom_xpos + 3 * g); mju_copy(&rx_mat_[9 * g], d->geom_xmat + 9 * g, 9); }
  for (int l = 0; l < m->nlight; l++) { mju_copy3(&rl_pos_[3 * l], d->light_xpos + 3 * l); mju_copy3(&rl_dir_[3 * l], d->light_xdir + 3 * l); }
  mju_copy3(rc_pos_, cpos); mju_copy(rc_mat_, cmat, 9);
  if (flick != 1.0) std::fill(tileValid_.begin(), tileValid_.end(), 0);
  Blur();
  prof_render_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tr0).count();
  prof_tiles_ += NT; prof_tiles_dirty_ += ndirty;
}

void EventCamera::Blur() {
  double s = P_.psf_sigma_px;
  if (s <= 0.05) { rate_ = raw_; return; }
  int r = std::max(1, (int)std::ceil(3 * s));
  std::vector<float> k(2 * r + 1);
  double sum = 0;
  for (int i = -r; i <= r; i++) { k[i + r] = (float)std::exp(-0.5 * i * i / (s * s)); sum += k[i + r]; }
  for (auto& x : k) x = (float)(x / sum);
  parallelFor(H_, nthread_, [&](int vb, int ve, int) {
    for (int v = vb; v < ve; v++)
      for (int u = 0; u < W_; u++) {
        float acc = 0;
        for (int i = -r; i <= r; i++) acc += k[i + r] * raw_[v * W_ + std::min(W_ - 1, std::max(0, u + i))];
        tmp_[v * W_ + u] = acc;
      }
  });
  parallelFor(H_, nthread_, [&](int vb, int ve, int) {
    for (int v = vb; v < ve; v++)
      for (int u = 0; u < W_; u++) {
        float acc = 0;
        for (int i = -r; i <= r; i++) acc += k[i + r] * tmp_[std::min(H_ - 1, std::max(0, v + i)) * W_ + u];
        rate_[v * W_ + u] = acc;
      }
  });
}

int EventCamera::MotionSubsteps(const mjModel* m, const mjData* d, const mjtNum* st, double dt) const {
  const mjtNum* pg_pos = st + Lo_.pose;
  const mjtNum* pg_mat = pg_pos + 3 * m->ngeom;
  const mjtNum* pc_pos = pg_mat + 9 * m->ngeom;
  const mjtNum* pc_mat = pc_pos + 3;
  const mjtNum* probe = st + Lo_.probe;
  const mjtNum* cpos = d->cam_xpos + 3 * cam_;
  const mjtNum* cmat = d->cam_xmat + 9 * cam_;
  double maxd = 0;
  for (int g : vis_geoms_) {
    if (m->geom_type[g] == mjGEOM_PLANE || m->body_weldid[m->geom_bodyid[g]] == 0) continue;
    const mjtNum* ab = m->geom_aabb + 6 * g;
    for (int c = 0; c < 8; c++) {
      mjtNum loc[3] = {ab[0] + ((c & 1) ? ab[3] : -ab[3]), ab[1] + ((c & 2) ? ab[4] : -ab[4]),
                       ab[2] + ((c & 4) ? ab[5] : -ab[5])}, w0[3], w1[3];
      mju_mulMatVec3(w0, pg_mat + 9 * g, loc); mju_addTo3(w0, pg_pos + 3 * g);
      mju_mulMatVec3(w1, d->geom_xmat + 9 * g, loc); mju_addTo3(w1, d->geom_xpos + 3 * g);
      double u0, v0, u1, v1;
      if (!ProjectPx(pc_pos, pc_mat, w0, &u0, &v0) || !ProjectPx(cpos, cmat, w1, &u1, &v1)) continue;
      bool in0 = u0 > -50 && u0 < W_ + 50 && v0 > -50 && v0 < H_ + 50;
      bool in1 = u1 > -50 && u1 < W_ + 50 && v1 > -50 && v1 < H_ + 50;
      if (!in0 && !in1) continue;
      maxd = std::max(maxd, std::hypot(u1 - u0, v1 - v0));
    }
  }
  for (int k = 0; k < PGX * PGY; k++) {
    const mjtNum* p = probe + 3 * k;
    if (std::isnan(p[0])) continue;
    double u0, v0, u1, v1;
    if (!ProjectPx(pc_pos, pc_mat, p, &u0, &v0) || !ProjectPx(cpos, cmat, p, &u1, &v1)) continue;
    maxd = std::max(maxd, std::hypot(u1 - u0, v1 - v0));
  }
  int n = (int)std::ceil(maxd / P_.max_px_motion);
  if (P_.flicker_hz > 0 && P_.flicker_depth > 0) n = std::max(n, (int)std::ceil(dt * P_.flicker_hz * 16));
  return std::max(1, std::min(P_.max_substeps, n));
}

// ----------------------------------------------------------------- pixel circuit
void EventCamera::UpdatePixels(mjtNum* st, double ta, double tb, uint64_t sample) {
  const double dt = tb - ta;
  const double trefr = P_.eff_refr_us * 1e-6;
  const double fmax = P_.eff_cutoff;
  const double hpfa = P_.eff_hpf > 0 ? 1.0 - std::exp(-2 * kPi * P_.eff_hpf * dt) : 0.0;
  const double tau2 = P_.sf_cutoff_hz > 0 ? 1.0 / (2 * kPi * P_.sf_cutoff_hz) : 0.0;
  mjtNum* S = st + Lo_.pix;
  mjtNum *XIN = S + F_XIN * N_, *L1 = S + F_L1 * N_, *L2 = S + F_L2 * N_, *NZ = S + F_NOISE * N_,
         *REF = S + F_REF * N_, *RFR = S + F_REFR * N_;
  for (auto& v : tev_) v.clear();
  auto ramp = [](double y0, double x0, double x1, double tau, double dt) {
    // exact 1st-order low-pass response to a linear input ramp x0 -> x1 over dt
    if (tau <= 0) return x1;
    double s = (x1 - x0) / dt, a = std::exp(-dt / tau);
    return x1 - s * tau + (y0 - x0 + s * tau) * a;
  };
  // Shot noise is band-limited by the two photoreceptor poles (a = fc, b = source follower
  // or 3*fc), so the noisy photoreceptor output is a smooth Gaussian process.  Instead of
  // sampling its path (which makes event rates depend on the time step), noise-induced
  // threshold crossings are generated from Rice's up-crossing intensity
  //     h(t) = nu0 * exp(-c(t)^2 / (2 sigma^2)),   nu0 = sqrt(a b),
  //     sigma^2 = g * pi a b / (r (a + b)),        c(t) = distance of the noiseless signal to
  // the threshold, integrated in closed form over each linear segment (erf) -> the noise-event
  // statistics are exact and independent of the MuJoCo time step.
  const double SQ2 = std::sqrt(2.0), SQPI2 = std::sqrt(kPi / 2.0);
  auto cumHaz = [&](double c0, double sc, double T, double nu0, double sig) {
    // integral_0^T nu0 * exp(-(c0 - sc t)^2 / (2 sig^2)) dt
    if (T <= 0) return 0.0;
    if (std::fabs(sc) * T < 1e-6 * sig) return nu0 * T * std::exp(-c0 * c0 / (2 * sig * sig));
    double x0 = c0 / (sig * SQ2), x1 = (c0 - sc * T) / (sig * SQ2);
    double diff;
    if (x0 > 0 && x1 > 0) diff = std::erfc(x1) - std::erfc(x0);          // stable tails
    else if (x0 < 0 && x1 < 0) diff = std::erfc(-x0) - std::erfc(-x1);
    else diff = std::erf(x0) - std::erf(x1);
    return std::max(0.0, nu0 * sig * SQPI2 / sc * diff);
  };
  parallelFor(H_, nthread_, [&](int vb, int ve, int tid) {
    auto& out = tev_[tid];
    for (int i = vb * W_; i < ve * W_ && i < N_; i++) {
      double r = std::max(0.0, (double)rate_[i]) + P_.dark_current;
      double x1 = std::log(r), x0 = XIN[i];
      double fc = fmax * r / (r + r_half_);
      double tau = 1.0 / (2 * kPi * fc);
      double l1a = L1[i], l1b = ramp(l1a, x0, x1, tau, dt);
      double l2a = tau2 > 0 ? L2[i] : l1a, l2b = tau2 > 0 ? ramp(l2a, l1a, l1b, tau2, dt) : l1b;
      double fb = P_.sf_cutoff_hz > 0 ? P_.sf_cutoff_hz : 3.0 * fc;
      double sig = std::sqrt(P_.noise_gain * kPi * fc * fb / (r * (fc + fb)));
      double nu0 = std::sqrt(fc * fb);
      const bool noisy = P_.shot_noise && sig > 0;
      double V0 = l2a, V1 = l2b;
      double slopeV = (V1 - V0) / dt;
      double lk = leak_[i] * th_on_[i];
      double thon = th_on_[i], thoff = th_off_[i];
      double tc = ta, Vc = V0, refc = REF[i], ru = RFR[i];
      int guard = 0;
      while (tc < tb && guard++ < 256) {
        if (ru > tc) {                             // reset switch closed: reference tracks
          if (ru >= tb) { refc = V1; tc = tb; break; }
          Vc = V0 + slopeV * (ru - ta); tc = ru;
          refc = Vc + (noisy ? sig * gauss(keyOf(P_.seed, i, sample, 300 + guard)) : 0.0);  // reset noise
        }
        double D0 = Vc - refc, sD = slopeV + lk, rem = tb - tc;
        double Dend = D0 + sD * rem, te = tb; int pol = -1;
        // deterministic (signal-driven) crossing
        if (Dend >= thon) { te = (D0 >= thon || sD <= 0) ? tc : tc + (thon - D0) / sD; pol = 1; }
        else if (Dend <= -thoff) { te = (D0 <= -thoff || sD >= 0) ? tc : tc + (-thoff - D0) / sD; pol = 0; }
        te = std::min(tb, std::max(tc, te));
        // noise-driven crossing before te (Rice hazard, closed form)
        bool noiseEv = false;
        if (noisy) {
          double T = te - tc;
          double con = thon - D0, coff = thoff + D0;      // distances at tc (>0)
          double Lon = cumHaz(con, sD, T, nu0, sig), Loff = cumHaz(coff, -sD, T, nu0, sig);
          double E = -std::log(u01(keyOf(P_.seed, i, sample, 400 + guard)));
          if (E < Lon + Loff) {
            double lo = 0, hi = T;                   // invert the cumulative hazard
            for (int it = 0; it < 30; it++) {
              double mid = 0.5 * (lo + hi);
              if (cumHaz(con, sD, mid, nu0, sig) + cumHaz(coff, -sD, mid, nu0, sig) < E) lo = mid; else hi = mid;
            }
            double tn = 0.5 * (lo + hi);
            double c1 = con - sD * tn, c2 = coff + sD * tn;
            double hon = std::exp(-c1 * c1 / (2 * sig * sig)), hoff = std::exp(-c2 * c2 / (2 * sig * sig));
            pol = u01(keyOf(P_.seed, i, sample, 500 + guard)) * (hon + hoff) < hon ? 1 : 0;
            te = tc + tn; noiseEv = true;
          }
        }
        if (pol < 0) { refc -= lk * rem; tc = tb; break; }
        double tout = te + 1e-6 * (P_.latency_us + P_.latency_jitter_us * gauss(keyOf(P_.seed, i, sample, 100 + guard)));
        tout = std::max(te, tout);
        Event e;
        e.x = (uint16_t)(i % W_); e.y = (uint16_t)(i / W_); e.p = (int16_t)pol;
        e.t = (int64_t)std::llround(tout * 1e6); e.te = te;
        e.geom = lblG_[i]; e.depth = lblD_[i]; e.fx = lblFx_[i]; e.fy = lblFy_[i];
        out.push_back(e);
        // reset: memorise the (noisy) level at which the comparator fired
        double refAt = refc - lk * (te - tc);
        Vc = V0 + slopeV * (te - ta);
        refc = noiseEv ? refAt + (pol ? thon : -thoff) : Vc;
        tc = te; ru = te + trefr;
        if (trefr <= 0 && sD == 0 && !noiseEv) break;
      }
      if (hpfa > 0) refc += (V1 - refc) * hpfa;
      XIN[i] = x1; L1[i] = l1b; L2[i] = l2b; NZ[i] = 0; REF[i] = refc; RFR[i] = ru;
    }
  });
}

// ----------------------------------------------------------------- readout & ESP
void EventCamera::Readout(mjtNum* st, std::vector<Event>& evs, std::vector<Event>& pend, double t1,
                          long long* dropped) {
  mjtNum* S = st + Lo_.pix;
  mjtNum *LT = S + F_LAST_T * N_, *LP = S + F_LAST_P * N_, *AT = S + F_AFK_T * N_,
         *AP = S + F_AFK_PER * N_, *AU = S + F_AFK_UNTIL * N_;
  // 1. ROI (pixels outside are disabled)
  if (P_.roi[2] > P_.roi[0] && P_.roi[3] > P_.roi[1]) {
    int b = P_.binning;
    evs.erase(std::remove_if(evs.begin(), evs.end(), [&](const Event& e) {
      int x = e.x * b, y = e.y * b;
      return x < P_.roi[0] || x >= P_.roi[2] || y < P_.roi[1] || y >= P_.roi[3]; }), evs.end());
  }
  // 2. time order
  std::stable_sort(evs.begin(), evs.end(), [](const Event& a, const Event& b) {
    return a.t < b.t || (a.t == b.t && a.te < b.te); });
  // 3. arbiter / bus throughput: single-server queue -> load-dependent latency
  if (P_.readout_mevps > 0) {
    double svc = 1.0 / (P_.readout_mevps * 1e6);
    double busy = st[H_ARB_BUSY];
    for (auto& e : evs) {
      double t = e.t * 1e-6;
      double to = std::max(t, busy);
      busy = to + svc;
      e.t = (int64_t)std::llround(to * 1e6);
    }
    st[H_ARB_BUSY] = busy;
  }
  // merge with events held back in earlier steps (already through ROI + arbiter)
  if (!pend.empty()) {
    std::vector<Event> merged; merged.reserve(evs.size() + pend.size());
    std::merge(pend.begin(), pend.end(), evs.begin(), evs.end(), std::back_inserter(merged),
               [](const Event& a, const Event& b) { return a.t < b.t || (a.t == b.t && a.te < b.te); });
    evs.swap(merged);
  }
  // 4. hold back events whose output time is in the future (keeps the stream monotonic)
  const int64_t tnow = (int64_t)std::llround(t1 * 1e6);
  auto split = std::upper_bound(evs.begin(), evs.end(), tnow, [](int64_t t, const Event& e) { return t < e.t; });
  std::vector<Event> future(split, evs.end());
  evs.erase(split, evs.end());
  // 5. ESP: anti-flicker (AFK) and STC / trail filters on the time-ordered stream
  const double afk_pmin = P_.afk_max_hz > 0 ? 1.0 / P_.afk_max_hz : 0, afk_pmax = P_.afk_min_hz > 0 ? 1.0 / P_.afk_min_hz : 0;
  if (P_.afk || P_.stc_us > 0 || P_.trail_us > 0) {
    std::vector<Event> kept; kept.reserve(evs.size());
    for (auto& e : evs) {
      int i = e.y * W_ + e.x;
      double t = e.t * 1e-6, lt = LT[i];
      int lp = (int)LP[i];
      bool keep = true;
      if (P_.afk) {
        if (e.p == 1 && lp != 1) {                 // start of an ON burst
          double per = t - AT[i];
          if (AT[i] > 0 && per >= afk_pmin && per <= afk_pmax) {
            if (AP[i] > 0 && std::fabs(per - AP[i]) < 0.25 * AP[i]) AU[i] = t + 2.5 * per;
            AP[i] = per;
          } else AP[i] = 0;
          AT[i] = t;
        }
        if (t < AU[i]) keep = false;
      }
      if (keep && P_.stc_us > 0) keep = (lp == e.p) && (t - lt) <= P_.stc_us * 1e-6;
      if (keep && P_.trail_us > 0) keep = !((lp == e.p) && (t - lt) <= P_.trail_us * 1e-6);
      LT[i] = t; LP[i] = e.p;
      if (keep) kept.push_back(e); else (*dropped)++;
    }
    evs.swap(kept);
  }
  // 6. Event Rate Controller: windowed, keep-ratio from the previous window's measured rate
  if (P_.erc_mevps > 0) {
    const double win = P_.erc_window_us * 1e-6;
    const double allowed = P_.erc_mevps * 1e6 * win;
    std::vector<Event> kept; kept.reserve(evs.size());
    for (auto& e : evs) {
      double t = e.t * 1e-6;
      while (t >= st[H_ERC_WSTART] + win) {
        double cnt = st[H_ERC_COUNT];
        st[H_ERC_KEEP] = cnt > allowed ? std::min(1.0, 1.1 * allowed / cnt) : 1.0;  // cap trims the rest
        st[H_ERC_COUNT] = 0; st[H_ERC_OUT] = 0;
        st[H_ERC_WSTART] = (t - st[H_ERC_WSTART] > 10 * win) ? std::floor(t / win) * win : st[H_ERC_WSTART] + win;
      }
      st[H_ERC_COUNT] += 1;                          // incoming events in this window
      st[H_ERC_ACC] += st[H_ERC_KEEP];               // spatially-uniform decimation
      bool keep = false;
      if (st[H_ERC_ACC] >= 1.0) { st[H_ERC_ACC] -= 1.0; keep = true; }
      if (keep && st[H_ERC_OUT] >= allowed) keep = false;   // hard cap per window
      if (keep) { st[H_ERC_OUT] += 1; kept.push_back(e); } else (*dropped)++;
    }
    evs.swap(kept);
  }
  // 7. finite output buffer
  if ((int)evs.size() > P_.max_events) { *dropped += (long long)evs.size() - P_.max_events; evs.resize(P_.max_events); }
  // pending (future) events are part of the saved state
  int np = std::min((int)future.size(), P_.pendCap());
  *dropped += (long long)future.size() - np;
  mjtNum* pb = st + Lo_.pend;
  for (int j = 0; j < np; j++) {
    const Event& e = future[j];
    mjtNum* r = pb + kEvStride * j;
    r[0] = e.x; r[1] = e.y; r[2] = e.p; r[3] = (mjtNum)e.t; r[4] = e.geom; r[5] = e.depth; r[6] = e.fx; r[7] = e.fy;
  }
  st[H_NPEND] = np;
}

// ----------------------------------------------------------------- main compute
static void slerpMat(mjtNum out[9], const mjtNum a[9], const mjtNum b[9], double t) {
  mjtNum qa[4], qb[4], q[4];
  mju_mat2Quat(qa, a); mju_mat2Quat(qb, b);
  double dot = qa[0]*qb[0] + qa[1]*qb[1] + qa[2]*qb[2] + qa[3]*qb[3];
  if (dot < 0) { for (int i = 0; i < 4; i++) qb[i] = -qb[i]; dot = -dot; }
  if (dot > 0.9995) { for (int i = 0; i < 4; i++) q[i] = qa[i] + t * (qb[i] - qa[i]); }
  else {
    double th = std::acos(dot), s = std::sin(th);
    double wa = std::sin((1 - t) * th) / s, wb = std::sin(t * th) / s;
    for (int i = 0; i < 4; i++) q[i] = wa * qa[i] + wb * qb[i];
  }
  mju_normalize4(q);
  mju_quat2Mat(out, q);
}

void EventCamera::Compute(const mjModel* m, mjData* d, int instance) {
  mjtNum* st = d->plugin_state + m->plugin_stateadr[instance];
  mjtNum* sd = d->sensordata + m->sensor_adr[sensor_];
  // publish layout (for the python reader)
  st[H_CAP] = P_.max_events; st[H_W] = W_; st[H_H] = H_; st[H_STRIDE] = kEvStride;
  st[H_VERSION] = kLayoutVersion; st[H_PENDCAP] = P_.pendCap();
  st[H_OFF_EVENTS] = Lo_.events; st[H_OFF_PEND] = Lo_.pend; st[H_OFF_PIX] = Lo_.pix;
  st[H_OFF_POSE] = Lo_.pose; st[H_OFF_PROBE] = Lo_.probe; st[H_OFF_IMG] = P_.export_intensity ? Lo_.img : -1;
  st[H_NPIXF] = F_NPIX; st[H_BACKEND_EXTERNAL] = P_.backend_external; st[H_NGEOM] = m->ngeom;
  st[H_FX] = fx_; st[H_FY] = fy_; st[H_CX] = cx_; st[H_CY] = cy_;
  if (P_.backend_external) { st[H_NEV] = 0; for (int k = 0; k < kNSensorData; k++) sd[k] = 0; return; }

  const mjtNum* cpos = d->cam_xpos + 3 * cam_;
  const mjtNum* cmat = d->cam_xmat + 9 * cam_;
  const double t1 = d->time;
  mjtNum* S = st + Lo_.pix;
  mjtNum* pg_pos = st + Lo_.pose;
  mjtNum* pg_mat = pg_pos + 3 * m->ngeom;
  mjtNum* pc_pos = pg_mat + 9 * m->ngeom;
  mjtNum* pc_mat = pc_pos + 3;
  mjtNum* probe = st + Lo_.probe;
  const bool restored = st[H_TOKEN] != token_;     // state came from elsewhere -> caches invalid
  if (restored) std::fill(tileValid_.begin(), tileValid_.end(), 0);

  auto savePoses = [&]() {
    mju_copy(pg_pos, d->geom_xpos, 3 * m->ngeom);
    mju_copy(pg_mat, d->geom_xmat, 9 * m->ngeom);
    mju_copy3(pc_pos, cpos); mju_copy(pc_mat, cmat, 9);
  };
  auto finish = [&]() {
    if (P_.export_intensity) {
      mjtNum* img = st + Lo_.img;
      const mjtNum* L2 = S + F_L2 * N_; const mjtNum* NZ = S + F_NOISE * N_;
      for (int i = 0; i < N_; i++) img[i] = L2[i] + NZ[i];
    }
    token_ = (double)(mix64(uid_ * 0x9E3779B97F4A7C15ULL + (++tokctr_)) >> 12);
    st[H_TOKEN] = token_;
  };

  // ---- (re)initialisation
  if (st[H_INIT] == 0 || t1 < st[H_TPREV] - 1e-12 || t1 - st[H_TPREV] > 0.05) {
    std::fill(tileValid_.begin(), tileValid_.end(), 0);
    Render(m, d, cpos, cmat, t1, nullptr, nullptr, nullptr, nullptr, 0, probe, true);
    for (int i = 0; i < N_; i++) {
      double r = std::max(0.0, (double)rate_[i]) + P_.dark_current;
      double x = std::log(r);
      S[F_XIN * N_ + i] = x; S[F_L1 * N_ + i] = x; S[F_L2 * N_ + i] = x; S[F_NOISE * N_ + i] = 0;
      // random phase of the memorised reference (desynchronises leak / noise events)
      // D0 = x - ref uniform inside (-th_off, +th_on), 5 % margins
      S[F_REF * N_ + i] = x + th_off_[i] - (th_on_[i] + th_off_[i]) * (0.05 + 0.9 * u01(keyOf(P_.seed, i, 0, 22)));
      S[F_REFR * N_ + i] = -1.0;
      S[F_LAST_T * N_ + i] = -1.0; S[F_LAST_P * N_ + i] = -1;
      S[F_AFK_T * N_ + i] = 0; S[F_AFK_PER * N_ + i] = 0; S[F_AFK_UNTIL * N_ + i] = 0;
    }
    savePoses();
    st[H_TPREV] = t1; st[H_INIT] = 1; st[H_NPEND] = 0; st[H_NEV] = 0; st[H_NSUB] = 0;
    st[H_SAMPLE] = 1; st[H_ARB_BUSY] = t1;
    st[H_ERC_WSTART] = std::floor(t1 / (P_.erc_window_us * 1e-6)) * (P_.erc_window_us * 1e-6); st[H_ERC_COUNT] = 0;
    st[H_ERC_KEEP] = 1; st[H_ERC_ACC] = 0; st[H_ERC_OUT] = 0; st[H_TLAST] = t1;
    for (int k = 0; k < kNSensorData; k++) sd[k] = 0;
    finish();
    return;
  }
  if (t1 - st[H_TPREV] < 1e-9) { st[H_NEV] = 0; for (int k = 0; k < 6; k++) sd[k] = 0; return; }

  const double t0 = st[H_TPREV], DT = t1 - t0;
  int nsub = MotionSubsteps(m, d, st, DT);
  uint64_t sample = (uint64_t)st[H_SAMPLE];
  std::vector<Event> evs, pend;
  {
    int np = (int)st[H_NPEND];
    const mjtNum* pb = st + Lo_.pend;
    for (int j = 0; j < np; j++) {
      const mjtNum* r = pb + kEvStride * j;
      Event e; e.x = (uint16_t)r[0]; e.y = (uint16_t)r[1]; e.p = (int16_t)r[2]; e.t = (int64_t)r[3];
      e.te = r[3] * 1e-6; e.geom = (int)r[4]; e.depth = (float)r[5]; e.fx = (float)r[6]; e.fy = (float)r[7];
      pend.push_back(e);
    }
  }
  std::vector<mjtNum> sv_pos, sv_mat;
  if (nsub > 1) { sv_pos.assign(d->geom_xpos, d->geom_xpos + 3 * m->ngeom); sv_mat.assign(d->geom_xmat, d->geom_xmat + 9 * m->ngeom); }
  std::vector<mjtNum> a_pos(pg_pos, pg_pos + 3 * m->ngeom), a_mat(pg_mat, pg_mat + 9 * m->ngeom);
  mjtNum a_cp[3], a_cm[9];
  mju_copy3(a_cp, pc_pos); mju_copy(a_cm, pc_mat, 9);
  double ta = t0;
  for (int k = 1; k <= nsub; k++) {
    double f = (double)k / nsub, tb = t0 + f * DT;
    mjtNum ip[3], im[9];
    if (k < nsub) {
      for (int g : vis_geoms_) {
        for (int j = 0; j < 3; j++) d->geom_xpos[3*g+j] = (1 - f) * pg_pos[3*g+j] + f * sv_pos[3*g+j];
        slerpMat(d->geom_xmat + 9 * g, pg_mat + 9 * g, sv_mat.data() + 9 * g, f);
      }
      for (int j = 0; j < 3; j++) ip[j] = (1 - f) * pc_pos[j] + f * cpos[j];
      slerpMat(im, pc_mat, cmat, f);
      Render(m, d, ip, im, tb, a_pos.data(), a_mat.data(), a_cp, a_cm, tb - ta, nullptr, restored && k == 1);
    } else {
      if (nsub > 1) { mju_copy(d->geom_xpos, sv_pos.data(), 3 * m->ngeom); mju_copy(d->geom_xmat, sv_mat.data(), 9 * m->ngeom); }
      mju_copy3(ip, cpos); mju_copy(im, cmat, 9);
      Render(m, d, cpos, cmat, tb, a_pos.data(), a_mat.data(), a_cp, a_cm, tb - ta, probe, restored && k == 1);
    }
    auto tp0 = std::chrono::steady_clock::now();
    UpdatePixels(st, ta, tb, sample++);
    prof_update_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
    for (auto& v : tev_) evs.insert(evs.end(), v.begin(), v.end());
    for (int g : vis_geoms_) { mju_copy3(&a_pos[3 * g], d->geom_xpos + 3 * g); mju_copy(&a_mat[9 * g], d->geom_xmat + 9 * g, 9); }
    mju_copy3(a_cp, ip); mju_copy(a_cm, im, 9);
    ta = tb;
  }
  st[H_SAMPLE] = (mjtNum)sample;

  auto tr0 = std::chrono::steady_clock::now();
  long long dropped = 0;
  Readout(st, evs, pend, t1, &dropped);
  prof_readout_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tr0).count();
  st[H_DROPTOT] += dropped;

  int nev = (int)evs.size(), non = 0;
  mjtNum* eb = st + Lo_.events;
  for (int j = 0; j < nev; j++) {
    const Event& e = evs[j];
    mjtNum* r = eb + kEvStride * j;
    r[0] = e.x; r[1] = e.y; r[2] = e.p; r[3] = (mjtNum)e.t; r[4] = e.geom; r[5] = e.depth; r[6] = e.fx; r[7] = e.fy;
    non += e.p;
  }
  st[H_NEV] = nev; st[H_NSUB] = nsub; st[H_TLAST] = t1;
  double meanL = 0;
  const mjtNum* L2 = S + F_L2 * N_;
  for (int i = 0; i < N_; i += 7) meanL += L2[i];
  sd[0] = nev; sd[1] = non; sd[2] = nev - non; sd[3] = (mjtNum)dropped; sd[4] = nsub;
  sd[5] = nev / DT * 1e-6; sd[6] = meanL / ((N_ + 6) / 7); sd[7] = st[H_DROPTOT];
  savePoses();
  st[H_TPREV] = t1;
  finish();
  if (profile_ && ++prof_n_ % 20 == 0) {
    std::printf("[evk4] per step: render %.1f ms  pixels %.1f ms  readout %.1f ms  dirty tiles %.0f%%  (nsub %d)\n",
                1e3 * prof_render_ / 20, 1e3 * prof_update_ / 20, 1e3 * prof_readout_ / 20,
                100.0 * prof_tiles_dirty_ / std::max(1L, prof_tiles_), nsub);
    std::fflush(stdout);
    prof_render_ = prof_update_ = prof_readout_ = 0; prof_tiles_ = prof_tiles_dirty_ = 0;
  }
}

// ----------------------------------------------------------------- visualisation
void EventCamera::Visualize(const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn, int instance) {
  if (!P_.show_frustum || P_.backend_external || fx_ <= 0) return;
  const mjtNum* p = d->cam_xpos + 3 * cam_;
  const mjtNum* R = d->cam_xmat + 9 * cam_;
  const double len = 0.12;
  int cu[4] = {0, W_, W_, 0}, cvv[4] = {0, 0, H_, H_};
  mjtNum corner[4][3];
  for (int k = 0; k < 4; k++) {
    const float* c = &corner_[2 * (cvv[k] * (W_ + 1) + cu[k])];
    mjtNum lc[3] = {c[0] * len, -c[1] * len, -len}, w[3];
    mju_mulMatVec3(w, R, lc);
    for (int j = 0; j < 3; j++) corner[k][j] = p[j] + w[j];
  }
  const float rgba[4] = {0.25f, 0.65f, 1.0f, 1.0f};
  auto line = [&](const mjtNum* a, const mjtNum* b) {
    if (scn->ngeom >= scn->maxgeom) return;
    mjvGeom* g = scn->geoms + scn->ngeom;
    mjv_initGeom(g, mjGEOM_LINE, nullptr, nullptr, nullptr, rgba);
    mjv_connector(g, mjGEOM_LINE, 2, a, b);
    g->category = mjCAT_DECOR;
    g->objtype = mjOBJ_UNKNOWN; g->objid = -1; g->segid = -1;   // decoration: no object / segment
    scn->ngeom++;
  };
  for (int k = 0; k < 4; k++) { line(p, corner[k]); line(corner[k], corner[(k + 1) % 4]); }
}

// ----------------------------------------------------------------- registration
static void RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "prophesee.sensor.evk4";
  plugin.capabilityflags |= mjPLUGIN_SENSOR;
  static const char* attributes[] = {
      "resolution", "binning", "pixel_pitch_um", "focal_length_mm", "f_number", "qe", "fx", "fy",
      "cx", "cy", "distortion", "spp", "psf_sigma_px", "specular_scale", "rgb_weights",
      "lux_per_unit", "background_lux", "dark_current", "shadows", "cutoff_hz", "bw_half_lux",
      "sf_cutoff_hz", "shot_noise", "noise_gain", "contrast_on", "contrast_off", "contrast_sigma",
      "refractory_us", "hpf_hz", "leak_rate_hz", "leak_sigma", "temperature_c", "leak_doubling_c",
      "hot_pixel_fraction", "hot_pixel_rate_hz", "bias_file", "bias_diff_on", "bias_diff_off",
      "bias_fo", "bias_hpf", "bias_refr", "bias_diff_scale", "bias_fo_scale",
      "bias_hpf_hz_per_unit", "bias_refr_scale", "latency_us", "latency_jitter_us",
      "readout_mevps", "erc_mevps", "erc_window_us", "afk", "afk_min_hz", "afk_max_hz", "stc_us",
      "trail_us", "roi", "max_events", "flicker_hz", "flicker_depth", "max_px_motion",
      "max_substeps", "export_intensity", "show_frustum", "nthread", "skip_static_tiles", "seed",
      "geomgroup", "backend"};
  plugin.nattribute = sizeof(attributes) / sizeof(attributes[0]);
  plugin.attributes = attributes;
  plugin.nstate = +[](const mjModel* m, int instance) {
    return Layout::Make(Params::Read(m, instance), m->ngeom).total;
  };
  plugin.nsensordata = +[](const mjModel* m, int instance, int sensor_id) { return kNSensorData; };
  plugin.needstage = mjSTAGE_POS;
  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto* ec = EventCamera::Create(m, d, instance);
    if (!ec) return -1;
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(ec);
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<EventCamera*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data, int instance) {
    std::memset(plugin_state, 0, sizeof(mjtNum) * m->plugin_statenum[instance]);
  };
  plugin.compute = +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
    reinterpret_cast<EventCamera*>(d->plugin_data[instance])->Compute(m, d, instance);
  };
  plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn, int instance) {
    reinterpret_cast<EventCamera*>(d->plugin_data[instance])->Visualize(m, d, opt, scn, instance);
  };
  mjp_registerPlugin(&plugin);
}

mjPLUGIN_LIB_INIT(evk4_event_camera) { RegisterPlugin(); }

}  // namespace evk4
