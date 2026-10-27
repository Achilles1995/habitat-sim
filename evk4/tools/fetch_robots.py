"""
Download robot models from MuJoCo Menagerie (github.com/google-deepmind/mujoco_menagerie) - only the
robots you ask for, no git and no GitHub API needed (the MJCF files are crawled for their assets).

    python tools/fetch_robots.py panda gen3 x2          # -> robots/menagerie/<folder>/
    python tools/fetch_robots.py ur5e --dest D:/models
    python tools/fetch_robots.py unitree_go2            # any Menagerie folder name works too

Each robot keeps its own LICENSE file (Apache-2.0 / BSD ... - see the folder).
"""
import argparse
import os
import posixpath
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main"
ALIASES = {"panda": "franka_emika_panda", "gen3": "kinova_gen3", "x2": "skydio_x2",
           "ur5e": "universal_robots_ur5e", "ur10e": "universal_robots_ur10e", "g1": "unitree_g1",
           "go2": "unitree_go2", "h1": "unitree_h1", "spot": "boston_dynamics_spot"}
TEX_ATTRS = ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "evk4-mujoco-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _save(folder, rel, dest, optional=False):
    dst = os.path.join(dest, folder, *rel.split("/"))
    if os.path.exists(dst):
        return open(dst, "rb").read()
    try:
        data = _get(f"{RAW}/{folder}/{rel}")
    except urllib.error.HTTPError as e:
        if optional:
            return None
        raise RuntimeError(f"download failed: {folder}/{rel} ({e.code})") from e
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)
    return data


def fetch_robot(name, dest, entry="scene.xml"):
    folder = ALIASES.get(name, name)
    dirs = {"meshdir": "", "texturedir": ""}
    todo, seen, assets = [entry], set(), []
    while todo:                                        # MJCF files (entry + <include>s)
        rel = posixpath.normpath(todo.pop())
        if rel in seen:
            continue
        seen.add(rel)
        root = ET.fromstring(_save(folder, rel, dest))
        base = posixpath.dirname(rel)
        for c in root.iter("compiler"):
            if c.get("assetdir"):
                dirs["meshdir"] = dirs["texturedir"] = c.get("assetdir")
            for k in ("meshdir", "texturedir"):
                if c.get(k):
                    dirs[k] = c.get(k)
        for inc in root.iter("include"):
            todo.append(posixpath.join(base, inc.get("file")))
        for tag, key in (("mesh", "meshdir"), ("hfield", "meshdir"), ("skin", "meshdir"), ("texture", "texturedir")):
            for el in root.iter(tag):
                for a in (TEX_ATTRS if tag == "texture" else ("file",)):
                    if el.get(a):
                        assets.append((key, el.get(a)))
    files = sorted({posixpath.normpath(posixpath.join(dirs[k], f)) for k, f in assets})
    print(f"{folder}: {len(seen)} MJCF + {len(files)} asset files")
    for i, f in enumerate(files):
        _save(folder, f, dest)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(files)}")
    for extra in ("LICENSE", "README.md"):
        _save(folder, extra, dest, optional=True)
    return os.path.join(dest, folder, entry)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("robots", nargs="+", help="panda gen3 x2 ur5e ... or Menagerie folder names")
    ap.add_argument("--dest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robots",
                                                   "menagerie"))
    a = ap.parse_args()
    ok = True
    for r in a.robots:
        try:
            print("  ->", os.path.abspath(fetch_robot(r, os.path.abspath(a.dest))))
        except Exception as ex:                        # keep going with the other robots
            ok = False
            print(f"{r}: {ex}", file=sys.stderr)
    sys.exit(0 if ok else 1)
