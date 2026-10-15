"""Rasterise the walls of a Webots .wbt world into a small occupancy grid JSON.

Used to build the maps for the browser demo. Only wall-like geometry is kept:
Wall nodes, the arena boundary and tall Box obstacles. Furniture PROTOs
(beds, sofas, tables) are not expanded, so those rooms come out emptier than
the Webots versions. Only rotations about the vertical axis are handled.

    python demo/tools/wbt_to_grid.py webots_worlds/SLAM_Project.wbt demo/maps/rooms.json --name "Rooms"
"""
import argparse
import json
import math
import re

import numpy as np


def tokenize(src):
    src = re.sub(r"#[^\n]*", "", src)
    return re.findall(r'"[^"]*"|\{|\}|\[|\]|[^\s{}\[\]]+', src)


NUM = re.compile(r"^[-+0-9.eE]+$")


class Parser:
    def __init__(self, toks):
        self.t = toks
        self.p = 0

    def is_node_start(self):
        t = self.t
        return re.match(r"^[A-Z]", t[self.p]) and self.p + 1 < len(t) and t[self.p + 1] == "{"

    def value(self):
        t = self.t[self.p]
        if t == "[":
            self.p += 1
            items = []
            while self.t[self.p] != "]":
                items.append(self.value())
            self.p += 1
            return items
        if t == "DEF":
            self.p += 2
            return self.value()
        if t == "USE":
            self.p += 2
            return None
        if self.is_node_start():
            return self.node()
        vals = []
        while self.p < len(self.t) and NUM.match(self.t[self.p]):
            vals.append(float(self.t[self.p]))
            self.p += 1
        if not vals:
            self.p += 1
            return t
        return vals

    def node(self):
        typ = self.t[self.p]
        self.p += 2
        fields = {}
        while self.t[self.p] != "}":
            name = self.t[self.p]
            self.p += 1
            if name == "hidden":
                self.p += 1
                self.value()
                continue
            fields[name] = self.value()
        self.p += 1
        return typ, fields

    def top_level(self):
        nodes = []
        while self.p < len(self.t):
            if self.t[self.p] == "DEF":
                self.p += 2
            elif self.is_node_start():
                nodes.append(self.node())
            else:
                self.p += 1
        return nodes


def vec(v, default):
    return v if isinstance(v, list) and v and isinstance(v[0], float) else default


def compose(T, tr, rot):
    yaw = rot[3] * (1 if rot[2] >= 0 else -1) if abs(rot[2]) > 0.9 else 0.0
    x, y, a = T
    c, s = math.cos(a), math.sin(a)
    return (x + c * tr[0] - s * tr[1], y + s * tr[0] + c * tr[1], a + yaw)


def collect(nodes):
    rects, arena, robot = [], None, None

    def walk(node, T):
        nonlocal arena, robot
        if not isinstance(node, tuple):
            return
        typ, f = node
        T2 = compose(T, vec(f.get("translation"), [0, 0, 0]), vec(f.get("rotation"), [0, 0, 1, 0]))
        if typ == "Wall":
            sz = vec(f.get("size"), [0.1, 1.0, 2.4])
            rects.append((T2[0], T2[1], T2[2], sz[0], sz[1]))
            return
        if typ == "Robot":
            robot = T2
            return
        if typ == "RectangleArena":
            arena = (T2, vec(f.get("floorSize"), [1.0, 1.0]))
        if typ == "Box":
            sz = vec(f.get("size"), None)
            if sz and sz[2] > 0.2:
                rects.append((T[0], T[1], T[2], sz[0], sz[1]))
        for key in ("children", "geometry"):
            v = f.get(key)
            child_T = T if typ == "Shape" else T2
            if isinstance(v, list):
                for c in v:
                    walk(c, child_T)
            elif isinstance(v, tuple):
                walk(v, child_T)

    for n in nodes:
        walk(n, (0.0, 0.0, 0.0))
    return rects, arena, robot


def rasterise(rects, arena, res):
    (ax, ay, ayaw), (w, h) = arena
    if abs(math.sin(ayaw)) > 0.5:
        w, h = h, w
    ox, oy = ax - w / 2, ay - h / 2
    nx, ny = int(round(w / res)), int(round(h / res))
    g = np.zeros((ny, nx), dtype=np.uint8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 1
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    cx = ox + (jj + 0.5) * res
    cy = oy + (ii + 0.5) * res
    for x, y, yaw, sx, sy in rects:
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = cx - x, cy - y
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        g[(np.abs(lx) <= sx / 2) & (np.abs(ly) <= sy / 2)] = 1
    return g, ox, oy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wbt")
    ap.add_argument("out")
    ap.add_argument("--name", required=True)
    ap.add_argument("--res", type=float, default=0.05)
    args = ap.parse_args()

    nodes = Parser(tokenize(open(args.wbt).read())).top_level()
    rects, arena, robot = collect(nodes)
    g, ox, oy = rasterise(rects, arena, args.res)
    out = {
        "name": args.name,
        "source": args.wbt.split("/")[-1],
        "note": "Walls rasterised from the Webots world; furniture omitted.",
        "resolution": args.res,
        "width": int(g.shape[1]),
        "height": int(g.shape[0]),
        "origin_x": round(ox, 4),
        "origin_y": round(oy, 4),
        "start": [round(robot[0], 3), round(robot[1], 3), round(robot[2], 3)] if robot else [0, 0, 0],
        # row i is y = origin_y + (i + 0.5) * resolution, matching OccupancyGrid
        "rows": ["".join("1" if v else "0" for v in row) for row in g],
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"{args.out}: {g.shape[1]}x{g.shape[0]} cells, {len(rects)} shapes, start {out['start']}")


if __name__ == "__main__":
    main()
