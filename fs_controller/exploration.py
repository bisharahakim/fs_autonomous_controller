"""First (exploration) lap on an unknown track: cone perception -> KinoRRT -> PFPP tracking.

On the first lap the car does not know the track. It only sees the cones in front of it.
This module turns those cones into a short drivable path every few tenths of a second and
tracks that path with the PFPP steering law:

    ConePerception   simulated cone detector (range + field of view, optional noise and
                     missed detections) and a cone map that accumulates every detection.
                     Localisation is perfect, so the map stands in for SLAM; a real SLAM
                     map can replace it later through the same interface (cones + colours).
    KinoRRT          kinodynamic RRT in the car's own state space (x, y, heading, path
                     curvature). Branches are arcs whose curvature is limited by the car's
                     tightest turn and whose curvature CHANGE is limited by the steering
                     rate, so every branch is drivable. A branch is rejected when the car's
                     body would come within a margin of a known cone, or when it leaves the
                     corridor: blue cones must be on the left, yellow on the right, and both
                     must be within reach. The best branch is the longest one, minus
                     penalties for little clearance and for curvature and its changes.
    LocalPath        the chosen branch, densely sampled, with a cautious speed profile
                     (speed cap, lateral-acceleration limit, can stop before the end of what
                     is known).
    ExplorationLateral  the steering + speed interface the benchmark runner uses. Steering
                     is PFPP on the local path: speed-scaled lookahead, pure pursuit
                     correction plus the path's own curvature as feed-forward. The runner
                     adds PFPP's delay prediction, rear-axle reference and steering map.

Everything is plain numpy and car-agnostic: the car enters through its wheelbase, body size
and tightest turn radius. Randomness comes from a seeded generator, so a lap is repeatable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .lateral import LateralCommand

BLUE, YELLOW = 0, 1          # FS convention: blue cones on the left, yellow on the right
CONE_RADIUS_M = 0.20
ROWS_PER_STEP = 4            # dense path samples per planner segment


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ----------------------------------------------------------------------------- perception
@dataclass(frozen=True)
class PerceptionConfig:
    range_m: float = 15.0
    fov_deg: float = 120.0
    sensor_forward_m: float = 1.5     # sensor position ahead of the car's reference point
    noise_m: float = 0.0              # standard deviation of each detection's position (m)
    miss_prob: float = 0.0            # chance that a visible cone is not detected in a frame
    merge_radius_m: float = 0.7       # detections this close to a mapped cone update it
    seed: int = 1


class ConePerception:
    """Simulated cone detector plus the cone map it builds (perfect localisation)."""

    def __init__(self, cones_xy: np.ndarray, colours: np.ndarray, config: PerceptionConfig) -> None:
        self.true_xy = np.asarray(cones_xy, dtype=float)
        self.true_col = np.asarray(colours, dtype=int)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.map_xy = np.zeros((0, 2))
        self.map_col = np.zeros(0, dtype=int)
        self._hits = np.zeros(0)                   # detections merged into each map cone
        self.visible_ids = np.zeros(0, dtype=int)  # true cone ids seen in the last frame
        self.ever_seen = np.zeros(len(self.true_xy), dtype=bool)

    def observe(self, x: float, y: float, yaw: float) -> None:
        c = self.config
        sx, sy = x + c.sensor_forward_m * math.cos(yaw), y + c.sensor_forward_m * math.sin(yaw)
        d = self.true_xy - (sx, sy)
        dist = np.hypot(d[:, 0], d[:, 1])
        bearing = _wrap(np.arctan2(d[:, 1], d[:, 0]) - yaw)
        ids = np.where((dist <= c.range_m) & (np.abs(bearing) <= math.radians(c.fov_deg) / 2.0))[0]
        if c.miss_prob > 0.0 and len(ids):
            ids = ids[self.rng.random(len(ids)) >= c.miss_prob]
        self.visible_ids = ids
        self.ever_seen[ids] = True
        det = self.true_xy[ids].copy()
        if c.noise_m > 0.0 and len(ids):
            det += self.rng.normal(0.0, c.noise_m, det.shape)
        self._merge(det, self.true_col[ids])

    def _merge(self, det: np.ndarray, col: np.ndarray) -> None:
        if not len(det):
            return
        new_xy, new_col = [], []
        for p, k in zip(det, col):
            if len(self.map_xy):
                dist = np.hypot(self.map_xy[:, 0] - p[0], self.map_xy[:, 1] - p[1])
                dist[self.map_col != k] = np.inf
                j = int(np.argmin(dist))
                if dist[j] < self.config.merge_radius_m:
                    n = self._hits[j]
                    self.map_xy[j] = (self.map_xy[j] * n + p) / (n + 1.0)   # running mean
                    self._hits[j] = min(n + 1.0, 50.0)
                    continue
            new_xy.append(p)
            new_col.append(k)
        if new_xy:
            self.map_xy = np.vstack((self.map_xy, new_xy))
            self.map_col = np.concatenate((self.map_col, new_col))
            self._hits = np.concatenate((self._hits, np.ones(len(new_xy))))

    def map_near(self, x: float, y: float, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.map_xy):
            return self.map_xy, self.map_col
        keep = np.hypot(self.map_xy[:, 0] - x, self.map_xy[:, 1] - y) <= radius_m
        return self.map_xy[keep], self.map_col[keep]


# ----------------------------------------------------------------------------- planner
@dataclass(frozen=True)
class KinoRRTConfig:
    iterations: int = 400
    step_m: float = 1.0               # length of one branch segment
    check_every_m: float = 0.25       # collision-check spacing along a segment
    horizon_m: float = 15.0           # branches stay within this distance of the root
    min_radius_m: float = 2.2         # tightest turn the planner may use (rear axle)
    dkappa_per_m: float = 0.25        # largest curvature change per metre of path (the speed plan
                                      # then keeps the steering rate within the car's limit)
    margin_m: float = 0.30            # smallest allowed gap between the car body and a cone
    margin_tight_m: float = 0.12      # ... used only when nothing longer than good_m fits otherwise
    good_m: float = 6.0
    clearance_ref_m: float = 1.00     # gaps smaller than this are penalised
    support_m: float = 5.5            # a node needs a blue AND a yellow cone this close
    link_m: float = 6.0               # same-colour cones closer than this form a track limit
    # Penalties are small next to the reward of 1 per metre of branch, so the longest drivable
    # branch wins and the penalties only choose between branches of similar length.
    w_clear: float = 0.4              # per metre, per metre of clearance below clearance_ref_m
    w_dkappa: float = 0.15            # per segment that changes curvature at the full rate
    w_kappa: float = 0.05             # per metre at the tightest curvature
    tries: int = 6                    # nearest nodes tried per sample before giving up
    reuse_m: float = 5.0              # how much of the previous plan (beyond the root) is re-used
    smooth_m: float = 1.0             # curvature smoothing of the chosen branch (Gaussian sigma)
    bias_mid: float = 0.75            # share of samples near midpoints between blue and yellow
    seed: int = 1


@dataclass(frozen=True)
class CarGeometry:
    wheelbase_m: float
    half_length_m: float
    half_width_m: float
    body_centre_ahead_m: float        # body centre ahead of the rear axle (the planned point)


class KinoRRT:
    """Kinodynamic RRT over (x, y, heading, curvature) for the rear axle."""

    def __init__(self, config: KinoRRTConfig, car: CarGeometry) -> None:
        self.c = config
        self.car = car
        self.rng = np.random.default_rng(config.seed)
        self.kappa_max = 1.0 / config.min_radius_m
        n = config.iterations + 64                    # room for the re-used previous plan
        self.nx, self.ny, self.nyaw, self.nk = (np.zeros(n) for _ in range(4))
        self.ns, self.npen = np.zeros(n), np.zeros(n)
        self.nparent = np.full(n, -1, dtype=int)
        self.count = 0
        self.last_nodes = 0
        self.smoothed = 0
        self.tight_plans = 0
        self.margin = config.margin_m
        self.last_raw = None

    # --- geometry ---------------------------------------------------------------------
    @staticmethod
    def arc(x: float, y: float, yaw: float, kappa: float, lengths: np.ndarray):
        """Poses along a constant-curvature arc after each length in `lengths`."""
        if abs(kappa) < 1e-6:
            return x + lengths * math.cos(yaw), y + lengths * math.sin(yaw), yaw + 0.0 * lengths
        th = yaw + kappa * lengths
        return (x + (np.sin(th) - math.sin(yaw)) / kappa,
                y - (np.cos(th) - math.cos(yaw)) / kappa, th)

    def clearance(self, px, py, pyaw, cones: np.ndarray) -> np.ndarray:
        """Gap (m) from the car body to the nearest cone surface, per pose."""
        if not len(cones):
            return np.full(len(px), np.inf)
        car = self.car
        bx = px + car.body_centre_ahead_m * np.cos(pyaw)
        by = py + car.body_centre_ahead_m * np.sin(pyaw)
        dx = cones[None, :, 0] - bx[:, None]
        dy = cones[None, :, 1] - by[:, None]
        cs, sn = np.cos(pyaw)[:, None], np.sin(pyaw)[:, None]
        lx = np.abs(cs * dx + sn * dy) - car.half_length_m
        ly = np.abs(-sn * dx + cs * dy) - car.half_width_m
        gap = np.hypot(np.maximum(lx, 0.0), np.maximum(ly, 0.0)) - CONE_RADIUS_M
        return gap.min(axis=1)

    @staticmethod
    def boundaries(cones: np.ndarray, link_m: float) -> np.ndarray:
        """Track limits from one colour of cones: each cone joined to its two nearest neighbours
        of the same colour within link_m. Returns segments as rows (x1, y1, x2, y2)."""
        if len(cones) < 2:
            return np.zeros((0, 4))
        d = np.hypot(cones[:, None, 0] - cones[None, :, 0], cones[:, None, 1] - cones[None, :, 1])
        np.fill_diagonal(d, np.inf)
        pairs = set()
        for i, row in enumerate(d):
            for j in np.argsort(row)[:2]:
                if row[j] <= link_m:
                    pairs.add((min(i, int(j)), max(i, int(j))))
        if not pairs:
            return np.zeros((0, 4))
        a, b = np.array(sorted(pairs)).T
        return np.column_stack((cones[a], cones[b]))

    @staticmethod
    def crosses(px: np.ndarray, py: np.ndarray, segs: np.ndarray) -> bool:
        """Does the polyline (px, py) cross any of the segments?"""
        if not len(segs) or len(px) < 2:
            return False
        ax, ay, bx, by = px[:-1, None], py[:-1, None], px[1:, None], py[1:, None]
        cx, cy, dx, dy = segs[None, :, 0], segs[None, :, 1], segs[None, :, 2], segs[None, :, 3]

        def orient(x1, y1, x2, y2, x3, y3):
            return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

        o1 = orient(ax, ay, bx, by, cx, cy)
        o2 = orient(ax, ay, bx, by, dx, dy)
        o3 = orient(cx, cy, dx, dy, ax, ay)
        o4 = orient(cx, cy, dx, dy, bx, by)
        return bool(np.any((o1 * o2 < 0.0) & (o3 * o4 < 0.0)))

    def in_corridor(self, x: float, y: float, yaw: float, blue: np.ndarray, yellow: np.ndarray) -> bool:
        """Blue cone nearest to the body is on its left, yellow on its right, both in reach."""
        if not len(blue) or not len(yellow):
            return False
        car = self.car
        bx, by = x + car.body_centre_ahead_m * math.cos(yaw), y + car.body_centre_ahead_m * math.sin(yaw)
        cs, sn = math.cos(yaw), math.sin(yaw)
        for cones, sign in ((blue, 1.0), (yellow, -1.0)):
            d = np.hypot(cones[:, 0] - bx, cones[:, 1] - by)
            j = int(np.argmin(d))
            if d[j] > self.c.support_m:
                return False
            side = -sn * (cones[j, 0] - bx) + cs * (cones[j, 1] - by)
            if side * sign <= 0.0:
                return False
        return True

    # --- planning ---------------------------------------------------------------------
    def _add(self, x, y, yaw, k, s, pen, parent) -> int:
        i = self.count
        self.nx[i], self.ny[i], self.nyaw[i], self.nk[i] = x, y, yaw, k
        self.ns[i], self.npen[i], self.nparent[i] = s, pen, parent
        self.count += 1
        return i

    def _midpoints(self, blue: np.ndarray, yellow: np.ndarray) -> np.ndarray:
        if not len(blue) or not len(yellow):
            return np.zeros((0, 2))
        d = np.hypot(blue[:, None, 0] - yellow[None, :, 0], blue[:, None, 1] - yellow[None, :, 1])
        j = d.argmin(axis=1)
        ok = d[np.arange(len(blue)), j] < 8.0
        return 0.5 * (blue[ok] + yellow[j[ok]])

    def _sample(self, rx, ry, ryaw, mids):
        c = self.c
        if self.rng.random() < c.bias_mid and len(mids):
            p = mids[self.rng.integers(len(mids))]
            return p[0] + self.rng.normal(0.0, 0.7), p[1] + self.rng.normal(0.0, 0.7)
        r = self.rng.uniform(1.0, c.horizon_m)
        a = ryaw + self.rng.uniform(-1.3, 1.3)
        return rx + r * math.cos(a), ry + r * math.sin(a)

    def _edge(self, p: int, k_new: float, cones, blue, yellow, x0, y0):
        """Check the arc from node p with curvature k_new; return (end pose, penalty) or None."""
        c = self.c
        px, py, pyaw = self.arc(self.nx[p], self.ny[p], self.nyaw[p], k_new, self._checks)
        if self._crosses_limits(np.r_[self.nx[p], px], np.r_[self.ny[p], py], np.r_[self.nyaw[p], pyaw]):
            self.reject["limits"] += 1
            return None
        ex, ey, eyaw = float(px[-1]), float(py[-1]), float(pyaw[-1])
        if math.hypot(ex - x0, ey - y0) > c.horizon_m:
            self.reject["horizon"] += 1
            return None
        gap = float(self.clearance(px, py, pyaw, cones).min())
        if gap < self.margin:
            self.reject["cone"] += 1
            return None
        if not self.in_corridor(ex, ey, eyaw, blue, yellow):
            self.reject["corridor"] += 1
            return None
        dk = c.dkappa_per_m * c.step_m
        pen = (c.w_clear * max(0.0, c.clearance_ref_m - gap) * c.step_m
               + c.w_dkappa * abs(k_new - self.nk[p]) / max(dk, 1e-9)
               + c.w_kappa * (k_new / self.kappa_max) ** 2 * c.step_m)
        return (ex, ey, eyaw), pen

    def plan(self, x: float, y: float, yaw: float, kappa0: float, cones_xy: np.ndarray,
             cones_col: np.ndarray, chain: np.ndarray | None = None):
        """Plan with the normal cone margin; if nothing of good_m fits, plan again with the tight one."""
        self.margin = self.c.margin_m
        best = self._plan(x, y, yaw, kappa0, cones_xy, cones_col, chain)
        if (best is None or best[-1, 0] < self.c.good_m) and self.c.margin_tight_m < self.c.margin_m:
            normal, reject, raw = best, dict(self.reject), self.last_raw
            n = self.count
            saved = [a[:n].copy() for a in (self.nx, self.ny, self.nyaw, self.nk, self.ns, self.npen, self.nparent)]
            self.margin = self.c.margin_tight_m
            tight = self._plan(x, y, yaw, kappa0, cones_xy, cones_col, chain)
            if tight is not None and (normal is None or tight[-1, 0] > normal[-1, 0] + 0.5):
                self.tight_plans += 1
                return tight
            # keep the normal tree (for drawing and the record)
            for arr, keep in zip((self.nx, self.ny, self.nyaw, self.nk, self.ns, self.npen, self.nparent), saved):
                arr[:n] = keep
            self.count = self.last_nodes = n
            self.reject, self.last_raw, self.margin = reject, raw, self.c.margin_m
            best = normal
        return best

    def _plan(self, x: float, y: float, yaw: float, kappa0: float, cones_xy: np.ndarray,
              cones_col: np.ndarray, chain: np.ndarray | None = None):
        """Grow a tree from the root pose; return the best branch as dense samples.

        chain: the rest of the previous plan as node curvatures (one per step_m, starting at
        this root). It is re-checked against the current cone map and kept as long as it is
        still valid, so a new plan can only replace it by scoring better.

        Returns rows (s, x, y, yaw, kappa) every step_m / ROWS_PER_STEP metres starting at the
        root (s = 0), or None when no branch of at least one segment exists.
        """
        c = self.c
        reach = c.horizon_m + c.support_m + 3.0
        near = np.hypot(cones_xy[:, 0] - x, cones_xy[:, 1] - y) <= reach if len(cones_xy) else np.zeros(0, bool)
        cones, col = cones_xy[near], cones_col[near]
        blue, yellow = cones[col == BLUE], cones[col == YELLOW]
        mids = self._midpoints(blue, yellow)
        self._limits = np.vstack((self.boundaries(blue, c.link_m), self.boundaries(yellow, c.link_m)))
        if len(mids):
            far = np.hypot(mids[:, 0] - x, mids[:, 1] - y)
            mids = mids[(far > 1.0) & (far < c.horizon_m + 1.0)]
        self.count = 0
        self.reject = {"horizon": 0, "cone": 0, "limits": 0, "corridor": 0, "chain_kept": 0}
        self._add(x, y, yaw, float(np.clip(kappa0, -self.kappa_max, self.kappa_max)), 0.0, 0.0, -1)
        self._checks = np.arange(1, ROWS_PER_STEP + 1) * (c.step_m / ROWS_PER_STEP)
        dk = c.dkappa_per_m * c.step_m

        if chain is not None:                          # re-use the previous plan while it is valid
            parent = 0
            for k_node in chain[:int(round(c.reuse_m / c.step_m))]:
                if self.count >= len(self.nx):
                    break
                edge = self._edge(parent, float(k_node), cones, blue, yellow, x, y)
                if edge is None:
                    break
                (ex, ey, eyaw), pen = edge
                parent = self._add(ex, ey, eyaw, float(k_node), self.ns[parent] + c.step_m,
                                   self.npen[parent] + pen, parent)
                self.reject["chain_kept"] += 1

        for _ in range(c.iterations):
            if self.count >= len(self.nx):
                break
            sx, sy = self._sample(x, y, yaw, mids)
            n = self.count
            dx, dy = sx - self.nx[:n], sy - self.ny[:n]
            cs, sn = np.cos(self.nyaw[:n]), np.sin(self.nyaw[:n])
            lx, ly = cs * dx + sn * dy, -sn * dx + cs * dy
            metric = np.hypot(lx, ly) + 2.0 * np.abs(np.arctan2(ly, lx))
            metric[lx < 0.3] = np.inf
            order = np.argsort(metric)[:c.tries]
            for p in (int(q) for q in order):
                if not np.isfinite(metric[p]):
                    break
                k_want = 2.0 * ly[p] / max(lx[p] ** 2 + ly[p] ** 2, 1e-6)
                k_new = float(np.clip(k_want, self.nk[p] - dk, self.nk[p] + dk))
                k_new = float(np.clip(k_new, -self.kappa_max, self.kappa_max))
                ex, ey, eyaw = (float(v[-1]) for v in self.arc(self.nx[p], self.ny[p], self.nyaw[p], k_new,
                                                                self._checks[-1:]))
                close = (np.hypot(self.nx[:n] - ex, self.ny[:n] - ey) < 0.35) & \
                        (np.abs(_wrap(self.nyaw[:n] - eyaw)) < 0.15)
                if close.any():                        # near-duplicate: keeps the tree spread out
                    break
                edge = self._edge(p, k_new, cones, blue, yellow, x, y)
                if edge is None:
                    continue                           # blocked: try the next-nearest node
                (ex, ey, eyaw), pen = edge
                self._add(ex, ey, eyaw, k_new, self.ns[p] + c.step_m, self.npen[p] + pen, p)
                break

        self.last_nodes = self.count
        if self.count < 2:
            return None
        score = self.ns[:self.count] - self.npen[:self.count]
        best = int(np.argmax(score))
        if best == 0:
            return None
        nodes = []
        while best > 0:
            nodes.append(best)
            best = int(self.nparent[best])
        nodes.reverse()
        # Dense samples of the branch (the curvature stored at node i is that of the arc into it).
        rows = [(0.0, x, y, yaw, self.nk[nodes[0]])]
        for i in nodes:
            par = int(self.nparent[i])
            ax, ay, ayaw = self.arc(self.nx[par], self.ny[par], self.nyaw[par], self.nk[i], self._checks)
            rows.extend((self.ns[par] + f, a, b, h, self.nk[i]) for f, a, b, h in zip(self._checks, ax, ay, ayaw))
        rows = np.array(rows)
        self.last_raw = rows
        smooth = self.smooth(rows, float(self.nk[0]))
        if smooth is not None and self._valid(smooth, cones, blue, yellow):
            self.smoothed += 1
            return smooth
        return rows

    def smooth(self, rows: np.ndarray, kappa0: float) -> np.ndarray | None:
        """Low-pass the branch's curvature profile and integrate it again from the root.

        The tree's branches are chains of 1 m arcs, so their curvature steps and can zigzag.
        Smoothing the curvature (not the positions) keeps the start pose and heading exact and
        gives the tracker a path with continuous curvature.
        """
        sigma = self.c.smooth_m / (self.c.step_m / ROWS_PER_STEP)
        if sigma <= 0.0 or len(rows) < 3:
            return None
        k = rows[:, 4]
        half = int(3 * sigma) + 1
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.concatenate((np.full(half, kappa0), k, np.full(half, k[-1])))
        ks = np.clip(np.convolve(padded, kernel, mode="valid"), -self.kappa_max, self.kappa_max)
        out = rows.copy()
        out[:, 4] = ks
        ds = np.diff(rows[:, 0])
        for i in range(1, len(out)):
            ax, ay, ayaw = self.arc(out[i - 1, 1], out[i - 1, 2], out[i - 1, 3], ks[i], np.array([ds[i - 1]]))
            out[i, 1], out[i, 2], out[i, 3] = ax[0], ay[0], ayaw[0]
        return out

    def _crosses_limits(self, px, py, pyaw) -> bool:
        a = self.car.body_centre_ahead_m
        return self.crosses(px + a * np.cos(pyaw), py + a * np.sin(pyaw), self._limits)

    def _valid(self, rows: np.ndarray, cones, blue, yellow) -> bool:
        if self._crosses_limits(rows[:, 1], rows[:, 2], rows[:, 3]):
            return False
        if float(self.clearance(rows[:, 1], rows[:, 2], rows[:, 3], cones).min()) < self.margin:
            return False
        return all(self.in_corridor(r[1], r[2], r[3], blue, yellow) for r in rows[ROWS_PER_STEP::ROWS_PER_STEP])

    def tree_edges(self) -> list[np.ndarray]:
        """Every branch segment of the last tree as a short polyline (for drawing)."""
        out = []
        pts = np.linspace(0.0, self.c.step_m, 5)
        for i in range(1, self.count):
            par = int(self.nparent[i])
            ax, ay, _ = self.arc(self.nx[par], self.ny[par], self.nyaw[par], self.nk[i], pts)
            out.append(np.column_stack((ax, ay)))
        return out


# ----------------------------------------------------------------------------- path + speed
class LocalPath:
    """Open path from the planner with a cautious speed profile.

    Rows before root_row are the part of the previous plan the car is driving on; from
    root_row on, every ROWS_PER_STEP-th row is a planner node (where the next tree may start).
    """

    def __init__(self, samples: np.ndarray, v_max: float, ay_max: float, decel: float,
                 root_row: int = 0, kappa_rate: float = 0.8) -> None:
        self.s, self.x, self.y, self.yaw, self.k = (samples[:, i].copy() for i in range(5))
        self.s -= self.s[0]
        self.root_row = root_row
        self.length = float(self.s[-1])
        n = len(self.s)
        # Curvature the speed must respect: the tightest within +-1 m.
        half = 4
        kabs = np.abs(self.k)
        kwin = np.array([kabs[max(0, i - half):i + half + 1].max() for i in range(n)])
        v = np.minimum(v_max, np.sqrt(ay_max / np.maximum(kwin, 1e-6)))
        # Steering rate: curvature changing by dk/ds along the path needs dk/dt = v dk/ds.
        dkds = np.abs(np.gradient(self.k, self.s)) if n > 1 else np.zeros(n)
        dwin = np.array([dkds[max(0, i - half):i + half + 1].max() for i in range(n)])
        v = np.minimum(v, kappa_rate / np.maximum(dwin, 1e-6))
        # Must be able to stop within what is known: standstill 1 m before the end of the path.
        v[self.s > self.s[-1] - 1.0] = 0.0
        for i in range(n - 2, -1, -1):
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * decel * (self.s[i + 1] - self.s[i])))
        self.v = v
        ds = np.maximum(np.diff(self.s), 1e-6)
        self.a = np.append((v[1:] ** 2 - v[:-1] ** 2) / (2.0 * ds), 0.0)

    def nearest(self, x: float, y: float) -> int:
        return int(np.argmin(np.hypot(self.x - x, self.y - y)))

    def at(self, s: float) -> int:
        return int(min(np.searchsorted(self.s, s), len(self.s) - 1))

    def xy(self) -> np.ndarray:
        return np.column_stack((self.x, self.y))

    def rows(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        return np.column_stack((self.s, self.x, self.y, self.yaw, self.k))[start:stop]

    def node_rows(self) -> np.ndarray:
        return np.arange(self.root_row, len(self.s), ROWS_PER_STEP)


# ----------------------------------------------------------------------------- controller
@dataclass(frozen=True)
class ExplorationConfig:
    v_max_mps: float = 5.0
    ay_max_mps2: float = 3.0
    decel_mps2: float = 2.0
    kappa_rate: float = 0.8           # fastest curvature change the steering can make (1/(m s));
                                      # 0.8 is about 2 rad/s at the wheels for a 2.5 m wheelbase
    emergency_decel_mps2: float = 5.0
    replan_s: float = 0.2
    commit_m: float = 1.0             # the new tree starts this far beyond where the car will be
    rejoin_m: float = 0.4             # off the plan by more than this: start again from the car
    rejoin_rad: float = 0.25          # ... or with a heading error larger than this
    lookahead_gain_s: float = 0.45    # PFPP lookahead law
    lookahead_min_m: float = 3.5
    lookahead_max_m: float = 6.0


class ExplorationLateral:
    """Steering + speed for the exploration lap, in the runner's controller interface.

    observe() feeds the current (true) pose to the cone detector. compute() is called with
    the (delay-predicted, rear-axle) pose: it replans every replan_s and steers along the
    current local path with the PFPP law.
    """

    name = "explore"

    def __init__(self, perception: ConePerception, planner: KinoRRT, config: ExplorationConfig,
                 wheelbase_m: float, max_steer_rad: float) -> None:
        self.perception, self.planner, self.config = perception, planner, config
        self.wheelbase_m, self.max_steer_rad = wheelbase_m, max_steer_rad
        self.reset()

    def reset(self) -> None:
        self.path: LocalPath | None = None
        self._t = 0.0
        self._next_plan = 0.0
        self._kappa_cmd = 0.0
        self.plan_ms: list[float] = []
        self.failures = 0
        self.history: list[np.ndarray] = []           # chosen path of every replan
        self.history_t: list[float] = []
        self.trees: list[np.ndarray] = []             # every tree: rows (x, y, yaw, kappa, parent)
        self.replan_t: list[float] = []               # time of every replan (trees, plan_ms)
        self._speed = (0.0, 0.0)
        self.debug: list | None = None                 # set to [] to record every replan

    def observe(self, x: float, y: float, yaw: float) -> None:
        if self._t + 1e-9 >= self._next_plan:          # perception runs at the planning rate
            self.perception.observe(x, y, yaw)

    def _root(self, x: float, y: float, yaw: float, v: float):
        """Where the new tree starts.

        Normally on the current plan, a little beyond where the car will be by the next replan,
        so the path the car is on does not change under it (receding horizon). The part of the
        plan after that point is handed to the planner to be re-used. If the car is too far off
        the plan, the tree starts at the car instead.
        """
        path = self.path
        if path is not None:
            i = path.nearest(x, y)
            off = math.hypot(x - path.x[i], y - path.y[i])
            heading_err = abs(math.atan2(math.sin(yaw - path.yaw[i]), math.cos(yaw - path.yaw[i])))
            if off < self.config.rejoin_m and heading_err < self.config.rejoin_rad:
                ahead = path.s[i] + max(v, 1.0) * self.config.replan_s + self.config.commit_m
                nodes = path.node_rows()
                later = nodes[path.s[nodes] >= ahead]
                if len(later):
                    r = int(later[0])
                    keep_from = max(0, i - 8)              # 2 m of the path behind the car
                    prefix = path.rows(keep_from, r + 1)
                    chain_rows = nodes[nodes > r]
                    # One curvature per planner segment: the mean over the segment's rows.
                    chain = (np.array([path.k[n - ROWS_PER_STEP + 1:n + 1].mean() for n in chain_rows])
                             if len(chain_rows) else None)
                    k0 = float(path.k[r])
                    return (float(path.x[r]), float(path.y[r]), float(path.yaw[r])), k0, prefix, chain
        return (x, y, yaw), self._kappa_cmd, None, None

    def _replan(self, x: float, y: float, yaw: float, v: float) -> None:
        (rx, ry, ryaw), k0, prefix, chain = self._root(x, y, yaw, v)
        cones_xy, cones_col = self.perception.map_near(rx, ry, 40.0)
        t0 = time.perf_counter()
        branch = self.planner.plan(rx, ry, ryaw, k0, cones_xy, cones_col, chain)
        self.plan_ms.append(1000.0 * (time.perf_counter() - t0))
        n = self.planner.count
        pl = self.planner
        self.replan_t.append(self._t)
        self.trees.append(np.column_stack((pl.nx[:n], pl.ny[:n], pl.nyaw[:n], pl.nk[:n], pl.nparent[:n])).astype(np.float32))
        if self.debug is not None:
            self.debug.append((self._t, (rx, ry, ryaw), k0, prefix is not None,
                               None if chain is None else len(chain), self.planner.last_nodes,
                               dict(self.planner.reject), None if branch is None else float(branch[-1, 0]),
                               self.planner.tree_edges(), cones_xy.copy(), cones_col.copy(),
                               None if branch is None else branch.copy(),
                               None if self.planner.last_raw is None else self.planner.last_raw.copy()))
        if branch is None or branch[-1, 0] < 1.5:
            self.failures += 1
            return                                     # keep driving the previous path
        c = self.config
        if prefix is not None:
            branch = branch.copy()
            branch[:, 0] += prefix[-1, 0]
            samples = np.vstack((prefix, branch[1:]))
            root_row = len(prefix) - 1
        else:
            samples, root_row = branch, 0
        self.path = LocalPath(samples, c.v_max_mps, c.ay_max_mps2, c.decel_mps2, root_row, c.kappa_rate)
        self.history.append(self.path.xy()[::2].astype(np.float32))
        self.history_t.append(self._t)

    def compute(self, x: float, y: float, yaw: float, v: float, dt_s: float) -> LateralCommand:
        c = self.config
        if self._t + 1e-9 >= self._next_plan:
            self._replan(x, y, yaw, v)
            self._next_plan += c.replan_s
        self._t += dt_s
        path = self.path
        if path is None:                               # nothing planned yet: stand still
            self._speed = (0.0, 0.0)
            return LateralCommand(0.0, 0, 0, x, y, 0.0, 0.0, "explore")
        i = path.nearest(x, y)
        lookahead = min(max(c.lookahead_gain_s * v, c.lookahead_min_m), c.lookahead_max_m)
        j = path.at(path.s[i] + lookahead)
        tx, ty = float(path.x[j]), float(path.y[j])

        def pp(px, py, pyaw):
            dx, dy = tx - px, ty - py
            lx = math.cos(pyaw) * dx + math.sin(pyaw) * dy
            ly = -math.sin(pyaw) * dx + math.cos(pyaw) * dy
            return 0.0 if lx <= 0.0 else 2.0 * ly / max(lx * lx + ly * ly, 1e-6)

        # PFPP: pure pursuit correction, chord estimate swapped for the path's own curvature.
        kappa = pp(x, y, yaw) - pp(float(path.x[i]), float(path.y[i]), float(path.yaw[i])) + float(path.k[i])
        steering = max(-self.max_steer_rad, min(self.max_steer_rad, math.atan(self.wheelbase_m * kappa)))
        self._kappa_cmd = math.tan(steering) / self.wheelbase_m
        self._speed = (float(path.v[i]), float(path.a[i]))
        left = path.length - path.s[i]
        if left < v * v / (2.0 * c.decel_mps2) + 1.0:   # cannot stop in time on what is known
            self._speed = (0.0, -c.emergency_decel_mps2)
        # Signed distance to the local path (left = +), for the log.
        hx, hy = math.cos(path.yaw[i]), math.sin(path.yaw[i])
        xte = hx * (y - path.y[i]) - hy * (x - path.x[i])
        lat_ok = j < len(path.s) - 1
        return LateralCommand(steering, i, j, tx, ty, lookahead, float(xte),
                              "explore" if lat_ok else "explore, path end")

    def speed_target(self) -> tuple[float, float]:
        return self._speed

    def params(self) -> dict:
        out = {"exploration": self.config.__dict__, "kinorrt": self.planner.c.__dict__,
               "perception": self.perception.config.__dict__}
        return out

    def stats(self) -> dict:
        ms = np.array(self.plan_ms) if self.plan_ms else np.zeros(1)
        return {
            "replans": len(self.plan_ms),
            "failed_replans": self.failures,
            "plan_ms_mean": round(float(ms.mean()), 2),
            "plan_ms_p95": round(float(np.percentile(ms, 95)), 2),
            "plan_ms_max": round(float(ms.max()), 2),
            "tight_margin_plans": int(self.planner.tight_plans),
            "cones_mapped": int(len(self.perception.map_xy)),
            "cones_seen_of_total": [int(self.perception.ever_seen.sum()), int(len(self.perception.true_xy))],
        }
