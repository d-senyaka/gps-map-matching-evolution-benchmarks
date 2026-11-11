"""
Map matching experiment: two historic methods on the DiffMM Porto data.

  Method 1: nearest-road snapping (point-to-curve), White et al. (2000)
  Method 2: HMM map matching (emission + transition + Viterbi), Newson & Krumm (2009)

Usage (from the folder that contains the DiffMM repo):
  pip install numpy scipy shapely matplotlib
  git clone https://github.com/decisionintelligence/DiffMM.git

  # 1) tune the HMM on the VALIDATION split (never tune on test)
  python mm_experiment.py --split valid --n 200 --tune

  # 2) final run on the TEST split with the chosen parameters
  python mm_experiment.py --split test --n 1000 --sigma 10 --beta 50

Outputs: results.csv and results.png (accuracy vs gap) + a printed table.
"""
import argparse, csv, math, random, time
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# ---------- 1. Load road network and trajectories (DiffMM Porto format) ----------
LAT0 = 41.15
KX = 111320 * math.cos(math.radians(LAT0)); KY = 110540   # metres per degree (local, Porto)
def xy(lat, lon): return ((lon + 8.6) * KX, (lat - LAT0) * KY)

def load_network(root):
    nodes = {}
    for l in open(f"{root}/roadnetwork/nodeOSM.txt"):
        p = l.split(); nodes[int(p[0])] = (float(p[1]), float(p[2]))
    edges = {}                                   # edge id -> (from_node, to_node, LineString, length_m)
    for l in open(f"{root}/roadnetwork/edgeOSM.txt"):
        p = l.split("\t"); eid, u, v, n = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        c = [float(x) for x in p[4:4 + 2 * n]]
        ls = LineString([xy(a, b) for a, b in zip(c[0::2], c[1::2])])
        edges[eid] = (u, v, ls, ls.length)
    return nodes, edges

def load_trajs(fn):
    """each trajectory = list of (timestamp, lat, lon, true_edge_id); trajectories end with a '-N' line"""
    T, cur = [], []
    for l in open(fn):
        l = l.strip()
        if not l: continue
        if l.startswith("-"): T.append(cur); cur = []
        else:
            t, la, lo, sg = l.split(); cur.append((int(t), float(la), float(lo), int(sg)))
    return T

class Net:
    def __init__(self, root):
        self.nodes, self.E = load_network(root)
        self.ids = sorted(self.E)
        self.tree = STRtree([self.E[e][2] for e in self.ids])
        n = max(self.nodes) + 1; best = {}
        for e, (u, v, ls, L) in self.E.items():
            best[(u, v)] = min(best.get((u, v), 1e18), max(L, 0.01))
        r, c, w = zip(*[(u, v, L) for (u, v), L in best.items()])
        self.G = csr_matrix((w, (r, c)), shape=(n, n))      # directed road graph, weight = metres

    def candidates(self, p, radius):
        pt = Point(*p); out = []
        for i in self.tree.query(pt.buffer(radius)):
            e = self.ids[i]; ls = self.E[e][2]; d = ls.distance(pt)
            if d <= radius: out.append((e, ls.project(pt), d))      # (edge, offset along edge, distance)
        if not out:                                                 # nothing within radius -> nearest edge
            e = self.ids[int(self.tree.nearest(pt))]; ls = self.E[e][2]
            out = [(e, ls.project(pt), ls.distance(pt))]
        return out

# ---------- 2. Method 1: nearest-road snapping (White et al. 2000) ----------
def match_snap(net, obs):
    """each GPS point is snapped to the closest road segment, independently of the others"""
    res = []
    for p in obs:
        e = net.ids[int(net.tree.nearest(Point(*p)))]
        res.append(e)
    return res

# ---------- 3. Method 2: HMM + Viterbi (Newson & Krumm 2009) ----------
def match_hmm(net, obs, radius=50, sigma=10, beta=50):
    """
    emission  : -0.5*(d/sigma)^2            (closer road = more likely; models GPS noise)
    transition: -|great_circle - route|/beta (route length close to straight-line gap = plausible)
    Viterbi picks the most likely chain of road segments.
    """
    C = [net.candidates(p, radius) for p in obs]
    prev = [-0.5 * (d / sigma) ** 2 for (_, _, d) in C[0]]; back = []
    for t in range(1, len(obs)):
        dgc = math.dist(obs[t - 1], obs[t]); cutoff = max(500.0, 3 * dgc + 200)
        sp = {}
        for (e1, _, _) in C[t - 1]:
            n = net.E[e1][1]
            if n not in sp: sp[n] = dijkstra(net.G, directed=True, indices=n, limit=cutoff)
        cur, bp = [], []
        for (e2, o2, d2) in C[t]:
            bv, bi = -1e18, -1
            for i, (e1, o1, _) in enumerate(C[t - 1]):
                if e1 == e2 and o2 >= o1: route = o2 - o1
                else:
                    s = sp[net.E[e1][1]][net.E[e2][0]]
                    if not np.isfinite(s): continue
                    route = (net.E[e1][3] - o1) + s + o2
                v = prev[i] - abs(dgc - route) / beta
                if v > bv: bv, bi = v, i
            cur.append(-1e18 if bi < 0 else bv - 0.5 * (d2 / sigma) ** 2); bp.append(bi)
        if max(cur) <= -1e17:                    # no route found: restart the chain from emissions only
            cur = [-0.5 * (d / sigma) ** 2 for (_, _, d) in C[t]]; bp = [-1] * len(cur)
        back.append(bp); prev = cur
    k = int(np.argmax(prev)); path = [k]
    for bp in reversed(back):
        k = bp[k] if bp[k] >= 0 else 0; path.append(k)
    path.reverse()
    return [C[t][path[t]][0] for t in range(len(obs))]

# ---------- 4. Evaluation (same sampling rule as DiffMM) ----------
def sparsify(tr, ratio, rng):
    """keep first + last point and a random fraction `ratio` of the points in between (as DiffMM does)"""
    L = len(tr)
    mid = rng.sample(range(1, L - 1), int((L - 2) * ratio)) if L > 2 else []
    return [0] + sorted(mid) + [L - 1]

def evaluate(net, trajs, ratio, method, **kw):
    accs, secs = [], 0.0
    for idx, tr in enumerate(trajs):
        if len(tr) < 3: continue
        keep = sparsify(tr, ratio, random.Random(1000 * idx + int(ratio * 1000)))   # same points for both methods
        obs = [xy(tr[i][1], tr[i][2]) for i in keep]; truth = [tr[i][3] for i in keep]
        t0 = time.time(); pred = method(net, obs, **kw); secs += time.time() - t0
        accs.append(np.mean([p == t for p, t in zip(pred, truth)]))
    return 100 * float(np.mean(accs)), secs / len(accs) * 1000     # accuracy %, seconds per 1000 trajectories

GAPS = {0.2: 75, 0.1: 150, 0.05: 300, 0.025: 600}               # kept fraction -> average gap (s), 15 s data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="DiffMM/data/porto"); ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--tune", action="store_true")
    ap.add_argument("--sigma", type=float, default=10); ap.add_argument("--beta", type=float, default=50)
    a = ap.parse_args()
    net = Net(a.root); trajs = load_trajs(f"{a.root}/traj_{a.split}.txt")[:a.n]
    print(f"{len(net.E)} road segments, {len(trajs)} {a.split} trajectories")
    if a.tune:                                                    # grid search on the validation split
        rows = []
        for s in (5, 10, 20, 30):
            for b in (20, 50, 100, 200):
                accs = [evaluate(net, trajs, r, match_hmm, sigma=s, beta=b)[0] for r in (0.1, 0.05)]
                rows.append((np.mean(accs), s, b)); print(f"sigma={s:>3} beta={b:>4}  mean acc @150s,300s = {np.mean(accs):.1f}%")
        best = max(rows); print(f"\nBEST on {a.split}: sigma={best[1]} beta={best[2]} ({best[0]:.1f}%)")
    else:
        out = []
        for r, gap in GAPS.items():
            s_acc, s_t = evaluate(net, trajs, r, match_snap)
            h_acc, h_t = evaluate(net, trajs, r, match_hmm, sigma=a.sigma, beta=a.beta)
            out.append((gap, s_acc, h_acc, s_t, h_t))
        print(f"\n{'gap(s)':>7} {'snap acc%':>10} {'HMM acc%':>9} {'snap s/1000':>12} {'HMM s/1000':>11}")
        for g, sa, ha, st, ht in out: print(f"{g:>7} {sa:>10.1f} {ha:>9.1f} {st:>12.2f} {ht:>11.2f}")
        with open("results.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["gap_s", "snap_acc", "hmm_acc", "snap_s_per_1000", "hmm_s_per_1000"]); w.writerows(out)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3.4)); g = [o[0] for o in out]
        plt.plot(g, [o[1] for o in out], "o-", label="Nearest-road snapping (2000)")
        plt.plot(g, [o[2] for o in out], "s-", label="HMM (2009)")
        plt.xlabel("Average gap between GPS points (s)"); plt.ylabel("Point accuracy (%)"); plt.ylim(0, 100)
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig("results.png", dpi=200); print("saved results.csv, results.png")
