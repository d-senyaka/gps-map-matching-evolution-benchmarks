"""
Convert the DiffMM Porto data into the three JSON files that HSTGMatch expects.

  python porto_to_hstgmatch.py --porto DiffMM/data/porto --out hstg_porto --keep_ratio 0.1 --n_train 6000 --n_extra 2000

Output folder (give it to HSTGMatch with --data_dir):
  trajectories.json  sparse GPS points (first + last point kept, plus a random share of the rest, same rule as DiffMM)
  labels.json        true route = the true road segment ids of ALL original points, repeated ids merged
  road_network.json  road segments (same ids as DiffMM)
HSTGMatch splits the data into train/val/test by itself (ratios in its config), so test trips are a random part of this pool.
"""
import argparse, json, os, random

def load_trajs(fn):
    T, cur = [], []
    for l in open(fn):
        l = l.strip()
        if not l: continue
        if l.startswith("-"): T.append(cur); cur = []
        else:
            t, la, lo, sg = l.split(); cur.append((int(t), float(la), float(lo), int(sg)))
    return T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--porto", default="DiffMM/data/porto"); ap.add_argument("--out", default="hstg_porto")
    ap.add_argument("--keep_ratio", type=float, default=0.1); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_train", type=int, default=6000, help="trips taken from traj_train.txt")
    ap.add_argument("--n_extra", type=int, default=2000, help="trips taken from traj_test.txt (extra pool)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = random.Random(a.seed)

    # road network
    nodes = {}
    for l in open(f"{a.porto}/roadnetwork/nodeOSM.txt"):
        p = l.split(); nodes[int(p[0])] = (float(p[1]), float(p[2]))          # node id -> (lat, lon)
    segs = []
    for l in open(f"{a.porto}/roadnetwork/edgeOSM.txt"):
        p = l.split("\t"); eid, u, v, n = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        c = [float(x) for x in p[4:4 + 2 * n]]
        geom = [{"lon": lo, "lat": la} for la, lo in zip(c[0::2], c[1::2])]    # file stores lat lon pairs
        segs.append({"seg_id": eid, "start_node": {"lon": nodes[u][1], "lat": nodes[u][0]},
                     "end_node": {"lon": nodes[v][1], "lat": nodes[v][0]}, "geometry": geom})
    segs.sort(key=lambda s: s["seg_id"])
    assert [s["seg_id"] for s in segs] == list(range(len(segs))), "segment ids must be 0..N-1"
    json.dump({"segments": segs}, open(f"{a.out}/road_network.json", "w"))

    # trips
    pool = load_trajs(f"{a.porto}/traj_train.txt")[:a.n_train] + load_trajs(f"{a.porto}/traj_test.txt")[:a.n_extra]
    trajs, labels = [], []
    for tid, tr in enumerate(pool):
        L = len(tr)
        if L < 3: continue
        mid = rng.sample(range(1, L - 1), int((L - 2) * a.keep_ratio))
        keep = [0] + sorted(mid) + [L - 1]
        trajs.append({"traj_id": len(trajs), "points": [{"lon": tr[i][2], "lat": tr[i][1], "timestamp": tr[i][0]} for i in keep]})
        route = []
        for p in tr:
            if not route or route[-1] != p[3]: route.append(p[3])
        labels.append({"traj_id": len(labels), "route": route})
    json.dump(trajs, open(f"{a.out}/trajectories.json", "w")); json.dump(labels, open(f"{a.out}/labels.json", "w"))
    print(f"wrote {len(trajs)} trips, {len(segs)} segments to {a.out}/ (keep_ratio={a.keep_ratio})")

if __name__ == "__main__":
    main()
