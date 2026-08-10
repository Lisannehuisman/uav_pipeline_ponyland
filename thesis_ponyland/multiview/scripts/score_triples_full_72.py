#!/usr/bin/env python3
import os, argparse
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_csv", required=True)
    ap.add_argument("--out_triples", required=True)
    ap.add_argument("--out_best_per_class", required=True)
    ap.add_argument("--metric", choices=["oracle_max","mean"], default="oracle_max")
    ap.add_argument("--flush_instances", type=int, default=1)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.single_csv)
    need = {"instance_id","base_class","view_id","v_single"}
    if not need.issubset(df.columns):
        raise KeyError(f"Need {need}, have {df.columns.tolist()}")

    df["v_single"] = pd.to_numeric(df["v_single"], errors="coerce").fillna(0.0)

    # sanity: must be 72 per instance
    vc = df.groupby("instance_id")["view_id"].nunique()
    if args.debug:
        print("[debug] views/inst min/mean/max:", int(vc.min()), float(vc.mean()), int(vc.max()))
    if int(vc.min()) != 72 or int(vc.max()) != 72:
        raise RuntimeError("Not 72 views per instance. Fix single_csv first.")

    groups = list(df.groupby(["instance_id","base_class"], sort=False))

    os.makedirs(os.path.dirname(args.out_triples), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_best_per_class), exist_ok=True)

    out_cols = ["instance_id","base_class","view_a","view_b","view_c","v_abc"]
    wrote_header = False
    best_inst_rows = []

    for start in tqdm(range(0, len(groups), args.flush_instances), desc="instances"):
        chunk = groups[start:start+args.flush_instances]
        out_rows = []

        for (inst, cls), sub in chunk:
            sub = sub.sort_values("view_id")
            views = sub["view_id"].tolist()
            vals  = sub["v_single"].to_numpy(dtype=float)

            best_v = -1.0
            best_t = None

            for i,j,k in combinations(range(72), 3):
                if args.metric == "oracle_max":
                    vabc = float(max(vals[i], vals[j], vals[k]))
                else:  # mean
                    vabc = float((vals[i] + vals[j] + vals[k]) / 3.0)

                a,b,c = views[i], views[j], views[k]
                out_rows.append([inst, cls, a, b, c, vabc])

                if vabc > best_v:
                    best_v = vabc
                    best_t = (a,b,c)

            best_inst_rows.append([inst, cls, best_t[0], best_t[1], best_t[2], float(best_v)])

        pd.DataFrame(out_rows, columns=out_cols).to_csv(
            args.out_triples, mode="a", header=(not wrote_header), index=False
        )
        wrote_header = True

    best_inst = pd.DataFrame(best_inst_rows, columns=out_cols)
    best_cls = (best_inst.sort_values("v_abc", ascending=False)
                .groupby("base_class", as_index=False)
                .head(1)
                .reset_index(drop=True))
    best_cls.to_csv(args.out_best_per_class, index=False)

    print("[OK] wrote:", args.out_triples)
    print("[OK] wrote:", args.out_best_per_class)

if __name__ == "__main__":
    main()
