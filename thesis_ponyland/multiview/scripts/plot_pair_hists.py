import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True, help="CSV from score_pairs_true2view.py")
    ap.add_argument("--outdir", required=True, help="Output directory for PNGs")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--min_pairs", type=int, default=50, help="Skip classes with fewer pairs than this")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.pairs_csv)

    required = {"base_class", "v_ab"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pairs_csv missing columns: {missing}. Found: {df.columns.tolist()}")

    # Ensure numeric
    df["v_ab"] = pd.to_numeric(df["v_ab"], errors="coerce")
    df = df.dropna(subset=["v_ab", "base_class"])

    # Plot one histogram per class
    for cls, g in df.groupby("base_class"):
        if len(g) < args.min_pairs:
            continue

        best = g["v_ab"].max()

        plt.figure(figsize=(7.2, 4.2))
        plt.hist(g["v_ab"], bins=args.bins)
        plt.axvline(best, linewidth=2, label=f"Best pair = {best:.3f}")

        plt.title(f"{cls}: distribution of TRUE 2-view scores")
        plt.xlabel(r"$v(\{i,j\})$  (pair score)")
        plt.ylabel("Number of view pairs")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(args.outdir, f"hist_{cls}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

    print(f"[plot_pair_hists] wrote histograms to: {args.outdir}")


if __name__ == "__main__":
    main()
