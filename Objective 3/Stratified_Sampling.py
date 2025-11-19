import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------------------------------------
# CONFIG: set your folder and filename here
# By default this resolves paths relative to the script file
# (so running the script from another CWD still finds files
# placed next to the script). You can change `DATA_FOLDER`
# if you keep a dedicated `data/` subfolder.
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# default to script directory (change to os.path.join(BASE_DIR, 'data') if you store CSVs in a data/ subfolder)
DATA_FOLDER = os.path.join(BASE_DIR)
FILENAME = "Step 4 Resources/PR_Optimal_Range.csv"  # your CSV (can include subfolders)
FILE_PATH = os.path.normpath(os.path.join(DATA_FOLDER, FILENAME))

# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------

def read_data(path):
    return pd.read_csv(path)

def numeric_df(df, cols=None):
    if cols:
        return df[cols].select_dtypes(include=[np.number]).copy()
    else:
        return df.select_dtypes(include=[np.number]).copy()

def compute_mahalanobis_distances(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)

    try:
        inv_cov = linalg.inv(cov)
    except linalg.LinAlgError:
        reg = 1e-8 * np.eye(cov.shape[0])
        inv_cov = linalg.inv(cov + reg)

    diffs = X - mu
    d2 = np.einsum('ij,jk,ik->i', diffs, inv_cov, diffs)
    return np.sqrt(np.clip(d2, 0, None))

def mark_edges(df_num, pct=0.05):
    edges = {}
    for col in df_num.columns:
        vals = df_num[col].to_numpy()
        mn, mx = vals.min(), vals.max()
        rng = mx - mn if mx != mn else 1
        low = mn + pct * rng
        high = mx - pct * rng
        edges[col] = (vals <= low) | (vals >= high)
    return edges

def assign_strata(df_num, dist, edge_flags, quantiles=(0.25,0.5,0.75)):
    q1, q2, q3 = np.quantile(dist, quantiles)
    labels = []

    for i, d in enumerate(dist):
        flagged = [c for c, mask in edge_flags.items() if mask[i]]
        if flagged:
            labels.append("edge:" + "|".join(flagged))
            continue

        if d <= q1:
            labels.append("heart")
        elif d <= q2:
            labels.append("mid-low")
        elif d <= q3:
            labels.append("mid-high")
        else:
            labels.append("outer-edge")

    return pd.Series(labels, index=df_num.index)

def stratified_sample(df, strata, n_each=1, random_state=42):
    df = df.copy()
    df["_str"] = strata
    rng = np.random.RandomState(random_state)
    chosen = []

    for s in df["_str"].unique():
        grp = df[df["_str"] == s]
        k = min(n_each, len(grp))
        chosen.extend(rng.choice(grp.index, k, replace=False))

    return df.loc[chosen].drop(columns=["_str"])

def plot_strata(df_num, strata, sampled_idx):
    # PCA if >2 dims
    if df_num.shape[1] > 2:
        pca = PCA(n_components=2)
        comp = pca.fit_transform(df_num)
        x, y = comp[:,0], comp[:,1]
        xl, yl = "PC1", "PC2"
    else:
        x = df_num.iloc[:,0]
        y = df_num.iloc[:,1] if df_num.shape[1] > 1 else np.zeros(len(df_num))
        xl, yl = df_num.columns[:2]

    labs = strata.unique()
    cmap = plt.get_cmap("tab10")
    color_map = {lab: cmap(i % 10) for i, lab in enumerate(labs)}

    plt.figure(figsize=(8,6))
    for lab in labs:
        mask = strata == lab
        plt.scatter(x[mask], y[mask],
                    color=color_map[lab], label=lab,
                    edgecolor='k', alpha=0.7)

    sampled_mask = np.isin(df_num.index, sampled_idx)
    plt.scatter(x[sampled_mask], y[sampled_mask],
                facecolors='none', edgecolors='red',
                s=150, linewidths=1.5, label="sampled")

    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    plt.title("Stratified Sampling Coverage")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------
if __name__ == "__main__":

    # load data
    # If the primary path doesn't exist, try a few sensible fallbacks
    tried = [FILE_PATH]
    if not os.path.exists(FILE_PATH):
        alt1 = os.path.normpath(os.path.join(BASE_DIR, FILENAME))
        alt2 = os.path.normpath(os.path.join(BASE_DIR, os.path.basename(FILENAME)))
        tried.extend([alt1, alt2])
        if os.path.exists(alt1):
            FILE_PATH = alt1
        elif os.path.exists(alt2):
            FILE_PATH = alt2
        else:
            # try a quick recursive search under the script directory for the basename
            found = None
            for root, dirs, files in os.walk(BASE_DIR):
                if os.path.basename(FILENAME) in files:
                    found = os.path.join(root, os.path.basename(FILENAME))
                    tried.append(found)
                    break
            if found:
                FILE_PATH = found

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"CSV not found. Tried: {tried}")

    df = read_data(FILE_PATH)

    # set index robustly: prefer `Trial_ID`, fall back to `Workload`, else leave default
    if "Trial_ID" in df.columns:
        df = df.set_index("Trial_ID")
    elif "Workload" in df.columns:
        df = df.set_index("Workload")
    else:
        print("Warning: no 'Trial_ID' or 'Workload' column found; using default integer index.")

    # select numeric variables with variance
    strat_cols = [c for c in df.columns if df[c].nunique() > 1]
    df_num = numeric_df(df, cols=strat_cols)

    # 1) Mahalanobis
    dist = compute_mahalanobis_distances(df_num)

    # 2) per-variable edges
    edges = mark_edges(df_num)

    # 3) strata assignment
    strata = assign_strata(df_num, dist, edges)

    # 4) sampling
    sampled = stratified_sample(df, strata, n_each=1)
    print("\nSelected samples:")
    print(sampled)

    # 4.b) save sampled results to CSV (in the script directory)
    try:
        out_name = os.path.splitext(os.path.basename(FILE_PATH))[0] + "_stratified_sample.txt"
        out_path = os.path.join(BASE_DIR, out_name)
        sampled.to_txt(out_path, index=True)
        print(f"Saved stratified sample to: {out_path}")
    except Exception as e:
        print(f"Warning: failed to save sample to TXT: {e}")
    # 5) plot
    plot_strata(df_num, strata, sampled.index)
