# build_lastfm_small.py
import pandas as pd
import numpy as np
from pathlib import Path


# ----------------- knobs for quick demos -----------------
RECENCY_MONTHS = 0  # 0 = no time filter; try 1 or 6 for your T-CLDS story
MIN_USER_INTER = 2  # drop users with <2 interactions (so each has train+test)
SAMPLE_USERS = 500  # set None to keep all users
TOPK_ITEMS = 5000  # set None to keep all items
RNG_SEED = 42
# ---------------------------------------------------------

RAW = Path("data/raw/lastfm")
OUT = Path(f"data/preprocessed/lastfm_small_m{RECENCY_MONTHS}")
OUT.mkdir(parents=True, exist_ok=True)


rng = np.random.default_rng(RNG_SEED)

# 1) Load core files (tab-separated)
ua = pd.read_csv(RAW / "user_artists.dat", sep="\t")  # userID, artistID, weight
uts = pd.read_csv(
    RAW / "user_taggedartists-timestamps.dat", sep="\t"
)  # userID, artistID, tagID, timestamp(ms)
uf = pd.read_csv(RAW / "user_friends.dat", sep="\t")  # userID, friendID

# 2) One timestamp per (user, artist): latest tag time, ms->s
ts = uts.groupby(["userID", "artistID"])["timestamp"].max().reset_index()
ts["time"] = (ts["timestamp"] // 1000).astype("int64")
ts = ts.drop(columns=["timestamp"])

# 3) Join timestamps to interactions (unique (u,i))
df = (
    ua[["userID", "artistID"]]
    .drop_duplicates()
    .merge(ts, on=["userID", "artistID"], how="left")
)

# 4) Fill missing time: per-user median, then global median
user_med = df.groupby("userID")["time"].transform(lambda s: s.fillna(s.median()))
glob_med = df["time"].median()
df["time"] = user_med.fillna(glob_med).astype("int64")

# ----------------- optional: recency filter -----------------
if RECENCY_MONTHS and RECENCY_MONTHS > 0:
    cutoff = int(df["time"].max() - RECENCY_MONTHS * 30 * 24 * 60 * 60)  # month≈30 days
    df = df[df["time"] >= cutoff].copy()

# ----------------- sparsity control (per-user >=2) ----------
cnt = df.groupby("userID").size()
keep_users = set(cnt[cnt >= MIN_USER_INTER].index)
df = df[df["userID"].isin(keep_users)].copy()

# ----------------- user sampling (for speed) ----------------
if SAMPLE_USERS is not None and SAMPLE_USERS < df["userID"].nunique():
    sampled = rng.choice(df["userID"].unique(), size=SAMPLE_USERS, replace=False)
    df = df[df["userID"].isin(sampled)].copy()

# ----------------- item pruning (top-K popular) -------------
if TOPK_ITEMS is not None:
    top_items = df["artistID"].value_counts().head(TOPK_ITEMS).index
    df = df[df["artistID"].isin(top_items)].copy()

# Re-check sparsity after pruning; keep users with >=2
cnt = df.groupby("userID").size()
keep_users = set(cnt[cnt >= MIN_USER_INTER].index)
df = df[df["userID"].isin(keep_users)].copy()

# 5) Reindex to contiguous ints your loader expects
u_cat = pd.Categorical(df["userID"])
i_cat = pd.Categorical(df["artistID"])
df["user"] = u_cat.codes.astype(int)
df["item"] = i_cat.codes.astype(int)
df = df[["user", "item", "time"]]

# 6) Temporal split: most recent 1 per user -> test, rest -> train
df = df.sort_values(["user", "time"])
df["rank"] = df.groupby("user")["time"].rank(method="first", ascending=False)
test = df[df["rank"] <= 1][["user", "item", "time"]].reset_index(drop=True)
train = df[df["rank"] > 1][["user", "item", "time"]].reset_index(drop=True)

# Safety: if any user lost their train rows (rare), drop them entirely
bad_users = set(test["user"]) - set(train["user"])
if bad_users:
    test = test[~test["user"].isin(bad_users)]

# 7) Trust graph remap & filter to sampled users only
uf_small = uf.copy()
uf_small["user"] = pd.Categorical(uf_small["userID"], categories=u_cat.categories).codes
uf_small["friend"] = pd.Categorical(
    uf_small["friendID"], categories=u_cat.categories
).codes
# drop edges to users not in the sampled/reindexed set (coded as -1)
uf_small = uf_small[(uf_small["user"] >= 0) & (uf_small["friend"] >= 0)]
# keep edges among users that remain after all filtering
keep_u = set(train["user"]).union(set(test["user"]))
uf_small = uf_small[uf_small["user"].isin(keep_u) & uf_small["friend"].isin(keep_u)]
# optional: remove self-loops & duplicates
uf_small = uf_small[uf_small["user"] != uf_small["friend"]].drop_duplicates(
    subset=["user", "friend"]
)

# 8) Write files your code expects
train.to_csv(OUT / "train_set.txt", index=False)
test.to_csv(OUT / "test_set.txt", index=False)
uf_small[["user", "friend"]].to_csv(OUT / "trust.txt", index=False)

print(
    {
        "users": len(keep_u),
        "items": int(
            max(train["item"].max() if len(train) else -1, test["item"].max()) + 1
        ),
        "train_interactions": len(train),
        "test_interactions": len(test),
        "trust_edges": len(uf_small),
    }
)
print("done ->", OUT)
