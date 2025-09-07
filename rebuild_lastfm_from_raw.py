import pandas as pd
from pathlib import Path
import numpy as np

RAW = Path("data/raw/lastfm")
OUT = Path("data/preprocessed/lastfm")
OUT.mkdir(parents=True, exist_ok=True)

# 1) Load core files (tab-separated)
ua = pd.read_csv(RAW / "user_artists.dat", sep="\t")  # userID, artistID, weight
uts = pd.read_csv(
    RAW / "user_taggedartists-timestamps.dat", sep="\t"
)  # userID, artistID, tagID, timestamp (ms)

# 2) Reduce timestamps to one per (user, artist): take the latest tag time
ts = uts.groupby(["userID", "artistID"])["timestamp"].max().reset_index()
# ms -> seconds
ts["time"] = (ts["timestamp"] // 1000).astype("int64")
ts = ts.drop(columns=["timestamp"])

# 3) Join timestamps to user_artists (keep only pairs that exist in interactions)
df = (
    ua[["userID", "artistID"]]
    .drop_duplicates()
    .merge(ts, on=["userID", "artistID"], how="left")
)

# 4) Fill missing times: per-user median, then global median (as fallback)
user_med = df.groupby("userID")["time"].transform(lambda s: s.fillna(s.median()))
glob_med = df["time"].median()
df["time"] = user_med.fillna(glob_med).astype("int64")

# 5) Reindex to contiguous ints expected by your loader
u_cat = pd.Categorical(df["userID"])
i_cat = pd.Categorical(df["artistID"])
df["user"] = u_cat.codes.astype(int)
df["item"] = i_cat.codes.astype(int)
df = df[["user", "item", "time"]]

# 6) Temporal split: most recent 1 per user → test
df = df.sort_values(["user", "time"])
df["rank"] = df.groupby("user")["time"].rank(method="first", ascending=False)
test = df[df["rank"] <= 1][["user", "item", "time"]].reset_index(drop=True)
train = df[df["rank"] > 1][["user", "item", "time"]].reset_index(drop=True)

# 7) Write files your code expects
train.to_csv(OUT / "train_set.txt", index=False)
test.to_csv(OUT / "test_set.txt", index=False)

# 8) Trust graph passthrough
trust = pd.read_csv(RAW / "user_friends.dat", sep="\t")  # userID, friendID
# map to new indices
trust["user"] = pd.Categorical(trust["userID"], categories=u_cat.categories).codes
trust["friend"] = pd.Categorical(trust["friendID"], categories=u_cat.categories).codes
trust[["user", "friend"]].to_csv(OUT / "trust.txt", index=False)

print("done.")
