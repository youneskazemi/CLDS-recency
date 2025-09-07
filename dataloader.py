import gc
import json
import os
import re
from time import time
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch
import copy
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import world


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


class PairDataset:
    def __init__(self, src="lastfm"):
        self.src = src

        self.train_set = pd.read_csv(f"./data/preprocessed/{src}/train_set.txt")
        self.test_set = pd.read_csv(f"./data/preprocessed/{src}/test_set.txt")

        self.n_user = pd.concat([self.train_set, self.test_set])["user"].nunique()
        self.m_item = pd.concat([self.train_set, self.test_set])["item"].nunique()
        self.trainUser = np.array(self.train_set["user"])
        self.trainUniqueUser = np.unique(self.train_set["user"])
        self.trainItem = np.array(self.train_set["item"])
        self._trainDataSize = len(self.train_set)
        self._testDataSize = len(self.test_set)
        print(f"{self._trainDataSize} interactions for training")
        print(f"{self._testDataSize} interactions for testing")
        print(f"Number of users: {self.n_user}\n Number of items: {self.m_item}")
        print(f"Number of Ratings: {self._trainDataSize + self._testDataSize}")
        print(
            f"{world.dataset} Rating Density: {(self._trainDataSize + self._testDataSize) / self.n_user / self.m_item}"
        )

        # build (users,items), bipartite graph
        self.interactionGraph = None

        # === Temporal edge weights (train only) ===
        ts_col = "time" if "time" in self.train_set.columns else None
        if ts_col is None:
            # fallback: original unweighted graph
            edge_vals = np.ones(len(self.train_set), dtype=np.float32)
        else:
            now_ts = int(max(self.train_set[ts_col].max(), self.test_set[ts_col].max()))
            half_life_days = float(world.config.get("half_life_days", 90.0))
            half_life_secs = half_life_days * 24 * 3600.0
            ages = now_ts - self.train_set[ts_col].to_numpy(dtype="int64")
            edge_vals = np.exp(-np.log(2.0) * (ages / half_life_secs)).astype(
                np.float32
            )
            # avoid all-zero rows/cols by flooring very tiny weights
            edge_vals = np.clip(edge_vals, 1e-6, None)

            # === Option: time-decayed edge weights on the train graph ===
        use_decay = bool(world.config.get("edge_time_decay", 0))
        has_time = "time" in self.train_set.columns
        if use_decay and has_time:
            now_ts = int(
                max(
                    self.train_set["time"].max(),
                    (
                        self.test_set["time"].max()
                        if "time" in self.test_set.columns
                        else self.train_set["time"].max()
                    ),
                )
            )
            half_life_days = float(world.config.get("half_life_days", 90.0))
            half_life_secs = max(1.0, half_life_days * 24 * 3600.0)
            ages = now_ts - self.train_set["time"].to_numpy(dtype="int64")
            edge_vals = np.exp(-np.log(2.0) * (ages / half_life_secs)).astype(
                np.float32
            )
            # avoid exact zeros so rows/cols don't vanish
            edge_vals = np.clip(edge_vals, 1e-6, None)
        else:
            edge_vals = np.ones(len(self.train_set), dtype=np.float32)

        self.UserItemNet = csr_matrix(
            (edge_vals, (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item),
        )
        #  user's history interacted items
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        # get test dictionary
        self._testDic = self.__build_test()
        self._coldTestDic = self.__build_cold_test()
        self._userDic, self._itemDic = self._getInteractionDic()

    @property
    def userDic(self):
        return self._userDic

    @property
    def itemDic(self):
        return self._itemDic

    @property
    def testDict(self):
        return self._testDic

    @property
    def coldTestDict(self):
        return self._coldTestDic

    @property
    def allPos(self):
        return self._allPos

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self._trainDataSize

    def getUserPosItems(self, users):
        """
        Method of get user all positive items
        Returns
        -------
        [ndarray0,...,ndarray_users]
        """
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
            # item_u = self.UserItemNet[self.UserItemNet['user'] == user]
            # item_u = item_u['item'].values
            # posItems.append(item_u)
        return posItems

    def __build_test(self):
        """
        Method of build test dictionary
        Returns
        -------
            dict: {user: [items]}
        """
        test_data = {}
        for i in range(len(self.test_set)):
            user = self.test_set["user"][i]
            item = self.test_set["item"][i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def __build_cold_test(self):
        # build test dict
        test_data = {}
        for idx in range(len(self.test_set)):
            u = int(self.test_set["user"].iloc[idx])
            it = int(self.test_set["item"].iloc[idx])
            test_data.setdefault(u, []).append(it)
        # users with <=20 training interactions are "cold"
        vc = self.train_set["user"].value_counts()
        cold = {}
        for u, items in test_data.items():
            # safe lookup: default 0 if user not in training set
            if int(vc.get(u, 0)) <= 20 and len(items) > 0:
                cold[u] = items
        return cold

    def _getInteractionDic(self):
        user_interaction = {}
        item_interaction = {}

        def getDict(_set):
            for i in range(len(_set)):
                user = _set["user"][i]
                item = _set["item"][i]
                if user_interaction.get(user):
                    user_interaction[user].append(item)
                else:
                    user_interaction[user] = [item]
                if item_interaction.get(item):
                    item_interaction[item].append(user)
                else:
                    item_interaction[item] = [user]

        getDict(self.train_set)
        getDict(self.test_set)
        return user_interaction, item_interaction


class GraphDataset:
    def __init__(self, src="lastfm"):
        self.src = src
        self.train_set = pd.read_csv(f"./data/preprocessed/{src}/train_set.txt")
        self.test_set = pd.read_csv(f"./data/preprocessed/{src}/test_set.txt")
        self.friendNet = pd.read_csv(f"./data/preprocessed/{src}/trust.txt")

        self.n_user = pd.concat([self.train_set, self.test_set])["user"].nunique()
        self.m_item = pd.concat([self.train_set, self.test_set])["item"].nunique()
        self.trainUser = np.array(self.train_set["user"])  # train user
        self.trainUniqueUser = np.unique(self.train_set["user"])
        self.trainItem = np.array(self.train_set["item"])  # train item
        self._trainDataSize = len(self.train_set)
        self._testDataSize = len(self.test_set)
        print(f"{self._trainDataSize} interactions for training")
        print(f"{self._testDataSize} interactions for testing")
        print(f"Number of users: {self.n_user}\n Number of items: {self.m_item}")
        print(f"Number of Ratings: {self._trainDataSize + self._testDataSize}")
        print(
            f"{world.dataset} Rating Density: {(self._trainDataSize + self._testDataSize) / self.n_user / self.m_item}"
        )

        self.interactionGraph = None

        self.UserItemNet = csr_matrix(
            (np.ones(len(self.train_set)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item),
        )
        # list
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        # dict
        self._testDic = self.__build_test()
        self._coldTestDic = self.__build_cold_test()
        # Timestamp handling
        self._ts_col = self.__detect_ts_col(self.train_set, self.test_set)
        self._max_ts = self.__compute_max_ts(
            self.train_set, self.test_set, self._ts_col
        )

        # === Build user/item time buckets for node embeddings ===
        self.user_time_bucket = None
        self.item_time_bucket = None
        if self._ts_col is not None and self._max_ts is not None:
            # last activity per user / item (consider both train+test)
            all_df = pd.concat([self.train_set, self.test_set], ignore_index=True)
            last_user = all_df.groupby("user")[self._ts_col].max()
            last_item = all_df.groupby("item")[self._ts_col].max()
            # compute age in days relative to max_ts
            u_age_days = (self._max_ts - last_user).astype("int64") / (24 * 3600)
            i_age_days = (self._max_ts - last_item).astype("int64") / (24 * 3600)
            bucket_days = int(world.config.get("time_bucket_days", 30))
            u_bucket = np.floor(u_age_days / bucket_days).astype(int)
            i_bucket = np.floor(i_age_days / bucket_days).astype(int)
            # store as dense arrays indexed by id
            self.user_time_bucket = np.zeros(self.n_user, dtype=int)
            self.user_time_bucket[last_user.index.to_numpy()] = u_bucket.to_numpy()
            self.item_time_bucket = np.zeros(self.m_item, dtype=int)
            self.item_time_bucket[last_item.index.to_numpy()] = i_bucket.to_numpy()
            # Optional cap to avoid huge embedding tables
            self.max_time_bucket = int(
                max(self.user_time_bucket.max(), self.item_time_bucket.max(), 0)
            )
        else:
            self.max_time_bucket = 0
        # dict
        self._userDic, self._itemDic = self._getInteractionDic()

    @property
    def userDic(self):
        return self._userDic

    @property
    def itemDic(self):
        return self._itemDic

    @property
    def testDict(self):
        return self._testDic

    @property
    def coldTestDict(self):
        return self._coldTestDic

    @property
    def allPos(self):
        return self._allPos

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self._trainDataSize

    # ---------- NEW: recency-aware evaluation dict ----------

    def get_eval_dict(self, recency_months: int = 0, cold: bool = False):
        if recency_months == 0 or self._ts_col is None or self._max_ts is None:
            return self._coldTestDic if cold else self._testDic

        cutoff = self._max_ts - recency_months * 30 * 24 * 60 * 60  # month≈30d
        df = self.test_set
        recent = df[df[self._ts_col] >= cutoff]

        test_data = {}
        for i in range(len(recent)):
            u = int(recent["user"].iloc[i])
            it = int(recent["item"].iloc[i])
            test_data.setdefault(u, []).append(it)

        if cold:
            vc = self.train_set["user"].value_counts()
            cold_users = {u for u in test_data.keys() if int(vc.get(u, 0)) <= 20}
            test_data = {u: test_data[u] for u in cold_users if len(test_data[u]) > 0}
            return test_data
        else:
            test_data = {u: items for u, items in test_data.items() if len(items) > 0}
            return test_data

    # ---------- helpers ----------
    @staticmethod
    def __detect_ts_col(train_df: pd.DataFrame, test_df: pd.DataFrame):
        candidates = ["timestamp", "time", "ts"]
        for c in candidates:
            if c in train_df.columns and c in test_df.columns:
                return c
        return None

    @staticmethod
    def __compute_max_ts(train_df: pd.DataFrame, test_df: pd.DataFrame, ts_col: str):
        if ts_col is None:
            return None
        try:
            return int(max(train_df[ts_col].max(), test_df[ts_col].max()))
        except Exception:
            return None

    def getUserPosItems(self, users):
        """
        Method of get user all positive items
        Returns
        -------
        [ndarray0,...,ndarray_users]
        """
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
            # item_u = self.UserItemNet[self.UserItemNet['user'] == user]
            # item_u = item_u['item'].values
            # posItems.append(item_u)
        return posItems

    def __build_test(self):
        """
        Method of build test dictionary
        Returns
        -------
            dict: {user: [items]}
        """
        test_data = {}
        for i in range(len(self.test_set)):
            user = self.test_set["user"][i]
            item = self.test_set["item"][i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def __build_cold_test(self):
        # build test dict
        test_data = {}
        for idx in range(len(self.test_set)):
            u = int(self.test_set["user"].iloc[idx])
            it = int(self.test_set["item"].iloc[idx])
            test_data.setdefault(u, []).append(it)
        # users with <=20 training interactions are "cold"
        vc = self.train_set["user"].value_counts()
        cold = {}
        for u, items in test_data.items():
            # safe lookup: default 0 if user not in training set
            if int(vc.get(u, 0)) <= 20 and len(items) > 0:
                cold[u] = items
        return cold

    def getInteractionGraph(self):
        print("loading adjacency matrix")
        if self.interactionGraph is None:
            try:
                decay_tag = (
                    "decay" if world.config.get("edge_time_decay", 0) else "plain"
                )
                pre_adj_mat = sp.load_npz(
                    f"./data/preprocessed/{self.src}/interaction_adj1_mat__{decay_tag}.npz"
                )
                pre_adj_mat2 = sp.load_npz(
                    f"./data/preprocessed/{self.src}/interaction_adj2_mat__{decay_tag}.npz"
                )
                print("successfully loaded...")
                norm_adj = pre_adj_mat
                norm2_adj = pre_adj_mat2
            except IOError:
                print("generating adjacency matrix")
                start = time()

                adj_mat = sp.dok_matrix(
                    (self.n_user + self.m_item, self.n_user + self.m_item),
                    dtype=np.float32,
                )
                adj_mat = adj_mat.tolil()
                adj2_mat = sp.dok_matrix(
                    (self.n_user + self.m_item, self.n_user + self.m_item),
                    dtype=np.float32,
                )
                adj2_mat = adj2_mat.tolil()
                R = self.UserItemNet.tolil()
                U_U = R.dot(R.T)
                I_I = R.T.dot(R)
                print("u_u is", U_U.shape)
                print("i_i is", I_I.shape)
                print("n_user", self.n_user)
                print("adj_mat", adj_mat.shape)

                adj_mat[: self.n_user, self.n_user :] = R
                adj_mat[self.n_user :, : self.n_user] = R.T
                adj_mat = adj_mat.todok()

                adj2_mat[: self.n_user, : self.n_user] = U_U
                adj2_mat[self.n_user :, self.n_user :] = I_I
                adj2_mat = adj2_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))  # D
                d_inv = np.power(rowsum, -0.5).flatten()  # D^-0.5
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                rowsum2 = np.array(adj2_mat.sum(axis=1))  # D
                d2_inv = np.power(rowsum2, -0.5).flatten()  # D^-0.5
                d2_inv[np.isinf(d2_inv)] = 0.0
                d2_mat = sp.diags(d2_inv)

                # D^-0.5 * A * D^-0.5
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                norm2_adj = d_mat.dot(adj2_mat)
                norm2_adj = norm2_adj.dot(d2_mat)
                norm2_adj = norm2_adj.tocsr()
                print(f"costing {time() - start}s, saved norm_mat...")
                sp.save_npz(
                    f"./data/preprocessed/{self.src}/interaction_adj1_mat__{decay_tag}.npz",
                    norm_adj,
                )
                sp.save_npz(
                    f"./data/preprocessed/{self.src}/interaction_adj2_mat__{decay_tag}.npz",
                    norm2_adj,
                )

            self.interactionGraph = _convert_sp_mat_to_sp_tensor(norm_adj)
            self.interactionGraph = self.interactionGraph.coalesce().to(world.device)

            self.interactionGraph2 = _convert_sp_mat_to_sp_tensor(norm2_adj)
            self.interactionGraph2 = self.interactionGraph2.coalesce().to(world.device)
        return self.interactionGraph, self.interactionGraph2

    def _getInteractionDic(self):
        user_interaction = {}
        item_interaction = {}

        def getDict(_set):
            for i in range(len(_set)):
                user = _set["user"][i]
                item = _set["item"][i]
                if user_interaction.get(user):
                    user_interaction[user].append(item)
                else:
                    user_interaction[user] = [item]
                if item_interaction.get(item):
                    item_interaction[item].append(user)
                else:
                    item_interaction[item] = [user]

        getDict(self.train_set)
        getDict(self.test_set)
        return user_interaction, item_interaction


class SocialGraphDataset(GraphDataset):
    def __init__(self, src):
        super(SocialGraphDataset, self).__init__(src)

        self.friendNet = pd.read_csv(f"./data/preprocessed/{src}/trust.txt")

        self.socialNet = csr_matrix(
            (
                np.ones(len(self.friendNet)),
                (self.friendNet["user"], self.friendNet["friend"]),
            ),
            shape=(self.n_user, self.n_user),
        )

        self.interactionGraph = None
        self.socialGraph1 = None
        self.socialGraph2 = None
        self.socialGraph3 = None

        print(f"Number of Links: {len(self.friendNet)}")
        print(
            f"{world.dataset} Link Density: {len(self.friendNet) / self.n_user / self.n_user}"
        )

    def getSocialGraph1(self):
        if self.socialGraph1 is None:
            try:
                pre_adj_mat = sp.load_npz(
                    f"./data/preprocessed/{self.src}/social1_adj_mat.npz"
                )
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except IOError:
                print("generating adjacency matrix1")
                start = time()
                R = self.UserItemNet.tolil()

                rowsum = np.array(R.sum(axis=1))
                print(
                    "The number of user's average preference items:",
                    sum(rowsum) / len(rowsum),
                )
                d_inv = np.power(rowsum, -0.5).flatten()
                # d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                adj_mat = self.socialNet.tolil()
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)

                # user_mean = np.array(adj_mat.sum(axis=1))

                value = []
                kk = 0
                edge = 0
                tmp_R = np.array(R.todense())
                for i, j in zip(self.friendNet["user"], self.friendNet["friend"]):
                    tt = tmp_R[i] + tmp_R[j]
                    # print('####################################', tt.shape)
                    tmp_degree = sum(tt == 2)
                    kk += tmp_degree
                    edge += 1
                    # value.append(tmp_degree)
                    value.append(tmp_degree)
                    # if tmp_degree == 0:
                    #     user_mean[i] = user_mean[i] - 0.5
                    #     user_mean[j] = user_mean[j] - 0.5
                avg_kk = kk / edge

                print("avg_kk:", avg_kk)
                # user_mean = np.power(user_mean, -1).flatten()
                # user_mean[np.isinf(user_mean)] = 0.
                # user_mean = sp.diags(user_mean)

                # norm_adj = user_mean.dot(norm_adj)
                simNet = csr_matrix(
                    (
                        np.array(value),
                        (self.friendNet["user"], self.friendNet["friend"]),
                    ),
                    shape=(self.n_user, self.n_user),
                ).tolil()

                norm_adj = norm_adj.multiply(simNet)
                norm_adj = norm_adj + sp.eye(norm_adj.shape[0])
                norm_adj = norm_adj.tocsr()

                # adj_mat = self.socialNet.tolil()
                # rowsum = np.array(adj_mat.sum(axis=1))
                # d_inv = np.power(rowsum, -0.5).flatten()
                # d_inv[np.isinf(d_inv)] = 0.
                # d_mat = sp.diags(d_inv)
                #
                # # D^-0.5 * A * D^-0.5
                # norm_adj = d_mat.dot(adj_mat)
                # norm_adj = norm_adj.dot(d_mat)
                # norm_adj = norm_adj.tocsr()
                print(f"costing {time() - start}s, saved norm_mat...")
                sp.save_npz(
                    f"./data/preprocessed/{self.src}/social1_adj_mat.npz", norm_adj
                )

            self.socialGraph1 = _convert_sp_mat_to_sp_tensor(norm_adj)
            self.socialGraph1 = self.socialGraph1.coalesce().to(world.device)
        return self.socialGraph1

    def getSocialGraph2(self):
        if self.socialGraph2 is None:
            try:
                pre_adj_mat = sp.load_npz(
                    f"./data/preprocessed/{self.src}/social2_adj_mat.npz"
                )
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except IOError:
                print("generating adjacency matrix2")
                start = time()
                adj_mat = self.socialNet.tolil()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                print(f"costing {time() - start}s, saved norm_mat...")
                sp.save_npz(
                    f"./data/preprocessed/{self.src}/social2_adj_mat.npz", norm_adj
                )
            self.socialGraph2 = _convert_sp_mat_to_sp_tensor(norm_adj)
            self.socialGraph2 = self.socialGraph2.coalesce().to(world.device)
        return self.socialGraph2

    def getDenseSocialGraph(self):
        if self.socialGraph1 is None:
            self.socialGraph1 = self.getSocialGraph1().to_dense()
            self.socialGraph2 = self.getSocialGraph2().to_dense()
            # self.socialGraph3 = self.getSocialGraph3().to_dense()
        else:
            pass
        return self.socialGraph1, self.socialGraph2  # , self.socialGraph3
