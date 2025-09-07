import numpy as np
import torch

import utils
import world


def BPR_train_original(dataset, recommend_model, loss_class, epoch):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    allusers = list(range(dataset.n_users))
    S, sam_time = utils.UniformSample_original(
        allusers, dataset
    )  # [user,pos,neg], [times list]
    # print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config["bpr_batch_size"] + 1
    aver_loss = 0.0
    for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(
        utils.minibatch(
            users, posItems, negItems, batch_size=world.config["bpr_batch_size"]
        )
    ):

        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, epoch)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    return aver_loss


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:  # [10, 20]
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {
        "recall": np.array(recall),
        "precision": np.array(pre),
        "ndcg": np.array(ndcg),
    }


def Test(dataset, Recmodel, epoch, cold=False, w=None):
    # ensure integer batch size (your parse.py currently uses type=str)
    try:
        u_batch_size = int(world.config["test_u_batch_size"])
    except Exception:
        u_batch_size = 100

    # recency-aware ground truth
    recency_months = world.config.get("recency_months", 0)
    testDict: dict = dataset.get_eval_dict(recency_months=recency_months, cold=cold)

    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
    }

    users = list(testDict.keys())
    if len(users) == 0:
        print(f"[Eval] recency_months={recency_months}, users=0 -> no eval (skip).")
        return results

    # keep batch size reasonable
    max_reasonable = max(1, len(users) // 10)
    if u_batch_size > max_reasonable:
        print(
            f"test_u_batch_size ({u_batch_size}) too big; shrinking to {max_reasonable}."
        )
        u_batch_size = max_reasonable

    with torch.no_grad():
        users_list = []
        rating_list = []
        groundTrue_list = []

        # total_batch = len(users) // u_batch_size + 1
        total_batch = (len(users) + u_batch_size - 1) // u_batch_size
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]

            batch_users_gpu = torch.tensor(
                batch_users, dtype=torch.long, device=world.device
            )
            rating = Recmodel.getUsersRating(batch_users_gpu)

            # exclude training positives
            exclude_index, exclude_items = [], []
            for range_i, items in enumerate(allPos):
                if len(items) == 0:
                    continue
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            if len(exclude_index) > 0:
                rating[exclude_index, exclude_items] = -(1 << 10)

            _, rating_K = torch.topk(rating, k=max_K)
            del rating

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        assert total_batch == len(users_list)

        if total_batch != len(users_list):
            print(
                f"[WARN] expected {total_batch} batches, got {len(users_list)} "
                f"(users={len(users)}, u_batch_size={u_batch_size})"
            )
        pre_results = [test_one_batch(x) for x in zip(rating_list, groundTrue_list)]

        for r in pre_results:
            results["recall"] += r["recall"]
            results["precision"] += r["precision"]
            results["ndcg"] += r["ndcg"]

        denom = float(len(users))
        results["recall"] /= denom
        results["precision"] /= denom
        results["ndcg"] /= denom

        print(results)
        print(f"[Eval] recency_months={recency_months}, users={len(users)}")
        return results
