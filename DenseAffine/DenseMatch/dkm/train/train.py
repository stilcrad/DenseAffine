from tqdm import tqdm
from DenseMatch.dkm.utils.utils import to_cuda
import torch
import numpy as np
import cv2
def expand_homo_ones(arr2d, axis=1):
    """Raise 2D array to homogenous coordinates
    Args:
        - arr2d: (N, 2) or (2, N)
        - axis: the axis to append the ones
    """
    if axis == 0:
        ones = torch.ones((1, arr2d.shape[1]), dtype=arr2d.dtype, device=arr2d.device)
    else:
        ones = torch.ones((arr2d.shape[0], 1), dtype=arr2d.dtype, device=arr2d.device)
    expanded_arr = torch.cat([arr2d, ones], dim=axis)
    return expanded_arr
    # if axis == 0:
    #     ones = np.ones((1, arr2d.shape[1]))
    # else:
    #     ones = np.ones((arr2d.shape[0], 1))
    # return np.concatenate([arr2d, ones], axis=axis)




def sampson_distance(pts1, pts2, homos=True, eps=1e-8):
    """Calculate symmetric epipolar distance between 2 sets of points
    Args:
        - pts1, pts2: points correspondences in the two images,
          each has shape of (num_points, 2)
        - F: fundamental matrix that fulfills x2^T*F*x1=0,
          where x1 and x2 are the correspondence points in the 1st and 2nd image
    Return:
        A vector of (num_points,), containing root-squared epipolar distances

    """
    pt1 = np.array(pts1.detach().cpu())
    pt2 = np.array(pts2.detach().cpu())

    F, mask = cv2.findFundamentalMat(pt1, pt2, method=cv2.USAC_ACCURATE,ransacReprojThreshold=0.25, confidence=0.999,maxIters=120000)
    # Homogenous coordinates
    F = torch.tensor(F).cuda()
    if homos:
        pts1 = expand_homo_ones(pts1, axis=1)  # if pts1.shape[1] == 2 else pts1
        pts2 = expand_homo_ones(pts2, axis=1)  # if pts2.shape[1] == 2 else pts2

    # eg
    # l2=F*x1, l1=F^T*x2
    pts1 = pts1.type(torch.DoubleTensor).cuda()
    pts2 = pts2.type(torch.DoubleTensor).cuda()

    l2 = torch.matmul(F, pts1.t())  # 3,N
    l1 = torch.matmul(F.t(), pts2.t())
    dd = torch.sum(l2.t() * pts2, dim=1)  # Distance from pts2 to l2
    l1_sq_norm = l1[0, :] ** 2 + l1[1, :] ** 2
    l2_sq_norm = l2[0, :] ** 2 + l2[1, :] ** 2
    epipolar_sampson_loss = dd ** 2 / (eps + l1_sq_norm + l2_sq_norm)

    # l2 = np.dot(F, pt1.T)  # 3,N
    # l1 = np.dot(F.T, pt2.T)
    # dd = np.sum(l2.T * pt2, 1)  # Distance from pts2 to l2
    # epipolar_sampson_loss = dd ** 2 / (eps + l1[0, :] ** 2 + l1[1, :] ** 2 + l2[0, :] ** 2 + l2[1, :] ** 2)

    return epipolar_sampson_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_step(train_batch, model, objective1, objective2 , optimizer, **kwargs):
    optimizer.zero_grad()
    out = model(train_batch)
    warp, certainty = model.match(train_batch["query"][0], train_batch["support"][0], device=device,train = True)
    # Sample matches for estimation
    matches, certainty = model.sample(warp, certainty)
    kpts1, kpts2 = model.to_pixel_coordinates(matches, train_batch["query"].shape[1], train_batch["query"].shape[2], train_batch["support"].shape[1], train_batch["support"].shape[2])

    l1= objective1(out, train_batch)
    l2 = sampson_distance(kpts1,kpts2)
    l2 = torch.tensor(l2).mean()
    print(l1,l2)
    l = l1
    # l = l1 + l2
    l.backward()
    optimizer.step()
    return {"train_out": out, "train_loss": l.item()}


def train_k_steps(
    n_0, k, dataloader, model, objective1,objective2, optimizer, lr_scheduler, progress_bar=True
):
    for n in tqdm(range(n_0, n_0 + k), disable=not progress_bar):
        batch = next(dataloader)
        model.train(True)
        batch = to_cuda(batch)
        try:
            train_step(
                train_batch=batch,
                model=model,
                objective1=objective1,
                objective2=objective2,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                n=n,
            )
        except cv2.error :
            continue
        lr_scheduler.step()


def train_epoch(
    dataloader=None,
    model=None,
    objective=None,
    optimizer=None,
    lr_scheduler=None,
    epoch=None,
):
    model.train(True)
    print(f"At epoch {epoch}")
    for batch in tqdm(dataloader, mininterval=5.0):
        batch = to_cuda(batch)
        train_step(
            train_batch=batch, model=model, objective=objective, optimizer=optimizer
        )
    lr_scheduler.step()
    return {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "epoch": epoch,
    }


def train_k_epochs(
    start_epoch, end_epoch, dataloader, model, objective, optimizer, lr_scheduler
):
    for epoch in range(start_epoch, end_epoch + 1):
        train_epoch(
            dataloader=dataloader,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
        )
