"""Trajectory + OC-SVM Postprocessor for OpenOOD.

Extracts 77-dim trajectory features from ResNet18 BasicBlock outputs,
combines with Mahalanobis distance and Energy score to form 79-dim features,
then uses One-Class SVM for OOD detection.

Designed for ResNet18_32x32 with 9 sampling points:
  - Point 0: after conv1 + bn1 + relu  (64 channels)
  - Points 1-8: after each BasicBlock  (64,64,128,128,256,256,512,512)
"""

from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import OneClassSVM
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


# ============================================================
#  Hook manager for ResNet18 BasicBlocks
# ============================================================
class ResNetBlockHook:
    """Register forward hooks on conv1+bn1+relu and each BasicBlock
    to capture 9 intermediate feature maps for trajectory computation.

    Expected ResNet18_32x32 structure:
      net.conv1 -> net.bn1 -> relu   (point 0)
      net.layer1[0], net.layer1[1]   (points 1, 2)
      net.layer2[0], net.layer2[1]   (points 3, 4)
      net.layer3[0], net.layer3[1]   (points 5, 6)
      net.layer4[0], net.layer4[1]   (points 7, 8)
    """

    def __init__(self, net):
        self.features = {}  # {point_index: tensor}
        self.handles = []
        self._register(net)

    def _register(self, net):
        # Point 0: after bn1 (relu is applied in forward, we hook bn1 output)
        # We hook bn1 and the feature will be pre-relu;
        # we apply relu ourselves in get_features.
        self.handles.append(
            net.bn1.register_forward_hook(self._make_hook(0))
        )

        # Points 1-8: each BasicBlock in layer1..layer4
        point_idx = 1
        for layer in [net.layer1, net.layer2, net.layer3, net.layer4]:
            for block in layer:
                self.handles.append(
                    block.register_forward_hook(self._make_hook(point_idx))
                )
                point_idx += 1

    def _make_hook(self, point_idx):
        def hook_fn(module, input, output):
            self.features[point_idx] = output.detach()
        return hook_fn

    def get_features(self):
        """Return list of 9 feature maps [B, C, H, W].
        Point 0 has relu applied to match FeatureCapture behavior."""
        result = []
        for i in range(9):
            feat = self.features[i]
            if i == 0:
                feat = F.relu(feat)  # bn1 output needs relu
            result.append(feat)
        return result

    def clear(self):
        self.features = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ============================================================
#  77-dim trajectory feature computation
# ============================================================
def _gap(feat_map):
    """Global average pooling: [B, C, H, W] -> [B, C]."""
    return F.adaptive_avg_pool2d(feat_map, 1).view(feat_map.size(0), -1)


def compute_trajectory_features(block_features):
    """Compute 77-dim trajectory features from 9 sampling points.

    9 points, channels: [64, 64, 64, 128, 128, 256, 256, 512, 512]

    Group 1 (25d): 8 cos_sim + 8 norm_ratio + 9 sparsity
    Group 2 (27d): 9 act_mean + 9 act_std + 9 act_kurtosis
    Group 3 (17d): 8 drift_dir + 8 drift_mag + 1 drift_ratio
    Group 4 (8d):  4 direction_curve_stats + 4 norm_curve_stats
    Total: 77 dimensions

    Args:
        block_features: list of 9 tensors [B, C, H, W]

    Returns:
        [B, 77] trajectory feature tensor
    """
    # GAP each feature map
    gap_feats = [_gap(f) for f in block_features]  # list of [B, C_i]

    B = gap_feats[0].shape[0]
    device = gap_feats[0].device
    num_points = len(gap_feats)  # 9

    # Project all to common dim (512) for trajectory comparison
    projected = []
    for feat in gap_feats:
        if feat.shape[1] < 512:
            pad = torch.zeros(B, 512 - feat.shape[1], device=device)
            projected.append(torch.cat([feat, pad], dim=1))
        else:
            projected.append(feat)
    f = torch.stack(projected, dim=1)  # [B, 9, 512]

    # ---- Group 1: Basic trajectory (25d) ----
    dir_change = []
    for i in range(num_points - 1):
        cos = F.cosine_similarity(f[:, i, :], f[:, i + 1, :], dim=-1)
        dir_change.append(cos)
    dir_change = torch.stack(dir_change, dim=1)  # [B, 8]

    norms = torch.norm(f, dim=-1)  # [B, 9]
    norm_ratio = norms[:, 1:] / (norms[:, :-1] + 1e-8)  # [B, 8]

    sparsity = []
    for feat in gap_feats:
        sp = (feat.abs() < 0.01).float().mean(dim=-1)
        sparsity.append(sp)
    sparsity = torch.stack(sparsity, dim=1)  # [B, 9]

    basic_feat = torch.cat([dir_change, norm_ratio, sparsity], dim=1)  # [B, 25]

    # ---- Group 2: Activation statistics (27d) ----
    act_mean_list, act_std_list, act_kurt_list = [], [], []
    for feat in gap_feats:
        act_mean_list.append(feat.mean(dim=-1))
        act_std_list.append(feat.std(dim=-1))
        m = feat.mean(dim=-1, keepdim=True)
        diff = feat - m
        var = (diff ** 2).mean(dim=-1, keepdim=True) + 1e-8
        kurt = ((diff ** 4).mean(dim=-1, keepdim=True) / (var ** 2)) - 3.0
        act_kurt_list.append(kurt.squeeze(-1))

    act_feat = torch.cat([
        torch.stack(act_mean_list, dim=1),
        torch.stack(act_std_list, dim=1),
        torch.stack(act_kurt_list, dim=1),
    ], dim=1)  # [B, 27]

    # ---- Group 3: Relative trajectory (17d) ----
    f0 = f[:, 0, :]
    f0_norm = torch.norm(f0, dim=-1, keepdim=True) + 1e-8
    drift_dir, drift_mag = [], []
    for i in range(1, num_points):
        drift_dir.append(F.cosine_similarity(f[:, i, :], f0, dim=-1))
        drift_mag.append(
            torch.norm(f[:, i, :] - f0, dim=-1) / f0_norm.squeeze(-1)
        )
    drift_dir = torch.stack(drift_dir, dim=1)  # [B, 8]
    drift_mag = torch.stack(drift_mag, dim=1)  # [B, 8]
    mid_idx = num_points // 2 - 1
    drift_ratio = (
        drift_mag[:, -1] / (drift_mag[:, mid_idx] + 1e-8)
    ).unsqueeze(1)  # [B, 1]
    rel_feat = torch.cat([drift_dir, drift_mag, drift_ratio], dim=1)  # [B, 17]

    # ---- Group 4: Shape statistics (8d) ----
    dc_np = dir_change.cpu().numpy()
    nm_np = norms.cpu().numpy()
    shape_np = np.concatenate([
        dc_np.mean(axis=1, keepdims=True),
        dc_np.var(axis=1, keepdims=True),
        sp_stats.skew(dc_np, axis=1).reshape(-1, 1),
        sp_stats.kurtosis(dc_np, axis=1).reshape(-1, 1),
        nm_np.mean(axis=1, keepdims=True),
        nm_np.var(axis=1, keepdims=True),
        sp_stats.skew(nm_np, axis=1).reshape(-1, 1),
        sp_stats.kurtosis(nm_np, axis=1).reshape(-1, 1),
    ], axis=1)
    shape_feat = torch.tensor(
        shape_np, dtype=torch.float32, device=device
    )  # [B, 8]

    return torch.cat([basic_feat, act_feat, rel_feat, shape_feat], dim=1)  # [B, 77]


# ============================================================
#  Score functions
# ============================================================
def compute_maha_scores(traj_np, class_means_np, cov_inv_np):
    """Min Mahalanobis distance across classes. Higher = more OOD.

    Args:
        traj_np: [N, 77] numpy
        class_means_np: [K, 77] numpy
        cov_inv_np: [77, 77] numpy

    Returns:
        [N] numpy array of min Mahalanobis distances
    """
    n = traj_np.shape[0]
    k = class_means_np.shape[0]
    min_dist = np.full(n, np.inf)
    for c in range(k):
        diff = traj_np - class_means_np[c:c + 1]
        left = diff @ cov_inv_np
        dist = np.sqrt(np.sum(left * diff, axis=1).clip(min=0))
        min_dist = np.minimum(min_dist, dist)
    return min_dist


def compute_energy_scores(logits_np, T=1.0):
    """Negative energy score. Higher = more OOD.

    Args:
        logits_np: [N, num_classes] numpy
        T: temperature

    Returns:
        [N] numpy array
    """
    scaled = logits_np / T
    max_val = scaled.max(axis=1, keepdims=True)
    lse = T * (
        np.log(np.sum(np.exp(scaled - max_val), axis=1) + 1e-30)
        + max_val.squeeze(1)
    )
    return -lse  # higher = more OOD
    # return lse


# ============================================================
#  TrajOCSVM Postprocessor
# ============================================================
class TrajectoryPostprocessor(BasePostprocessor):
    """Trajectory + OC-SVM postprocessor for ResNet18 OOD detection.

    Pipeline:
      setup():
        1. Hook 9 BasicBlock outputs from ResNet18
        2. Extract 77-dim trajectory features for training set
        3. Compute class-conditional Maha distance + Energy score
        4. Concatenate into 79-dim feature vector
        5. StandardScaler normalization
        6. KMeans coreset (for efficiency)
        7. Train OC-SVM with RBF kernel

      postprocess():
        1. Forward pass triggers hooks
        2. Compute 77-dim trajectory + Maha + Energy -> 79-dim
        3. StandardScaler transform
        4. OC-SVM decision_function as confidence (positive = ID)
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False

        # Hyperparameters
        self.args = self.config.postprocessor
        self.nu = self.args.get('nu', 0.03)
        self.gamma = self.args.get('gamma', 'scale')
        self.coreset_size = self.args.get('coreset_size', 5000)
        self.energy_T = self.args.get('energy_T', 1.0)

        # Will be set during setup
        self.hook = None
        self.class_means_np = None
        self.cov_inv_np = None
        self.scaler = None
        self.ocsvm = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        net.eval()

        print('\n[TrajOCSVM] Starting setup...')

        # ------ Step 1: Register hooks ------
        print('[TrajOCSVM] Step 1: Registering hooks on BasicBlocks...')
        self.hook = ResNetBlockHook(net)

        # ------ Step 2: Extract training set features ------
        print('[TrajOCSVM] Step 2: Extracting training set trajectory features...')
        train_loader = id_loader_dict['train']

        all_traj = []
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(train_loader,
                              desc='[TrajOCSVM] Train extraction'):
                data = batch['data'].cuda()
                labels = batch['label']

                logits = net(data)
                block_feats = self.hook.get_features()  # 9 feature maps
                traj = compute_trajectory_features(block_feats)

                all_traj.append(traj.cpu())
                all_logits.append(logits.cpu())
                all_labels.append(deepcopy(labels))

        train_traj = torch.cat(all_traj, dim=0)      # [N, 77]
        train_logits = torch.cat(all_logits, dim=0)   # [N, num_classes]
        train_labels = torch.cat(all_labels, dim=0)   # [N]

        print(f'[TrajOCSVM]   Trajectory: {train_traj.shape}, '
              f'Logits: {train_logits.shape}')

        # ------ Step 3: Fit class-conditional Gaussian ------
        print('[TrajOCSVM] Step 3: Fitting class-conditional Gaussian...')
        D = train_traj.shape[1]  # 77
        class_means = torch.zeros(self.num_classes, D)
        centered_all = []
        for c in range(self.num_classes):
            mask = (train_labels == c)
            if mask.sum() == 0:
                continue
            fc = train_traj[mask]
            class_means[c] = fc.mean(dim=0)
            centered_all.append(fc - class_means[c])
        centered = torch.cat(centered_all, dim=0)
        cov = (centered.T @ centered) / (centered.shape[0] - 1)
        cov += 1e-5 * torch.eye(D)
        cov_inv = torch.linalg.inv(cov)

        self.class_means_np = class_means.numpy()
        self.cov_inv_np = cov_inv.numpy()
        print(f'[TrajOCSVM]   class_means: {class_means.shape}, '
              f'cov_inv: {cov_inv.shape}')

        # ------ Step 4: Compute 79-dim features ------
        print('[TrajOCSVM] Step 4: Computing 79-dim features (traj+maha+energy)...')
        train_traj_np = train_traj.numpy()
        train_logits_np = train_logits.numpy()

        train_maha = compute_maha_scores(
            train_traj_np, self.class_means_np, self.cov_inv_np
        )
        train_energy = compute_energy_scores(
            train_logits_np, T=self.energy_T
        )

        train_79d = np.concatenate([
            train_traj_np,
            train_maha.reshape(-1, 1),
            train_energy.reshape(-1, 1),
        ], axis=1)  # [N, 79]

        print(f'[TrajOCSVM]   79D features: {train_79d.shape}')
        print(f'[TrajOCSVM]   Maha: mean={train_maha.mean():.4f}, '
              f'std={train_maha.std():.4f}')
        print(f'[TrajOCSVM]   Energy: mean={train_energy.mean():.4f}, '
              f'std={train_energy.std():.4f}')

        # ------ Step 5: Normalize ------
        print('[TrajOCSVM] Step 5: Fitting StandardScaler...')
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(train_79d)

        # ------ Step 6: KMeans coreset ------
       
        if len(X_train) > self.coreset_size*2:
            self.coreset_size = min(self.coreset_size, len(X_train)//10)
            print(f'[TrajOCSVM] Step 6: Building KMeans coreset '
                  f'({len(X_train)} -> {self.coreset_size})...')
            kmeans = MiniBatchKMeans(
                n_clusters=self.coreset_size,
                random_state=42,
                batch_size=1024,
                n_init='auto'
            )
            kmeans.fit(X_train)
            X_fit = kmeans.cluster_centers_
        else:
            print('[TrajOCSVM] Step 6: Dataset small enough, skip coreset.')
            X_fit = X_train

        # ------ Step 7: Train OC-SVM ------
        print(f'[TrajOCSVM] Step 7: Training OC-SVM '
              f'(nu={self.nu}, gamma={self.gamma}, '
              f'n_samples={X_fit.shape[0]})...')
        self.ocsvm = OneClassSVM(
            kernel='rbf',
            gamma=self.gamma,
            nu=self.nu
        )
        self.ocsvm.fit(X_fit)

        n_sv = self.ocsvm.support_vectors_.shape[0]
        print(f'[TrajOCSVM]   Support vectors: {n_sv}')
        print('[TrajOCSVM] Setup complete.\n')

        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """Score a single batch.

        Args:
            net: ResNet18_32x32 model
            data: input tensor [B, C, H, W] on CUDA

        Returns:
            pred: [B] predicted class indices (torch, on device)
            conf: [B] OOD confidence scores (torch, on device)
                  Higher = more likely ID.
        """
        # Forward pass (triggers hooks)
        logits = net(data)
        pred = logits.argmax(dim=1)

        # Get block features and compute trajectory
        block_feats = self.hook.get_features()
        traj = compute_trajectory_features(block_feats)  # [B, 77]

        # Compute Maha and Energy
        traj_np = traj.cpu().numpy()
        logits_np = logits.cpu().numpy()

        maha = compute_maha_scores(
            traj_np, self.class_means_np, self.cov_inv_np
        )
        energy = compute_energy_scores(logits_np, T=self.energy_T)

        # Build 79-dim feature
        feat_79d = np.concatenate([
            traj_np,
            maha.reshape(-1, 1),
            energy.reshape(-1, 1),
        ], axis=1)

        # Normalize and score
        X = self.scaler.transform(feat_79d)
        # decision_function: positive = inside (ID), negative = outside (OOD)
        conf_np = self.ocsvm.decision_function(X)  # [B]

        conf = torch.from_numpy(conf_np.astype(np.float32)).to(data.device)

        return pred, conf
    
    
    def __getstate__(self):
        """pickle序列化时移除不可序列化的hook。"""
        state = self.__dict__.copy()
        state['hook'] = None  # hook含闭包，不可pickle
        return state    

    def __setstate__(self, state):
        """反序列化时恢复，hook需要重新setup。"""
        self.__dict__.update(state)
        self.hook = None
        # 标记需要重新注册hook（但ocsvm/scaler等已保存）