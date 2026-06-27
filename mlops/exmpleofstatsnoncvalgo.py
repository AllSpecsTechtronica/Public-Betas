# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Model_Name: Polymorph_Enhanced_alpha_23
# ------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob  # For listing checkpoint files

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For synchronous CUDA debugging

print("running...", flush=True)

# --------------------------------------------------
# 1. Configuration, Print Statements, and Random Seeds
# --------------------------------------------------
# Removed MODEL_SAVE_PATH; instead we use CHECKPOINT_DIR below.
CHECKPOINT_DIR = "/content/drive/My Drive/BCI/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SPECIALIST_MODELS_DIR = "/content/drive/My Drive/BCI/specialist_models/"
os.makedirs(SPECIALIST_MODELS_DIR, exist_ok=True)

DATA_FILE = "/content/drive/My Drive/BCI/BCI_HostDataPhonemes.csv"

MAX_EPOCHS = 1100  # Longer training time
MAX_PATIENCE = 150  # More patience
IMPROVEMENT_THRESHOLD = 0.001  # Smaller improvement threshold

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if not torch.cuda.is_available():
    raise RuntimeError("GPU is not available. Please run on a CUDA-enabled GPU.")
else:
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0")
print(f"Using device: {device}", flush=True)


# --------------------------------------------------
# 1.1 Helper: Load Checkpoint Prompt
# --------------------------------------------------
def load_checkpoint_prompt():
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth")))
    if len(checkpoint_files) == 0:
        print("no checkpoints found")
        cont = input("Type 'continue' to continue: ")
        return None
    print("Available checkpoints:")
    for i, f in enumerate(checkpoint_files, 1):
        base = os.path.basename(f)
        # Extract epoch number from filename (expected format: checkpoint_epoch_{epoch}.pth)
        epoch_str = base.replace("checkpoint_epoch_", "").replace(".pth", "")
        print(f"{i}. {base} - saved at epoch {epoch_str}")
    choice = input("Would you like to load from a checkpoint? (yes/no): ")
    if choice.lower().startswith('y'):
        try:
            idx = int(input("Enter the checkpoint number to load: "))
            if 1 <= idx <= len(checkpoint_files):
                return checkpoint_files[idx - 1]
            else:
                print("Invalid selection. Continuing without loading checkpoint.")
                return None
        except ValueError:
            print("Invalid input. Continuing without loading checkpoint.")
            return None
    else:
        return None


# --------------------------------------------------
# 2. Enhanced Hyperparameter Manager
# --------------------------------------------------
class HyperparameterManager:
    def __init__(self):
        # Original parameters
        self.window_size = 1024
        self.step_size = 256
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.dropout_rate = 0.5
        self.d_model = 256
        self.num_heads = 8
        self.num_layers = 6
        self.dim_feedforward = 1024

        # Enhanced parameters
        self.weight_decay = 5e-4
        self.mixup_alpha = 0.2
        self.label_smoothing = 0.1
        self.scheduler_t0 = 25
        self.use_mixup = True
        self.accumulation_steps = 4
        self.focal_loss_gamma = 2.0

    def get_params(self):
        return {
            'window_size': self.window_size,
            'step_size': self.step_size,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'weight_decay': self.weight_decay,
            'mixup_alpha': self.mixup_alpha,
            'label_smoothing': self.label_smoothing,
            'scheduler_t0': self.scheduler_t0,
            'use_mixup': self.use_mixup,
            'accumulation_steps': self.accumulation_steps,
            'focal_loss_gamma': self.focal_loss_gamma
        }


# --------------------------------------------------
# 3. Global Filters for Preprocessing
# --------------------------------------------------

NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
SAMPLING_RATE = 512

b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, NOTCH_Q, fs=SAMPLING_RATE)

BANDPASS_ORDER = 4
LOWCUT = 0.5
HIGHCUT = 40.0
nyquist = 0.5 * SAMPLING_RATE
low = LOWCUT / nyquist
high = HIGHCUT / nyquist
b_bandpass, a_bandpass = signal.butter(BANDPASS_ORDER, [low, high], btype='band')


# --------------------------------------------------
# 4. Enhanced Helper Functions for Preprocessing
# --------------------------------------------------

def bandpower(data, sf, band):
    band = np.asarray(band)
    fnq = sf / 2.
    lowcut = band[0] / fnq
    highcut = band[1] / fnq
    b, a = signal.butter(3, [lowcut, highcut], btype='band')
    filtered = signal.filtfilt(b, a, data)
    return np.sum(filtered ** 2) / len(filtered)


def add_spectral_features(data, sf=512):
    """Extract richer spectral features"""
    freqs, psd = signal.welch(data, sf, nperseg=256)
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha_low': (8, 10),
        'alpha_high': (10, 13),
        'beta_low': (13, 18),
        'beta_mid': (18, 25),
        'beta_high': (25, 30),
        'gamma_low': (30, 40),
        'gamma_high': (40, 45)
    }
    features = {}
    for band, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        features[band + '_mean'] = np.mean(psd[idx])
        features[band + '_std'] = np.std(psd[idx])
        features[band + '_max'] = np.max(psd[idx])
    features['theta_beta_ratio'] = features['theta_mean'] / features['beta_mid_mean']
    features['alpha_theta_ratio'] = features['alpha_high_mean'] / features['theta_mean']
    features['hjorth_mobility'] = stats.moment(np.diff(data), 2) / stats.moment(data, 2)
    features['hjorth_complexity'] = (stats.moment(np.diff(np.diff(data)), 2) /
                                     stats.moment(np.diff(data), 2)) / features['hjorth_mobility']
    return np.array(list(features.values()), dtype=np.float32)


def extract_connectivity_features(data, sf=512):
    """Extract cross-frequency coupling and phase synchrony features"""
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta_low': (13, 20),
        'beta_high': (20, 30),
        'gamma': (30, 45)
    }

    filtered_signals = {}
    for band_name, (low, high) in bands.items():
        nyq = sf / 2
        low_norm = low / nyq
        high_norm = high / nyq
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        filtered_signals[band_name] = signal.filtfilt(b, a, data)

    features = {}

    band_names = list(bands.keys())
    for i in range(len(band_names)):
        for j in range(i + 1, len(band_names)):
            band1 = band_names[i]
            band2 = band_names[j]
            env1 = np.abs(signal.hilbert(filtered_signals[band1]))
            env2 = np.abs(signal.hilbert(filtered_signals[band2]))
            features[f'{band1}_{band2}_env_corr'] = np.corrcoef(env1, env2)[0, 1]

    for phase_band in ['delta', 'theta', 'alpha']:
        for amp_band in ['beta_low', 'beta_high', 'gamma']:
            phase_signal = filtered_signals[phase_band]
            phase = np.angle(signal.hilbert(phase_signal))
            amp_signal = filtered_signals[amp_band]
            amplitude = np.abs(signal.hilbert(amp_signal))
            n_bins = 18  # 20-degree bins
            bin_means = np.zeros(n_bins)
            bin_width = 2 * np.pi / n_bins
            for bin_idx in range(n_bins):
                bin_start = -np.pi + bin_idx * bin_width
                bin_end = bin_start + bin_width
                bin_mask = np.logical_and(phase >= bin_start, phase < bin_end)
                if np.any(bin_mask):
                    bin_means[bin_idx] = np.mean(amplitude[bin_mask])
            if np.sum(bin_means) > 0:
                bin_means = bin_means / np.sum(bin_means)
                uniform = np.ones(n_bins) / n_bins
                features[f'{phase_band}_{amp_band}_mi'] = np.sum(bin_means * np.log(bin_means / uniform + 1e-10))

    return np.array(list(features.values()), dtype=np.float32)


def preprocess_data(data, sf=512):
    if np.std(data) < 1e-6:
        print("Warning: Very low signal variance detected", flush=True)
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).ravel()
    data = signal.filtfilt(b_notch, a_notch, data)
    data = signal.filtfilt(b_bandpass, a_bandpass, data)

    delta = bandpower(data, sf, [0.5, 4])
    theta = bandpower(data, sf, [4, 8])
    alpha_low = bandpower(data, sf, [8, 10])
    alpha_high = bandpower(data, sf, [10, 13])
    beta_low = bandpower(data, sf, [13, 20])
    beta_high = bandpower(data, sf, [20, 30])
    gamma = bandpower(data, sf, [30, 45])
    basic_features = np.array([delta, theta, alpha_low, alpha_high, beta_low, beta_high, gamma], dtype=np.float32)

    spectral_features = add_spectral_features(data, sf)
    connectivity_features = extract_connectivity_features(data, sf)
    combined_enhanced_features = np.concatenate([spectral_features, connectivity_features])

    return data, basic_features, combined_enhanced_features


# --------------------------------------------------
# 5. Enhanced Data Preparation
# --------------------------------------------------
def prepare_data(df, params):
    print(f"Input DataFrame shape: {df.shape}", flush=True)  # Should be ~300000 rows

    # Clean and encode
    if 'Timestamp' in df.columns:
        df.drop(columns=['Timestamp'], inplace=True)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    df['label'] = df['Label'].str.strip().str.lower()
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])

    # Convert numeric columns
    df['EEG_Value'] = pd.to_numeric(df['EEG_Value'], errors='coerce')
    freq_cols = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Alpha/Theta', 'Beta/Theta']
    for col in freq_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['EEG_Value', 'label'] + freq_cols, inplace=True)
    print(f"After cleaning: {df.shape}", flush=True)  # Should still be ~300000

    # Verify 300k samples
    expected_samples = 300000
    assert df.shape[0] >= expected_samples, f"Expected {expected_samples} samples, got {df.shape[0]}"
    if df.shape[0] > expected_samples:
        df = df.sample(n=expected_samples, random_state=RANDOM_SEED)
        print(f"Subsampled to {expected_samples} rows", flush=True)

    # Prepare X (single value per sample)
    X = df['EEG_Value'].values.reshape(-1, 1, 1)  # Shape: (300000, 1, 1)

    # Basic features from CSV
    basic_features = df[freq_cols].values.astype(np.float32)  # Shape: (300000, 7)

    # Enhanced features: Derive from basic features instead of EEG_Value
    print("Computing enhanced features from frequency bands...", flush=True)
    enhanced_features = np.zeros((df.shape[0], 12), dtype=np.float32)  # Example: 12 new features
    for i in tqdm(range(df.shape[0])):
        freqs = basic_features[i, :5]  # Delta, Theta, Alpha, Beta, Gamma
        ratios = basic_features[i, 5:7]  # Alpha/Theta, Beta/Theta

        # Statistical features
        enhanced_features[i, 0] = np.mean(freqs)
        enhanced_features[i, 1] = np.std(freqs) if np.std(freqs) > 1e-6 else 0.0
        enhanced_features[i, 2] = np.max(freqs)
        enhanced_features[i, 3] = np.min(freqs)

        # Additional ratios with safe division
        enhanced_features[i, 4] = freqs[2] / (freqs[1] + 1e-10)  # Alpha/Theta
        enhanced_features[i, 5] = freqs[3] / (freqs[1] + 1e-10)  # Beta/Theta
        enhanced_features[i, 6] = freqs[4] / (freqs[3] + 1e-10)  # Gamma/Beta

        # Log transforms (avoid log(0))
        enhanced_features[i, 7] = np.log(freqs[0] + 1e-10)  # Log Delta
        enhanced_features[i, 8] = np.log(freqs[1] + 1e-10)  # Log Theta
        enhanced_features[i, 9] = np.log(freqs[2] + 1e-10)  # Log Alpha
        enhanced_features[i, 10] = np.log(freqs[3] + 1e-10)  # Log Beta
        enhanced_features[i, 11] = np.log(freqs[4] + 1e-10)  # Log Gamma

    y = df['label_encoded'].values.astype(np.int64)  # Shape: (300000,)

    # Replace NaNs with 0
    enhanced_features = np.nan_to_num(enhanced_features, nan=0.0)

    print(f"Data prepared: X={X.shape}, basic={basic_features.shape}, "
          f"enhanced={enhanced_features.shape}, y={y.shape}", flush=True)

    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    print("Unique labels:", unique_labels)
    print("Expected range: 0 to", num_classes - 1)
    assert unique_labels.min() >= 0, "Found label below 0!"
    assert unique_labels.max() < num_classes, "Label values exceed number of classes!"

    return X, basic_features, enhanced_features, y, label_encoder


# --------------------------------------------------
# 6. Focal Loss Implementation
# --------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        if self.label_smoothing > 0:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            smoothed_targets = (1.0 - self.label_smoothing) * targets_one_hot + self.label_smoothing / num_classes
            log_probs = F.log_softmax(inputs, dim=1)
            loss = -(smoothed_targets * log_probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
            loss = ce_loss
        pt = torch.exp(-loss)
        focal_loss = (1 - pt) ** self.gamma * loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# --------------------------------------------------
# 7. Rotary Positional Encoding and Global Context Block
# --------------------------------------------------
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        # Compute inverse frequency for half the dimension
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))
        self.max_seq_len = 2048

    def forward(self, x):
        seq_len = x.shape[1]
        device = x.device
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        # Compute frequencies for half the dimension
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # shape: (seq_len, dim/2)
        cos_pos = freqs.cos().unsqueeze(0)  # shape: (1, seq_len, dim/2)
        sin_pos = freqs.sin().unsqueeze(0)  # shape: (1, seq_len, dim/2)
        x1, x2 = x.chunk(2, dim=-1)  # each of shape: (batch, seq_len, dim/2)
        x_out = torch.cat([x1 * cos_pos - x2 * sin_pos, x2 * cos_pos + x1 * sin_pos], dim=-1)
        return x_out


class GlobalContextBlock(nn.Module):
    """Global context attention for capturing document-level patterns"""

    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Softmax(dim=1)
        )
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

    def forward(self, x):
        attn_weights = self.attn(x)
        global_context = torch.sum(x * attn_weights, dim=1, keepdim=True)
        return x + self.proj(global_context)


# --------------------------------------------------
# 8. Stochastic Depth and Enhanced Residual Block
# --------------------------------------------------
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (torch.rand(shape, dtype=x.dtype, device=x.device) >= self.drop_prob).float()
        return x * random_tensor / keep_prob


class EnhancedResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, drop_prob=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop_path = StochasticDepth(drop_prob)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.drop_path(x)
        x += residual
        return F.relu(x)


# --------------------------------------------------
# 9. Enhanced Datasets with Adaptive Augmentation
# --------------------------------------------------
class AdaptiveEEGDataset(Dataset):
    def __init__(self, X, basic_features, enhanced_features, y,
                 confusion_matrix=None, underperforming_classes=None,
                 augment=True, augment_factor=2):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.basic_features = torch.tensor(basic_features, dtype=torch.float32)
        self.enhanced_features = torch.tensor(enhanced_features, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.augment_factor = augment_factor

        self.underperforming_classes = set(underperforming_classes or [])
        self.confusion_pairs = []
        if confusion_matrix is not None:
            for i in range(len(confusion_matrix)):
                for j in range(len(confusion_matrix)):
                    if i != j and confusion_matrix[i, j] > 0.15 * np.sum(confusion_matrix[i, :]):
                        self.confusion_pairs.append((i, j))
                        print(f"Identified confusion between classes {i} and {j}", flush=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        basic_f = self.basic_features[idx].clone()
        enhanced_f = self.enhanced_features[idx].clone()
        label = self.y[idx]

        cls = label.item()
        is_underperforming = cls in self.underperforming_classes
        is_confused = any(cls in pair for pair in self.confusion_pairs)

        if self.augment:
            if np.random.rand() < 0.3:
                noise_level = 0.03 if is_underperforming or is_confused else 0.02
                noise = torch.randn_like(x) * noise_level
                x = x + noise

            if np.random.rand() < 0.3:
                shift_range = 80 if is_underperforming or is_confused else 50
                shift = np.random.randint(-shift_range, shift_range)
                x = torch.roll(x, shift, dims=1)

            if np.random.rand() < 0.3:
                scale_range = (0.7, 1.3) if is_underperforming or is_confused else (0.8, 1.2)
                scale = np.random.uniform(*scale_range)
                x = x * scale

            if is_underperforming and np.random.rand() < 0.4:
                mask_prob = 0.9 if is_underperforming else 0.95
                mask = torch.bernoulli(torch.ones_like(x) * mask_prob)
                x = x * mask

            if is_confused and np.random.rand() < 0.4:
                relevant_pairs = [pair for pair in self.confusion_pairs if cls in pair]
                if relevant_pairs and np.random.rand() < 0.6:
                    feature_mask = torch.ones_like(enhanced_f)
                    for _ in range(2):
                        feature_idx = np.random.randint(0, len(enhanced_f))
                        feature_mask[feature_idx] = np.random.uniform(1.1, 1.5)
                    enhanced_f = enhanced_f * feature_mask

            if np.random.rand() < 0.3:
                x_flat = x.squeeze(-1)
                freq_domain = torch.fft.rfft(x_flat)
                max_mask_width = min(25, freq_domain.shape[-1] // 2)
                if max_mask_width > 0:
                    mask_width = np.random.randint(1, max_mask_width)
                    if freq_domain.shape[-1] - mask_width > 0:
                        mask_start = np.random.randint(0, freq_domain.shape[-1] - mask_width)
                        freq_domain[mask_start:mask_start + mask_width] = 0
                        x = torch.fft.irfft(freq_domain, n=x_flat.shape[-1]).unsqueeze(-1)

            if np.random.rand() < 0.2:
                x_flat = x.squeeze(-1)
                freq_domain = torch.fft.rfft(x_flat)
                magnitude = torch.abs(freq_domain)
                phase = torch.rand_like(freq_domain) * 2 * np.pi
                new_freq_domain = magnitude * torch.exp(1j * phase)
                x = torch.fft.irfft(new_freq_domain, n=x_flat.shape[-1]).unsqueeze(-1)

        return x, basic_f, enhanced_f, label


# --------------------------------------------------
# 10. Mixup Augmentation
# --------------------------------------------------
def mixup_data(x, basic_feats, enhanced_feats, y, alpha=0.2):
    batch_size = x.size(0)
    indices = torch.randperm(batch_size).to(x.device)
    mixed_x = alpha * x + (1 - alpha) * x[indices]
    mixed_basic = alpha * basic_feats + (1 - alpha) * basic_feats[indices]
    mixed_enhanced = alpha * enhanced_feats + (1 - alpha) * enhanced_feats[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, mixed_basic, mixed_enhanced, y_a, y_b, alpha


def mixup_criterion(criterion, pred, y_a, y_b, alpha):
    return alpha * criterion(pred, y_a) + (1 - alpha) * criterion(pred, y_b)


# --------------------------------------------------
# 11. Enhanced Transformer Model with Rotary Encoding, Global Context, and Weight Initialization
# --------------------------------------------------
class EnhancedEEGTransformer(nn.Module):
    def __init__(self, params, num_basic_feats=7, num_enhanced_feats=12, num_classes=5):
        super().__init__()
        self.d_model = params['d_model']
        self.num_heads = params['num_heads']
        self.num_layers = params['num_layers']
        self.dim_feedforward = params['dim_feedforward']
        self.dropout_rate = params['dropout_rate']

        # Adjust CNN branches for single-sample input
        self.cnn_branch1 = nn.Sequential(
            nn.Conv1d(1, self.d_model // 4, kernel_size=1),
            nn.BatchNorm1d(self.d_model // 4),
            nn.ReLU()
        )
        self.cnn_branch2 = nn.Sequential(
            nn.Conv1d(1, self.d_model // 4, kernel_size=1),
            nn.BatchNorm1d(self.d_model // 4),
            nn.ReLU()
        )
        self.cnn_branch3 = nn.Sequential(
            nn.Conv1d(1, self.d_model // 4, kernel_size=1),
            nn.BatchNorm1d(self.d_model // 4),
            nn.ReLU()
        )
        self.cnn_branch4 = nn.Sequential(
            nn.Conv1d(1, self.d_model // 4, kernel_size=1),
            nn.BatchNorm1d(self.d_model // 4),
            nn.ReLU()
        )

        self.se_block = SEBlock(self.d_model)
        self.rotary_encoder = RotaryPositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.global_context = GlobalContextBlock(self.d_model)
        self.attention_pooling = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Softmax(dim=1)
        )

        self.basic_feat_proj = nn.Linear(num_basic_feats, self.d_model // 2)
        self.enhanced_feat_proj = nn.Linear(num_enhanced_feats, self.d_model // 2)

        self.feature_integration = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )

        self.expert1 = nn.Linear(self.d_model, num_classes)
        self.expert2 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, num_classes)
        )
        self.expert3 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.d_model * 2, num_classes)
        )
        self.gate = nn.Linear(self.d_model, 3)

        self._initialize_weights()  # Call the method

    def _initialize_weights(self):
        """Initialize weights for all layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, basic_feats, enhanced_feats=None):
        x = x.transpose(1, 2)  # Shape: (batch, 1, 1) -> (batch, 1, 1)

        x1 = self.cnn_branch1(x)  # Shape: (batch, d_model//4, 1)
        x2 = self.cnn_branch2(x)
        x3 = self.cnn_branch3(x)
        x4 = self.cnn_branch4(x)

        x_concat = torch.cat([x1, x2, x3, x4], dim=1)  # Shape: (batch, d_model, 1)
        x_concat = self.se_block(x_concat)
        x_concat = x_concat.transpose(1, 2)  # Shape: (batch, 1, d_model)

        x_concat = self.rotary_encoder(x_concat)
        x_encoded = self.transformer_encoder(x_concat)
        x_encoded = self.global_context(x_encoded)

        attn_weights = self.attention_pooling(x_encoded)
        x_pooled = torch.sum(x_encoded * attn_weights, dim=1)  # Shape: (batch, d_model)

        basic_emb = self.basic_feat_proj(basic_feats)
        if enhanced_feats is not None:
            enhanced_emb = self.enhanced_feat_proj(enhanced_feats)
            freq_emb = torch.cat([basic_emb, enhanced_emb], dim=1)
        else:
            freq_emb = torch.cat([basic_emb, basic_emb], dim=1)

        combined = torch.cat([x_pooled, freq_emb], dim=1)
        combined = self.feature_integration(combined)

        logits1 = self.expert1(combined)
        logits2 = self.expert2(combined)
        logits3 = self.expert3(combined)

        gates = F.softmax(self.gate(combined), dim=1)
        out = (gates[:, 0:1] * logits1 +
               gates[:, 1:2] * logits2 +
               gates[:, 2:3] * logits3)

        return out

# --------------------------------------------------
# 12. SE Block
# --------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


# --------------------------------------------------
# 13. Specialist Models for Confused Classes
# --------------------------------------------------
class SpecialistModel(nn.Module):
    def __init__(self, params, num_basic_feats=7, num_enhanced_feats=55):
        super().__init__()
        scaled_params = params.copy()
        scaled_params['d_model'] = params['d_model'] // 2
        scaled_params['num_layers'] = max(1, params['num_layers'] // 2)

        self.model = EnhancedEEGTransformer(
            scaled_params,
            num_basic_feats=num_basic_feats,
            num_enhanced_feats=num_enhanced_feats,
            num_classes=2
        )

    def forward(self, x, basic_feats, enhanced_feats=None):
        return self.model(x, basic_feats, enhanced_feats)


# --------------------------------------------------
# 14. Auto-Detect Underperforming Classes and Utility Functions
# --------------------------------------------------
def detect_underperforming_classes(y_true, y_pred, label_encoder, threshold=0.7):
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:", flush=True)
    print(cm, flush=True)
    class_accuracy = {}
    for i in range(len(cm)):
        class_name = label_encoder.inverse_transform([i])[0]
        correct = cm[i, i]
        total = np.sum(cm[i, :])
        accuracy = correct / total if total > 0 else 0
        class_accuracy[i] = accuracy
        print(f"Class {class_name}: Accuracy = {accuracy:.2f}", flush=True)
    underperforming = [cls_idx for cls_idx, acc in class_accuracy.items() if acc < threshold]
    confusion_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] >= 3:
                confusion_pairs.append((i, j))
                print(
                    f"Class {label_encoder.inverse_transform([i])[0]} confused with {label_encoder.inverse_transform([j])[0]}: {cm[i, j]} samples",
                    flush=True)
    return underperforming, confusion_pairs, cm


def calculate_class_weights(y):
    class_counts = np.bincount(y)
    total = len(y)
    weights = torch.FloatTensor([total / (len(class_counts) * count) for count in class_counts])
    return weights


def save_checkpoint(model, optimizer, epoch, accuracy, params, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'params': params
    }, filename)


# --------------------------------------------------
# 15. Cosine Warmup Scheduler and Gradient Accumulation Training
# --------------------------------------------------
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_with_gradient_accumulation(model, optimizer, loader, criterion, scheduler, device, params,
                                     accumulation_steps=4):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        if len(batch) == 4:
            inputs, basic_feats, enhanced_feats, targets = batch
            inputs = inputs.to(device)
            basic_feats = basic_feats.to(device)
            enhanced_feats = enhanced_feats.to(device)
            targets = targets.to(device)
            outputs = model(inputs, basic_feats, enhanced_feats)
            loss = criterion(outputs, targets) / accumulation_steps
        else:
            inputs, basic_feats, targets = batch
            inputs = inputs.to(device)
            basic_feats = basic_feats.to(device)
            targets = targets.to(device)
            outputs = model(inputs, basic_feats)
            loss = criterion(outputs, targets) / accumulation_steps

        loss.backward()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss detected at step {i}")
            continue

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        running_loss += loss.item() * accumulation_steps
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_improved(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                inputs, basic_feats, enhanced_feats, targets = batch
                inputs = inputs.to(device)
                basic_feats = basic_feats.to(device)
                enhanced_feats = enhanced_feats.to(device)
                targets = targets.to(device)
                outputs = model(inputs, basic_feats, enhanced_feats)
            else:
                inputs, basic_feats, targets = batch
                inputs = inputs.to(device)
                basic_feats = basic_feats.to(device)
                targets = targets.to(device)
                outputs = model(inputs, basic_feats)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, predictions, true_labels


# --------------------------------------------------
# 16. Train Specialist Models
# --------------------------------------------------
def train_specialist_model(X, basic_features, enhanced_features, y, class_pair, label_encoder, params, device):
    cls1, cls2 = class_pair
    cls1_name = label_encoder.inverse_transform([cls1])[0]
    cls2_name = label_encoder.inverse_transform([cls2])[0]
    print(f"Training specialist model for classes {cls1_name} vs {cls2_name}", flush=True)
    idx1 = np.where(y == cls1)[0]
    idx2 = np.where(y == cls2)[0]
    indices = np.concatenate([idx1, idx2])
    X_pair = X[indices]
    basic_pair = basic_features[indices]
    enhanced_pair = enhanced_features[indices]
    y_pair = np.zeros(len(indices), dtype=np.int64)
    y_pair[len(idx1):] = 1
    pair_dataset = AdaptiveEEGDataset(
        X_pair, basic_pair, enhanced_pair, y_pair,
        augment=True, augment_factor=3
    )
    train_size = int(0.8 * len(pair_dataset))
    val_size = len(pair_dataset) - train_size
    train_set, val_set = random_split(pair_dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False)
    model = SpecialistModel(
        params,
        num_basic_feats=basic_features.shape[1],
        num_enhanced_feats=enhanced_features.shape[1]
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'] * 0.8,
        weight_decay=params['weight_decay']
    )
    total_steps = len(train_loader) * 50 // params['accumulation_steps']
    warmup_steps = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=1e-6
    )
    criterion = FocalLoss(gamma=params['focal_loss_gamma'])
    best_val_acc = 0.0
    patience_counter = 0
    max_epochs = 50
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_with_gradient_accumulation(
            model, optimizer, train_loader, criterion, scheduler, device, params,
            accumulation_steps=params['accumulation_steps']
        )
        val_loss, val_acc, _, _ = validate_improved(model, val_loader, device, criterion)
        print(
            f"Specialist {cls1_name}-{cls2_name} | Epoch {epoch}/{max_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%",
            flush=True)
        if val_acc > best_val_acc + IMPROVEMENT_THRESHOLD:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = os.path.join(SPECIALIST_MODELS_DIR, f"specialist_{cls1}_{cls2}.pth")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping triggered for specialist {cls1_name}-{cls2_name}", flush=True)
                break
    model_path = os.path.join(SPECIALIST_MODELS_DIR, f"specialist_{cls1}_{cls2}.pth")
    model.load_state_dict(torch.load(model_path))
    return model, best_val_acc


# --------------------------------------------------
# 17. Ensemble Prediction with Specialists
# --------------------------------------------------
def predict_with_specialists(main_model, specialist_models, inputs, basic_feats, enhanced_feats,
                             confusion_pairs, label_encoder, device):
    with torch.no_grad():
        main_outputs = main_model(inputs, basic_feats, enhanced_feats)
        main_probs = F.softmax(main_outputs, dim=1)
        main_pred = torch.argmax(main_probs, dim=1)
    main_pred_np = main_pred.cpu().numpy()
    main_probs_np = main_probs.cpu().numpy()
    final_pred = main_pred_np.copy()
    for i in range(len(main_pred_np)):
        pred_class = main_pred_np[i]
        confidence = main_probs_np[i, pred_class]
        if confidence < 0.8:
            for (cls1, cls2), specialist in specialist_models.items():
                if pred_class == cls1 or pred_class == cls2:
                    with torch.no_grad():
                        spec_output = specialist(
                            inputs[i:i + 1], basic_feats[i:i + 1], enhanced_feats[i:i + 1]
                        )
                        spec_prob = F.softmax(spec_output, dim=1)
                        spec_pred = torch.argmax(spec_prob, dim=1).item()
                    refined_pred = cls1 if spec_pred == 0 else cls2
                    if spec_prob[0, spec_pred].item() > 0.7:
                        final_pred[i] = refined_pred
                        print(
                            f"Specialist overrode prediction: {label_encoder.inverse_transform([pred_class])[0]} -> {label_encoder.inverse_transform([refined_pred])[0]}",
                            flush=True)
    return final_pred


# --------------------------------------------------
# 18. Main Training Pipeline
# --------------------------------------------------
def main():
    print(f"Using device: {device}", flush=True)
    hp_mgr = HyperparameterManager()
    params = hp_mgr.get_params()
    params['batch_size'] = 128
    params['learning_rate'] = 0.00005
    params['window_size'] = 1
    MAX_EPOCHS = 1100

    print("Loading and preprocessing data...", flush=True)
    df = pd.read_csv(DATA_FILE)
    print(f"Raw DataFrame shape: {df.shape}", flush=True)

    X, basic_features, enhanced_features, y, label_encoder = prepare_data(df, params)
    assert X.shape[0] == 300000, f"Expected 300k samples, got {X.shape[0]}"
    print(f"Data prepared with shape: X={X.shape}, basic_features={basic_features.shape}, "
          f"enhanced_features={enhanced_features.shape}, y={y.shape}", flush=True)

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.1, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}", flush=True)

    train_dataset = AdaptiveEEGDataset(
        X[train_idx], basic_features[train_idx], enhanced_features[train_idx], y[train_idx],
        augment=True
    )
    test_dataset = AdaptiveEEGDataset(
        X[test_idx], basic_features[test_idx], enhanced_features[test_idx], y[test_idx],
        augment=False
    )
    print(f"Train dataset: {len(train_dataset)}, Test dataset: {len(test_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)
    print(f"Train loader: {len(train_loader)} batches, {len(train_loader.dataset)} samples", flush=True)

    num_classes = len(np.unique(y))
    model = EnhancedEEGTransformer(
        params,
        num_basic_feats=basic_features.shape[1],  # 7
        num_enhanced_feats=enhanced_features.shape[1],  # 12
        num_classes=num_classes
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )

    # ------------------------------------------------------------------
    # New: Prompt user to load from checkpoint if available (before training begins)
    # ------------------------------------------------------------------
    selected_checkpoint = load_checkpoint_prompt()
    if selected_checkpoint is not None:
        checkpoint = torch.load(selected_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint['accuracy']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.2f}%")
    else:
        best_acc = 0.0
        start_epoch = 1

    class_weights = calculate_class_weights(y[train_idx]).to(device)
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=params['focal_loss_gamma'],
        label_smoothing=params['label_smoothing']
    )
    total_steps = len(train_loader) * MAX_EPOCHS // params['accumulation_steps']
    warmup_steps = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=1e-6
    )

    print("\n===== Phase 1: Initial Training =====", flush=True)
    best_checkpoint_path = None
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        train_loss, train_acc = train_with_gradient_accumulation(
            model, optimizer, train_loader, criterion, scheduler, device, params,
            accumulation_steps=params['accumulation_steps']
        )
        val_loss, val_acc, y_pred, y_true = validate_improved(
            model, test_loader, device, criterion
        )
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch}/{MAX_EPOCHS} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}",
            flush=True)
        if val_acc > best_acc + IMPROVEMENT_THRESHOLD:
            best_acc = val_acc
            best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, val_acc, params, best_checkpoint_path)
        else:
            # Early stopping for Phase 1: if no improvement for half of MAX_PATIENCE
            if epoch - start_epoch >= MAX_PATIENCE // 2:
                print("Early stopping triggered in Phase 1.", flush=True)
                break

    if best_checkpoint_path is not None:
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(best_checkpoint_path) if best_checkpoint_path is not None else None
    _, _, y_pred, y_true = validate_improved(model, test_loader, device, criterion)
    underperforming, confusion_pairs, cm = detect_underperforming_classes(
        y_true, y_pred, label_encoder, threshold=0.7
    )

    print("\n===== Phase 2: Training Specialist Models =====", flush=True)
    specialist_models = {}
    if confusion_pairs:
        for class_pair in confusion_pairs:
            specialist, specialist_acc = train_specialist_model(
                X[train_idx], basic_features[train_idx], enhanced_features[train_idx],
                y[train_idx], class_pair, label_encoder, params, device
            )
            specialist_models[class_pair] = specialist

    print("\n===== Phase 3: Final Training with Targeted Augmentation =====", flush=True)
    enhanced_train_dataset = AdaptiveEEGDataset(
        X[train_idx], basic_features[train_idx], enhanced_features[train_idx], y[train_idx],
        confusion_matrix=cm, underperforming_classes=underperforming,
        augment=True, augment_factor=3
    )
    enhanced_train_loader = DataLoader(
        enhanced_train_dataset, batch_size=params['batch_size'], shuffle=True
    )
    # Reload best checkpoint from Phase 1 before Phase 3 training
    if best_checkpoint_path is not None:
        model.load_state_dict(torch.load(best_checkpoint_path)['model_state_dict'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'] * 0.5,
        weight_decay=params['weight_decay'] * 1.5
    )
    total_steps = len(enhanced_train_loader) * MAX_EPOCHS // params['accumulation_steps']
    warmup_steps = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=1e-6
    )
    best_final_acc = 0.0
    best_final_checkpoint_path = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_with_gradient_accumulation(
            model, optimizer, enhanced_train_loader, criterion, scheduler, device, params,
            accumulation_steps=params['accumulation_steps']
        )
        val_loss, val_acc, y_pred, y_true = validate_improved(
            model, test_loader, device, criterion
        )
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Final Epoch {epoch}/{MAX_EPOCHS} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}",
            flush=True)
        if val_acc > best_final_acc + IMPROVEMENT_THRESHOLD:
            best_final_acc = val_acc
            best_final_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, val_acc, params, best_final_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= MAX_PATIENCE:
                print("Early stopping triggered in Phase 3.", flush=True)
                break

    if best_final_checkpoint_path is not None:
        checkpoint = torch.load(best_final_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    print("\n===== Final Evaluation with Specialist Ensemble =====", flush=True)
    all_inputs = []
    all_basic_feats = []
    all_enhanced_feats = []
    all_targets = []
    for batch in test_loader:
        inputs, basic_feats, enhanced_feats, targets = batch
        all_inputs.append(inputs)
        all_basic_feats.append(basic_feats)
        all_enhanced_feats.append(enhanced_feats)
        all_targets.append(targets)
    test_inputs = torch.cat(all_inputs).to(device)
    test_basic_feats = torch.cat(all_basic_feats).to(device)
    test_enhanced_feats = torch.cat(all_enhanced_feats).to(device)
    test_targets = torch.cat(all_targets).cpu().numpy()

    _, _, main_pred, main_true = validate_improved(model, test_loader, device, criterion)
    main_acc = accuracy_score(main_true, main_pred) * 100

    if specialist_models:
        ensemble_pred = predict_with_specialists(
            model, specialist_models, test_inputs, test_basic_feats, test_enhanced_feats,
            confusion_pairs, label_encoder, device
        )
        ensemble_acc = accuracy_score(test_targets, ensemble_pred) * 100
    else:
        ensemble_pred = main_pred
        ensemble_acc = main_acc

    print("\n================ FINAL RESULTS ================", flush=True)
    print(f"Phase 1 Best Accuracy: {best_acc:.2f}%", flush=True)
    print(f"Phase 3 Best Accuracy: {best_final_acc:.2f}%", flush=True)
    print(f"Main Model Final Accuracy: {main_acc:.2f}%", flush=True)
    print(f"Ensemble Final Accuracy: {ensemble_acc:.2f}%", flush=True)
    print("\nClassification Report (Ensemble):", flush=True)
    print(classification_report(test_targets, ensemble_pred, target_names=label_encoder.classes_), flush=True)
    print("\nConfusion Matrix (Ensemble):", flush=True)
    cm_final = confusion_matrix(test_targets, ensemble_pred)
    print(cm_final, flush=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_final, annot=True, fmt='d',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Final Ensemble Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_ensemble.png')
    plt.close()
    class_accuracies = []
    for i in range(len(cm_final)):
        correct = cm_final[i, i]
        total = np.sum(cm_final[i, :])
        acc = correct / total if total > 0 else 0
        class_accuracies.append(acc * 100)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(label_encoder.classes_, class_accuracies)
    plt.axhline(y=80, color='r', linestyle='--', label='Target: 80%')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Class')
    plt.title('Per-Class Accuracy (Final Ensemble)')
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('class_accuracies.png')
    plt.close()
    print("Training complete. Model saved in checkpoint directory:", CHECKPOINT_DIR, flush=True)


if __name__ == "__main__":
    main()
