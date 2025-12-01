"""Configuration helpers shared across the refactored bird song training bundle."""

from pathlib import Path


class DataConfig:
    """Dataset locations and split defaults."""

    # Resolve project root (repo root) from this config file location
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Dataset root in this repository: data/dataset/...
    DATA_ROOT = PROJECT_ROOT / "data" / "dataset"
    TRAIN_CSV = DATA_ROOT / "birdclef-2023" / "train_metadata.csv"
    TRAIN_AUDIO_DIR = DATA_ROOT / "birdclef-2023" / "train_audio"
    NOISE_ROOT_DIR = DATA_ROOT / "background_noise"
    ESC50_ROOT_DIR = DATA_ROOT / "ESC-50-master"
    ESC50_AUDIO_DIR = ESC50_ROOT_DIR / "audio"
    ESC50_META_PATH = ESC50_ROOT_DIR / "meta" / "esc50.csv"
    ESC50_CATEGORIES = [
        "breathing",
        "clapping",
        "coughing",
        "crying_baby",
        "laughing",
        "sneezing",
        "snoring",
    ]
    ESC50_DUPLICATION = 2
    USE_ESC50 = True

    VALID_AUDIO_EXTENSIONS = [".ogg", ".wav", ".mp3", ".mp4", ".flac"]
    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = 0

    VALID_SIZE = 0.10
    TEST_SIZE = 0.05
    RANDOM_STATE = 3407

    TRAIN_METADATA = PROJECT_ROOT / "data" / "train_metadata.csv"
    VALID_METADATA = PROJECT_ROOT / "data" / "valid_metadata.csv"
    TEST_METADATA = PROJECT_ROOT / "data" / "test_metadata.csv"
    PROCESSED_TRAIN_METADATA = PROJECT_ROOT / "data" / "train_metadata_processed.csv"
    PROCESSED_VALID_METADATA = PROJECT_ROOT / "data" / "valid_metadata_processed.csv"


class TrainingConfig(DataConfig):
    """Model and optimisation defaults."""

    TARGET_SR = 16000
    SEGMENT_SECONDS = 3
    SEGMENT_SAMPLES = TARGET_SR * SEGMENT_SECONDS
    ENERGY_WINDOW = 1024
    ENERGY_STRIDE = 256
    POS_JITTER_SECONDS = 0.25

    DEVICE = "cuda"
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    MIN_LR = 1e-5
    WARMUP_EPOCHS = 3
    GRAD_CLIP = 0.5
    NUM_WORKERS = 8

    base_channels = 12
    classifier_hidden = 24
    model_dropout = 0.05

    MODEL_EXPORT = Path("model.pth")

    # Audio preprocessing and augmentation
    ENABLE_PEAK_NORM = True
    TARGET_PEAK = 0.891
    PEAK_ONLY_ATTENUATE = True

    NOISE_ADD_PROB = 0.3  # probability to apply noise to a sample
    NOISE_REL_STD = 0.05  
