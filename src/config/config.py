"""Configuration helpers shared across the refactored bird song training bundle."""

from pathlib import Path


class DataConfig:
    """Dataset locations and split defaults."""

    DATA_ROOT = Path("dataset/train_data")
    TRAIN_CSV = DATA_ROOT / "birdclef-2023" / "train_metadata.csv"
    TRAIN_AUDIO_DIR = DATA_ROOT / "birdclef-2023" / "train_audio"
    NOISE_ROOT_DIR = DATA_ROOT / "background_noise"

    VALID_AUDIO_EXTENSIONS = [".ogg", ".wav", ".mp3", ".mp4", ".flac"]
    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = 0

    VALID_SIZE = 0.10
    TEST_SIZE = 0.05
    RANDOM_STATE = 42

    TRAIN_METADATA = Path("train_metadata.csv")
    VALID_METADATA = Path("valid_metadata.csv")
    TEST_METADATA = Path("test_metadata.csv")
    PROCESSED_TRAIN_METADATA = Path("train_metadata_processed.csv")
    PROCESSED_VALID_METADATA = Path("valid_metadata_processed.csv")


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
