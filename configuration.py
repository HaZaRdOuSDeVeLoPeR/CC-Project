# ===============================
# CLOUD INFRASTRUCTURE SETTINGS
# ===============================

# Initial number of active servers at environment reset
SERVER_COUNT = 10

# Minimum number of servers allowed when autoscaling is enabled
MIN_SERVER_COUNT = 5

# Maximum number of servers allowed when autoscaling is enabled
MAX_SERVER_COUNT = 15

# Enable or disable autoscaling
AUTOSCALING = True

# ===============================
# WORKLOAD CHARACTERISTICS
# ===============================

# Average number of incoming requests per time step (Poisson λ)
AVG_INCOMING_REQUESTS = 5

# Average computational workload per request (CPU units)
AVG_REQUEST_WORKLOAD = 10

# Variability in request workload (+/- range)
REQUEST_WORKLOAD_VARIANCE = 5

# ===============================
# SERVER PROCESSING PARAMETERS
# ===============================

# Maximum CPU capacity of each server (used for utilization calculation)
MAX_SERVER_UTILIZATION = 100

# CPU units processed per server per time step
PROCESSING_POWER = 10

# Maximum queue length per server (prevents explosion)
MAX_QUEUE_LENGTH = 200

# ===============================
# SLA & PERFORMANCE SETTINGS
# ===============================

# Latency threshold above which SLA violation occurs
SLA_VIOLATION_LATENCY = 80

# Set Minimum Reward (Reward Clipping to Stabilize Training)
# Always Negative
REWARD_MIN = -100

# ===============================
# SIMULATION SETTINGS
# ===============================

# Total Episodes used to Test the Models (Baselines and DQN)
TEST_EPISODE_COUNT = 300

# Number of time steps per episode for simulation and Training
EPISODE_LENGTH = 500

# ===============================
# TRAINING SETTINGS
# ===============================

# Number of training episodes
TRAIN_EPISODE_COUNT = 300

# select device for training ("auto", "cpu", "cuda")
DEVICE = "cpu"

# Model save path
MODEL_PATH = "./artifacts/dqn_trained_model.pth"

# Training rewards save path
REWARD_PATH = "./artifacts/training_rewards.npy"

# ===============================
# DQN HYPERPARAMETERS
# ===============================

# Discount factor (future reward importance)
GAMMA = 0.99

# Learning rate for optimizer
LEARNING_RATE = 0.0005

# Batch size for replay training (Keep under 128 for CPU and beyond 256 for GPU)
BATCH_SIZE = 128

# Replay buffer capacity (How many past states(steps) to store in memory)
REPLAY_BUFFER_SIZE = 10000

# Initial exploration probability
EPSILON_START = 1.0

# Minimum exploration probability
EPSILON_MIN = 0.05

# Epsilon decay per episode
EPSILON_DECAY = 0.98

# Target network update frequency (in steps)
TARGET_UPDATE_STEPS = 500

# Total Hidden Layers (Keep under 128 for CPU and beyond 256 for GPU)
HIDDEN_DIM = 128