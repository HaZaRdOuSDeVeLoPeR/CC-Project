# ── Infrastructure ────────────────────────────────────────────────────────────
SERVER_COUNT            = 10    # fixed number of servers
MAX_SERVER_UTILIZATION  = 100   # max CPU capacity per server
PROCESSING_POWER        = 10    # avg CPU units processed per server per step
PROCESSING_POWER_VARIANCE = 0.5  # server speed drawn from uniform(avg*(1-var), avg*(1+var))
MAX_QUEUE_LENGTH        = 200   # max queue depth per server

# ── Workload ──────────────────────────────────────────────────────────────────
AVG_INCOMING_REQUESTS   = 10     # Poisson λ — avg requests per step
AVG_REQUEST_WORKLOAD    = 10    # avg CPU units per request
REQUEST_WORKLOAD_VARIANCE = 5   # workload ± variance

# ── SLA ───────────────────────────────────────────────────────────────────────
SLA_VIOLATION_LATENCY   = 80    # avg queue depth threshold for SLA breach

# ── Simulation ────────────────────────────────────────────────────────────────
EPISODE_LENGTH          = 500   # steps per episode
TRAIN_EPISODE_COUNT     = 800   # training episodes
TEST_EPISODE_COUNT      = 300   # evaluation episodes

# ── DQN hyperparameters ───────────────────────────────────────────────────────
GAMMA                   = 0.99
LEARNING_RATE           = 0.0005
BATCH_SIZE              = 128
REPLAY_BUFFER_SIZE      = 25000
EPSILON_START           = 1.0
EPSILON_MIN             = 0.05
TARGET_UPDATE_STEPS     = 500
HIDDEN_DIM              = 128
DEVICE                  = "cpu"

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "./artifacts/dqn_trained_model.pth"
REWARD_PATH = "./artifacts/training_rewards.npy"