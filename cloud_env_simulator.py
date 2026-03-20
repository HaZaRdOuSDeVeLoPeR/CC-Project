import numpy as np
import random
from configuration import (
    SERVER_COUNT, SLA_VIOLATION_LATENCY, EPISODE_LENGTH,
    MAX_SERVER_UTILIZATION, PROCESSING_POWER, AVG_INCOMING_REQUESTS,
    AVG_REQUEST_WORKLOAD, REQUEST_WORKLOAD_VARIANCE, MAX_QUEUE_LENGTH
)

class CloudEnvironment:
    """
    Heterogeneous-fleet cloud load balancing environment.

    Each server has a DIFFERENT processing power drawn at environment
    construction time, with mean = PROCESSING_POWER. This is the key
    design that makes DQN non-trivially better than classical baselines:

      - Least Connections picks the shortest queue — ignores server speed.
        A short queue on a slow server can be worse than a longer queue
        on a fast server.
      - Round Robin ignores both queue length and server speed.
      - DQN observes both queue depths AND server speeds and learns to
        jointly optimise: prefer servers that are fast AND have short queues.

    Action space: SERVER_COUNT actions, one per server.
      action i  →  route all incoming requests this step to server i

    State (size = SERVER_COUNT * 3 + 2):
      cpu_usage[0..N-1]          normalised queue / MAX_QUEUE_LENGTH
      queues[0..N-1]             normalised queue length
      proc_rates[0..N-1]         normalised server speed / max_proc_rate
      avg_latency                fleet average queue (normalised)
      mean_cpu                   fleet average cpu utilisation

    Note: proc_rates are included in the state so the agent can learn
    to route to fast servers. Classical baselines don't use this info.

    Reward:
      -queue_before_routing   immediate signal: cost of routing to that server
      -avg_latency            fleet-wide health signal
      -5 * sla_violation      hard SLA breach penalty
    """

    def __init__(self, num_servers=SERVER_COUNT, avg_proc_rate=PROCESSING_POWER):
        self.num_servers  = num_servers
        self.avg_proc_rate = avg_proc_rate
        self.server_capacity = MAX_SERVER_UTILIZATION

        # Assign heterogeneous processing powers.
        # Draw from a uniform distribution and rescale so mean = avg_proc_rate.
        # Clamp to [avg/3, avg*3] to avoid degenerate servers.
        rng = np.random.default_rng()
        raw = rng.uniform(0.3, 1.7, size=num_servers)
        scaled = raw / raw.mean() * avg_proc_rate
        self.proc_rates = np.clip(scaled, avg_proc_rate / 3, avg_proc_rate * 3)
        self.max_proc_rate = float(self.proc_rates.max())

        self.action_map = []
        for i in range(self.num_servers):
            for j in range(i, self.num_servers):
                self.action_map.append((i, j))

        self.reset()

    def reset(self):
        self.cpu_usage      = np.zeros(self.num_servers)
        self.queues         = np.zeros(self.num_servers)
        self.time           = 0
        self.total_latency  = 0
        self.total_requests = 0
        return self._get_state()

    def step(self, action):
        # Capture queue depth BEFORE routing — immediate reward signal
        a, b = self.action_map[action]
        queue_before_routing = (self.queues[a] + self.queues[b]) / 2.0

        incoming = np.random.poisson(lam=AVG_INCOMING_REQUESTS)
        for _ in range(incoming):
            target = a if random.random() < 0.5 else b

            workload = random.randint(
                AVG_REQUEST_WORKLOAD - REQUEST_WORKLOAD_VARIANCE,
                AVG_REQUEST_WORKLOAD + REQUEST_WORKLOAD_VARIANCE,
            )

            self.queues[target] += workload
            self.queues[target] = min(self.queues[target], MAX_QUEUE_LENGTH)

        # Process queues — each server drains at its own speed
        for k in range(self.num_servers):
            processed         = min(self.proc_rates[k], self.queues[k])
            self.queues[k]   -= processed
            self.cpu_usage[k] = self.queues[k] / self.server_capacity

        avg_latency   = float(np.mean(self.queues))
        avg_latency = np.clip(avg_latency, 0, SLA_VIOLATION_LATENCY * 2)
        sla_violation = 1 if avg_latency > SLA_VIOLATION_LATENCY else 0

        # Throughput-aware reward
        # total processed (throughput)
        total_processed = np.sum([
            min(self.proc_rates[k], self.queues[k])
            for k in range(self.num_servers)
        ])

        # load imbalance
        queue_std = np.std(self.queues)

        # CPU utilization
        cpu_util = np.mean(self.cpu_usage)

        reward = (
            +0.5 * total_processed      # STRONG throughput reward
            +1.0 * cpu_util             # reward utilization
            -0.3 * queue_std            # mild balance penalty
            -0.5 * avg_latency          # reduce weight on latency
            -20.0 * sla_violation
        )

        self.total_latency  += avg_latency
        self.total_requests += incoming
        self.time           += 1

        done = self.time >= EPISODE_LENGTH
        return self._get_state(), reward, done

    def _get_state(self):
        state = []
        state.extend(self.cpu_usage.tolist())
        state.extend((self.queues / float(MAX_QUEUE_LENGTH)).tolist())
        # Normalised processing rates — this is what classical baselines LACK.
        # DQN uses this to learn "fast server, short queue = best choice".
        state.extend((self.proc_rates / self.max_proc_rate).tolist())
        state.append(float(np.mean(self.queues)) / float(MAX_QUEUE_LENGTH))
        state.append(float(np.mean(self.cpu_usage)))
        return np.array(state, dtype=np.float32)
