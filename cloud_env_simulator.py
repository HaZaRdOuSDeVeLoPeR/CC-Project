import numpy as np
import random
from configuration import SERVER_COUNT, MIN_SERVER_COUNT, MAX_SERVER_COUNT, AUTOSCALING, SLA_VIOLATION_LATENCY, EPISODE_LENGTH,\
MAX_SERVER_UTILIZATION, PROCESSING_POWER, AVG_INCOMING_REQUESTS, AVG_REQUEST_WORKLOAD, REQUEST_WORKLOAD_VARIANCE, MAX_QUEUE_LENGTH, REWARD_MIN

class CloudEnvironment:
    def __init__(self, num_servers = SERVER_COUNT, max_servers = MAX_SERVER_COUNT, min_servers = MIN_SERVER_COUNT, autoscale = AUTOSCALING):

        self.initial_servers = num_servers
        self.num_servers = num_servers
        self.max_servers = max_servers
        self.min_servers = min_servers
        self.autoscale = autoscale

        self.server_capacity = MAX_SERVER_UTILIZATION  # max CPU per server
        self.processing_rate = PROCESSING_POWER   # CPU units processed per step
        self.reset()

    def reset(self):
        self.num_servers = self.initial_servers
        self.cpu_usage = np.zeros(self.max_servers)
        self.queues = np.zeros(self.max_servers)
        self.time = 0
        self.total_latency = 0
        self.total_requests = 0
        return self._get_state()

    def step(self, action):
        reward = 0

        # simulate no of random requests in each time step
        incoming_requests = np.random.poisson(lam = AVG_INCOMING_REQUESTS)

        for _ in range(incoming_requests):
            if self.autoscale and action == self.num_servers:
                self._scale_up()
            elif self.autoscale and action == self.num_servers + 1:
                self._scale_down()
            else:
                server_id = action
                if server_id < self.num_servers:
                    self.queues[server_id] += random.randint(AVG_REQUEST_WORKLOAD - REQUEST_WORKLOAD_VARIANCE, AVG_REQUEST_WORKLOAD + REQUEST_WORKLOAD_VARIANCE)
                    self.queues[server_id] = min(self.queues[server_id], MAX_QUEUE_LENGTH)

        # Process queues
        for i in range(self.num_servers):
            processed = min(self.processing_rate, self.queues[i])
            self.queues[i] -= processed
            self.cpu_usage[i] = self.queues[i] / self.server_capacity

        avg_latency = np.mean(self.queues[:self.num_servers])
        load_variance = np.var(self.cpu_usage[:self.num_servers])
        sla_violation = 1 if avg_latency > SLA_VIOLATION_LATENCY else 0

        reward = - avg_latency - 5 * sla_violation - load_variance - 0.1 * self.num_servers
        
        # Reward clipping (stabilization)
        # reward = np.clip(reward, REWARD_MIN, 0)

        self.total_latency += avg_latency
        self.total_requests += incoming_requests
        self.time += 1

        done = self.time >= EPISODE_LENGTH

        return self._get_state(), reward, done

    def _scale_up(self):
        if self.num_servers < self.max_servers:
            self.num_servers += 1

    def _scale_down(self):
        if self.num_servers > self.min_servers:
            self.num_servers -= 1

    def _get_state(self):
        state = []

        state.extend(self.cpu_usage[:self.max_servers])
        state.extend(self.queues[:self.max_servers] / float(MAX_QUEUE_LENGTH))

        avg_latency = np.mean(self.queues[:self.num_servers])
        state.append(avg_latency / float(MAX_QUEUE_LENGTH))

        state.append(self.num_servers / self.max_servers)
        state.append(np.mean(self.cpu_usage[:self.num_servers]))

        return np.array(state, dtype=np.float32)