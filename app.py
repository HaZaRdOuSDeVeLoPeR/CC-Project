from flask import Flask, render_template, request, jsonify, session
import numpy as np
try:
    import torch
except ImportError:
    torch = None
import threading
import json
import os
import time
import random
from collections import deque

app = Flask(__name__)
app.secret_key = "rl_loadbalancer_secret_2024"

# Global state
training_state = {
    "running": False,
    "episode": 0,
    "total_episodes": 0,
    "rewards": [],
    "epsilon": 1.0,
    "done": False,
    "error": None,
    "q_values": [],
}

eval_state = {
    "running": False,
    "done": False,
    "error": None,
    "progress": 0,
    "results": {},
    "step_metrics": {
        "dqn": [],
        "random": [],
        "round_robin": [],
        "least_connections": [],
    },
}

train_config = {}


def get_default_config():
    return {
        "SERVER_COUNT": 10,
        "MIN_SERVER_COUNT": 5,
        "MAX_SERVER_COUNT": 15,
        "AUTOSCALING": True,
        "AVG_INCOMING_REQUESTS": 5,
        "AVG_REQUEST_WORKLOAD": 10,
        "REQUEST_WORKLOAD_VARIANCE": 5,
        "MAX_SERVER_UTILIZATION": 100,
        "PROCESSING_POWER": 10,
        "MAX_QUEUE_LENGTH": 200,
        "SLA_VIOLATION_LATENCY": 80,
        "EPISODE_LENGTH": 500,
        "TRAIN_EPISODE_COUNT": 300,
        "TEST_EPISODE_COUNT": 300,
        "GAMMA": 0.99,
        "LEARNING_RATE": 0.0005,
        "BATCH_SIZE": 128,
        "REPLAY_BUFFER_SIZE": 10000,
        "EPSILON_START": 1.0,
        "EPSILON_MIN": 0.05,
        "EPSILON_DECAY": 0.98,
        "TARGET_UPDATE_STEPS": 500,
        "HIDDEN_DIM": 128,
        "DEVICE": "cpu",
    }


def detect_available_devices():
    """Probe what compute devices are actually available on this machine."""
    devices = [{"id": "cpu", "label": "CPU", "available": True, "info": "Always available"}]
    if torch is not None:
        # CUDA
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024 ** 3)
                devices.append({
                    "id": f"cuda:{i}" if torch.cuda.device_count() > 1 else "cuda",
                    "label": f"GPU {i}: {props.name}",
                    "available": True,
                    "info": f"{mem_gb:.1f} GB VRAM · CUDA {props.major}.{props.minor}"
                })
        else:
            devices.append({
                "id": "cuda",
                "label": "CUDA GPU",
                "available": False,
                "info": "No CUDA-capable GPU detected (or CPU-only torch installed)"
            })
        # MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append({
                "id": "mps",
                "label": "Apple MPS (Metal)",
                "available": True,
                "info": "Apple Silicon GPU via Metal Performance Shaders"
            })
    else:
        devices.append({
            "id": "cuda",
            "label": "CUDA GPU",
            "available": False,
            "info": "torch not installed — install the GPU build of torch to enable"
        })
    return devices


@app.route("/")
def index():
    model_exists = os.path.exists("./artifacts/dqn_trained_model.pth")
    return render_template("index.html", model_exists=model_exists)


@app.route("/train")
def train_page():
    config = get_default_config()
    devices = detect_available_devices()
    return render_template("train.html", config=config, devices=devices)


@app.route("/api/devices")
def api_devices():
    return jsonify(detect_available_devices())


@app.route("/evaluate")
def evaluate_page():
    model_exists = os.path.exists("./artifacts/dqn_trained_model.pth")
    config = get_default_config()
    return render_template("evaluate.html", config=config, model_exists=model_exists, train_config=train_config)


@app.route("/api/start_training", methods=["POST"])
def start_training():
    global training_state, train_config

    if training_state["running"]:
        return jsonify({"error": "Training already running"}), 400

    cfg = request.json
    train_config = cfg

    training_state = {
        "running": True,
        "episode": 0,
        "total_episodes": int(cfg.get("TRAIN_EPISODE_COUNT", 300)),
        "rewards": [],
        "epsilon": float(cfg.get("EPSILON_START", 1.0)),
        "done": False,
        "error": None,
        "q_values": [],
        "device": str(cfg.get("DEVICE", "cpu")),
    }

    thread = threading.Thread(target=run_training, args=(cfg,), daemon=True)
    thread.start()

    return jsonify({"status": "started"})


def run_training(cfg):
    global training_state

    try:
        # ── Pull all values directly from cfg, never rely on module-level globals ──
        episodes     = int(cfg.get("TRAIN_EPISODE_COUNT", 300))
        max_steps    = int(cfg.get("EPISODE_LENGTH", 500))
        autoscale    = bool(cfg.get("AUTOSCALING", True))
        max_servers  = int(cfg.get("MAX_SERVER_COUNT", 15))
        min_servers  = int(cfg.get("MIN_SERVER_COUNT", 5))
        server_count = int(cfg.get("SERVER_COUNT", 10))
        device_str   = str(cfg.get("DEVICE", "cpu"))
        model_path   = "./artifacts/dqn_trained_model.pth"
        reward_path  = "./artifacts/training_rewards.npy"

        # Env params
        avg_req      = int(cfg.get("AVG_INCOMING_REQUESTS", 5))
        avg_wl       = int(cfg.get("AVG_REQUEST_WORKLOAD", 10))
        wl_var       = int(cfg.get("REQUEST_WORKLOAD_VARIANCE", 5))
        max_util     = int(cfg.get("MAX_SERVER_UTILIZATION", 100))
        proc_power   = int(cfg.get("PROCESSING_POWER", 10))
        max_queue    = int(cfg.get("MAX_QUEUE_LENGTH", 200))
        sla_thresh   = int(cfg.get("SLA_VIOLATION_LATENCY", 80))

        # DQN params
        gamma        = float(cfg.get("GAMMA", 0.99))
        lr           = float(cfg.get("LEARNING_RATE", 0.0005))
        batch_size   = int(cfg.get("BATCH_SIZE", 128))
        buf_size     = int(cfg.get("REPLAY_BUFFER_SIZE", 10000))
        eps_start    = float(cfg.get("EPSILON_START", 1.0))
        eps_min      = float(cfg.get("EPSILON_MIN", 0.05))
        target_upd   = int(cfg.get("TARGET_UPDATE_STEPS", 500))
        hidden_dim   = int(cfg.get("HIDDEN_DIM", 128))

        # ── Import env/agent directly (no reload — avoids half-init race) ──
        from cloud_env_simulator import CloudEnvironment
        from dqn_agent import DQN, ReplayBuffer
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import random as pyrandom

        # Build env with explicit params (override module constants via constructor)
        import configuration as C
        C.SERVER_COUNT           = server_count
        C.MIN_SERVER_COUNT       = min_servers
        C.MAX_SERVER_COUNT       = max_servers
        C.AUTOSCALING            = autoscale
        C.AVG_INCOMING_REQUESTS  = avg_req
        C.AVG_REQUEST_WORKLOAD   = avg_wl
        C.REQUEST_WORKLOAD_VARIANCE = wl_var
        C.MAX_SERVER_UTILIZATION = max_util
        C.PROCESSING_POWER       = proc_power
        C.MAX_QUEUE_LENGTH       = max_queue
        C.SLA_VIOLATION_LATENCY  = sla_thresh
        C.DEVICE                 = device_str

        from cloud_env_simulator import CloudEnvironment
        from dqn_agent import DQN, ReplayBuffer

        env = CloudEnvironment(
            num_servers=server_count,
            max_servers=max_servers,
            min_servers=min_servers,
            autoscale=autoscale
        )
        # Inject workload params directly — bypasses the module-level name bindings
        # that 'from configuration import X' copies at import time and ignores later patches
        import cloud_env_simulator as _cem
        _cem.AVG_INCOMING_REQUESTS   = avg_req
        _cem.AVG_REQUEST_WORKLOAD    = avg_wl
        _cem.REQUEST_WORKLOAD_VARIANCE = wl_var
        _cem.MAX_QUEUE_LENGTH        = max_queue
        _cem.SLA_VIOLATION_LATENCY   = sla_thresh
        env.server_capacity          = max_util
        env.processing_rate          = proc_power

        state_size  = len(env.reset())
        action_size = max_servers + 2 if autoscale else max_servers

        # ── Build agent inline — no dependency on module constants ──
        device = torch.device(device_str)

        policy_net = DQN(state_size, action_size).to(device)
        target_net = DQN(state_size, action_size).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        memory    = ReplayBuffer(buf_size)

        epsilon      = eps_start
        eps_decay    = (eps_min / eps_start) ** (1.0 / episodes)
        step_count   = 0
        rewards_per_episode = []

        for episode in range(episodes):
            if not training_state["running"]:
                break

            state        = env.reset()
            total_reward = 0.0

            for step in range(max_steps):
                # ε-greedy action
                if pyrandom.random() < epsilon:
                    action = pyrandom.randrange(action_size)
                else:
                    with torch.no_grad():
                        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        action = torch.argmax(policy_net(s_t)).item()

                next_state, reward, done = env.step(action)
                memory.push(state, action, reward, next_state, done)

                # Train step
                if len(memory) >= batch_size:
                    states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(batch_size)
                    states_b     = states_b.to(device)
                    actions_b    = actions_b.to(device)
                    rewards_b    = rewards_b.to(device)
                    next_states_b = next_states_b.to(device)
                    dones_b      = dones_b.to(device)

                    q_vals_cur = policy_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze()

                    with torch.no_grad():
                        next_acts   = policy_net(next_states_b).argmax(1)
                        max_next_q  = target_net(next_states_b).gather(1, next_acts.unsqueeze(1)).squeeze()
                        target_q    = rewards_b + gamma * max_next_q * (1 - dones_b)

                    loss = nn.SmoothL1Loss()(q_vals_cur, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                    optimizer.step()

                    step_count += 1
                    if step_count % target_upd == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                state        = next_state
                total_reward += reward
                if done:
                    break

            rewards_per_episode.append(total_reward)

            if epsilon > eps_min:
                epsilon *= eps_decay

            # Sample Q-values every 5 episodes
            q_vals = []
            if episode % 5 == 0:
                with torch.no_grad():
                    sv     = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_vals = policy_net(sv).cpu().numpy().tolist()[0]

            # ── Write to shared state (this is what the poll endpoint reads) ──
            training_state["episode"] = episode + 1
            training_state["rewards"] = rewards_per_episode[-200:]
            training_state["epsilon"] = epsilon
            training_state["q_values"] = q_vals if q_vals else training_state["q_values"]

        os.makedirs("./artifacts", exist_ok=True)
        torch.save(policy_net.state_dict(), model_path)
        np.save(reward_path, np.array(rewards_per_episode))

        training_state["done"]    = True
        training_state["running"] = False

    except Exception as e:
        import traceback
        training_state["error"]   = str(e) + "\n" + traceback.format_exc()
        training_state["running"] = False
        training_state["done"]    = True


@app.route("/api/training_status")
def training_status():
    return jsonify({
        "running": training_state["running"],
        "episode": training_state["episode"],
        "total_episodes": training_state["total_episodes"],
        "rewards": training_state["rewards"],
        "epsilon": training_state["epsilon"],
        "done": training_state["done"],
        "error": training_state["error"],
        "q_values": training_state["q_values"],
        "device": training_state.get("device", "cpu"),
    })


@app.route("/api/stop_training", methods=["POST"])
def stop_training():
    training_state["running"] = False
    return jsonify({"status": "stopped"})


@app.route("/api/start_evaluation", methods=["POST"])
def start_evaluation():
    global eval_state

    if eval_state["running"]:
        return jsonify({"error": "Evaluation already running"}), 400

    cfg = request.json

    eval_state = {
        "running": True,
        "done": False,
        "error": None,
        "progress": 0,
        "results": {},
        "step_metrics": {
            "dqn": [],
            "random": [],
            "round_robin": [],
            "least_connections": [],
        },
    }

    thread = threading.Thread(target=run_evaluation, args=(cfg,), daemon=True)
    thread.start()

    return jsonify({"status": "started"})


def run_evaluation(cfg):
    global eval_state

    try:
        # Pull all values directly from cfg
        episodes     = int(cfg.get("TEST_EPISODE_COUNT", 50))
        max_steps    = int(cfg.get("EPISODE_LENGTH", 500))
        autoscale    = bool(cfg.get("AUTOSCALING", True))
        max_servers  = int(cfg.get("MAX_SERVER_COUNT", 15))
        min_servers  = int(cfg.get("MIN_SERVER_COUNT", 5))
        server_count = int(cfg.get("SERVER_COUNT", 10))
        sla_thresh   = float(cfg.get("SLA_VIOLATION_LATENCY", 80))
        hidden_dim   = int(cfg.get("HIDDEN_DIM", 128))
        model_path   = "./artifacts/dqn_trained_model.pth"

        avg_req    = int(cfg.get("AVG_INCOMING_REQUESTS", 5))
        avg_wl     = int(cfg.get("AVG_REQUEST_WORKLOAD", 10))
        wl_var     = int(cfg.get("REQUEST_WORKLOAD_VARIANCE", 5))
        max_util   = int(cfg.get("MAX_SERVER_UTILIZATION", 100))
        proc_power = int(cfg.get("PROCESSING_POWER", 10))
        max_queue  = int(cfg.get("MAX_QUEUE_LENGTH", 200))

        from cloud_env_simulator import CloudEnvironment
        from dqn_agent import DQN
        import torch

        # Patch the module-level name bindings directly (C.X = v is useless
        # because 'from configuration import X' already copied the value)
        import cloud_env_simulator as _cem
        _cem.AVG_INCOMING_REQUESTS    = avg_req
        _cem.AVG_REQUEST_WORKLOAD     = avg_wl
        _cem.REQUEST_WORKLOAD_VARIANCE = wl_var
        _cem.MAX_QUEUE_LENGTH         = max_queue
        _cem.SLA_VIOLATION_LATENCY    = sla_thresh

        action_size = max_servers + 2 if autoscale else max_servers

        methods = ["dqn", "random", "round_robin", "least_connections"]
        all_results = {m: [] for m in methods}
        all_metrics = {m: {
            "avg_latency": [], "sla_violations": [], "cpu_utilization": [],
            "queue_lengths": [], "num_servers": [], "rewards": []
        } for m in methods}

        total_runs = len(methods) * episodes
        completed = 0

        for method in methods:
            # Setup — pass all params explicitly
            env = CloudEnvironment(
                num_servers=server_count,
                max_servers=max_servers,
                min_servers=min_servers,
                autoscale=autoscale
            )
            env.server_capacity = max_util
            env.processing_rate = proc_power

            # Load DQN agent if needed
            dqn_net = None
            dqn_device = torch.device("cpu")
            if method == "dqn":
                state_size = len(env.reset())
                dqn_net = DQN(state_size, action_size).to(dqn_device)
                dqn_net.load_state_dict(
                    torch.load(model_path, map_location=dqn_device, weights_only=True)
                )
                dqn_net.eval()

            # Baseline controllers
            rr_pointer = [0]

            for ep in range(episodes):
                if not eval_state["running"]:
                    break

                state = env.reset()
                total_reward = 0
                ep_latencies = []
                ep_sla = 0
                ep_cpu = []
                ep_queues = []
                ep_servers = []   # track server count every step, not just at end

                for step in range(max_steps):
                    if method == "dqn":
                        with torch.no_grad():
                            s_t = torch.FloatTensor(state).unsqueeze(0).to(dqn_device)
                            action = torch.argmax(dqn_net(s_t)).item()
                    elif method == "random":
                        action = np.random.randint(0, env.num_servers)
                    elif method == "round_robin":
                        action = rr_pointer[0] % env.num_servers
                        rr_pointer[0] += 1
                    elif method == "least_connections":
                        action = int(np.argmin(env.queues[:env.num_servers]))

                    next_state, reward, done = env.step(action)

                    avg_lat = float(np.mean(env.queues[:env.num_servers]))
                    avg_cpu = float(np.mean(env.cpu_usage[:env.num_servers]))
                    sla_viol = 1 if avg_lat > float(cfg.get("SLA_VIOLATION_LATENCY", 80)) else 0

                    ep_latencies.append(avg_lat)
                    ep_sla += sla_viol
                    ep_cpu.append(avg_cpu)
                    ep_queues.append(avg_lat)
                    ep_servers.append(env.num_servers)   # sample after step so scaling is captured
                    total_reward += reward

                    state = next_state
                    if done:
                        break

                all_results[method].append(total_reward)
                all_metrics[method]["avg_latency"].append(float(np.mean(ep_latencies)))
                all_metrics[method]["sla_violations"].append(ep_sla)
                all_metrics[method]["cpu_utilization"].append(float(np.mean(ep_cpu)))
                all_metrics[method]["queue_lengths"].append(float(np.mean(ep_queues)))
                all_metrics[method]["num_servers"].append(float(np.mean(ep_servers)))  # true per-step avg
                all_metrics[method]["rewards"].append(total_reward)

                completed += 1
                eval_state["progress"] = int((completed / total_runs) * 100)

                # Stream step metrics every episode
                eval_state["step_metrics"][method] = {
                    "rewards": all_results[method],
                    "avg_latency": all_metrics[method]["avg_latency"],
                    "sla_violations": all_metrics[method]["sla_violations"],
                    "cpu_utilization": all_metrics[method]["cpu_utilization"],
                }

        # Compute summary stats
        summary = {}
        for method in methods:
            rewards = all_results[method]
            latencies = all_metrics[method]["avg_latency"]
            sla = all_metrics[method]["sla_violations"]
            cpu = all_metrics[method]["cpu_utilization"]
            servers = all_metrics[method]["num_servers"]
            summary[method] = {
                "mean_reward": float(np.mean(rewards)) if rewards else 0,
                "std_reward": float(np.std(rewards)) if rewards else 0,
                "mean_latency": float(np.mean(latencies)) if latencies else 0,
                "total_sla_violations": int(np.sum(sla)) if sla else 0,
                "mean_cpu": float(np.mean(cpu)) if cpu else 0,
                "mean_servers": float(np.mean(servers)) if servers else 0,
                "rewards": rewards,
            }

        eval_state["results"] = summary
        eval_state["done"] = True
        eval_state["running"] = False

    except Exception as e:
        import traceback
        eval_state["error"] = str(e) + "\n" + traceback.format_exc()
        eval_state["running"] = False
        eval_state["done"] = True


@app.route("/api/evaluation_status")
def evaluation_status():
    return jsonify({
        "running": eval_state["running"],
        "done": eval_state["done"],
        "error": eval_state["error"],
        "progress": eval_state["progress"],
        "results": eval_state["results"],
        "step_metrics": eval_state["step_metrics"],
    })


@app.route("/api/model_exists")
def model_exists():
    exists = os.path.exists("./artifacts/dqn_trained_model.pth")
    return jsonify({"exists": exists})


@app.route("/api/get_train_config")
def get_train_config():
    return jsonify(train_config if train_config else get_default_config())


@app.route("/api/debug")
def debug_state():
    """Call this from browser: http://localhost:5000/api/debug
    It shows the raw training_state so you can see exactly what's happening."""
    import sys
    return jsonify({
        "training_state": {
            "running":        training_state["running"],
            "episode":        training_state["episode"],
            "total_episodes": training_state["total_episodes"],
            "epsilon":        training_state["epsilon"],
            "done":           training_state["done"],
            "error":          training_state["error"],
            "rewards_count":  len(training_state["rewards"]),
            "last_reward":    training_state["rewards"][-1] if training_state["rewards"] else None,
            "device":         training_state.get("device"),
        },
        "eval_state": {
            "running":  eval_state["running"],
            "done":     eval_state["done"],
            "progress": eval_state["progress"],
            "error":    eval_state["error"],
        },
        "torch_available": torch is not None,
        "torch_version":   str(torch.__version__) if torch else None,
        "cuda_available":  torch.cuda.is_available() if torch else False,
        "train_config_keys": list(train_config.keys()),
        "python":          sys.version,
    })


if __name__ == "__main__":
    os.makedirs("./artifacts", exist_ok=True)
    app.run(debug=True, threaded=True, port=5000)
