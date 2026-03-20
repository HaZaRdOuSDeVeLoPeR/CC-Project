import os, json, threading, traceback
import numpy as np
import torch
from flask import Flask, render_template, jsonify, request
from configuration import *

app = Flask(__name__)

# ── Shared state (written by background thread, read by poll endpoint) ────────
training_state = {
    "running": False, "episode": 0, "total_episodes": 500,
    "rewards": [], "epsilon": 1.0, "done": False, "error": None,
    "q_values": [], "q_history": [], "q_meta": None, "device": "cpu",
}
eval_state = {
    "running": False, "progress": 0, "total": 0,
    "done": False, "error": None, "results": None,
    "step_metrics": {}, "dqn_action_dist": {},
    "dqn_action_size": 0,
}

DEFAULT_CFG = {
    "SERVER_COUNT": SERVER_COUNT,
    "AVG_INCOMING_REQUESTS": AVG_INCOMING_REQUESTS,
    "AVG_REQUEST_WORKLOAD": AVG_REQUEST_WORKLOAD,
    "REQUEST_WORKLOAD_VARIANCE": REQUEST_WORKLOAD_VARIANCE,
    "MAX_SERVER_UTILIZATION": MAX_SERVER_UTILIZATION,
    "PROCESSING_POWER": PROCESSING_POWER,
    "MAX_QUEUE_LENGTH": MAX_QUEUE_LENGTH,
    "SLA_VIOLATION_LATENCY": SLA_VIOLATION_LATENCY,
    "EPISODE_LENGTH": EPISODE_LENGTH,
    "TRAIN_EPISODE_COUNT": TRAIN_EPISODE_COUNT,
    "TEST_EPISODE_COUNT": TEST_EPISODE_COUNT,
    "GAMMA": GAMMA,
    "LEARNING_RATE": LEARNING_RATE,
    "BATCH_SIZE": BATCH_SIZE,
    "REPLAY_BUFFER_SIZE": REPLAY_BUFFER_SIZE,
    "EPSILON_START": EPSILON_START,
    "EPSILON_MIN": EPSILON_MIN,
    "TARGET_UPDATE_STEPS": TARGET_UPDATE_STEPS,
    "HIDDEN_DIM": HIDDEN_DIM,
    "DEVICE": DEVICE
}

# ── Load persisted training config on startup ─────────────────────────────────
_saved_cfg_path = "./artifacts/last_train_config.json"
if os.path.exists(_saved_cfg_path):
    try:
        with open(_saved_cfg_path) as f:
            _saved = json.load(f)
        DEFAULT_CFG.update(_saved)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(cfg):
    global training_state
    try:
        episodes    = int(cfg.get("TRAIN_EPISODE_COUNT", 500))
        max_steps   = int(cfg.get("EPISODE_LENGTH", 500))
        n_servers   = int(cfg.get("SERVER_COUNT", 10))
        device_str  = str(cfg.get("DEVICE", "cpu"))
        gamma       = float(cfg.get("GAMMA", 0.99))
        lr          = float(cfg.get("LEARNING_RATE", 0.0005))
        batch_size  = int(cfg.get("BATCH_SIZE", 128))
        buf_size    = int(cfg.get("REPLAY_BUFFER_SIZE", 10000))
        eps_start   = float(cfg.get("EPSILON_START", 1.0))
        eps_min     = float(cfg.get("EPSILON_MIN", 0.05))
        target_upd  = int(cfg.get("TARGET_UPDATE_STEPS", 500))
        hidden_dim  = int(cfg.get("HIDDEN_DIM", 128))
        avg_req     = int(cfg.get("AVG_INCOMING_REQUESTS", 5))
        avg_wl      = int(cfg.get("AVG_REQUEST_WORKLOAD", 10))
        wl_var      = int(cfg.get("REQUEST_WORKLOAD_VARIANCE", 5))
        max_util    = int(cfg.get("MAX_SERVER_UTILIZATION", 100)) or 100
        proc_power  = int(cfg.get("PROCESSING_POWER", 10)) or 10
        max_queue   = int(cfg.get("MAX_QUEUE_LENGTH", 200))
        sla_thresh  = int(cfg.get("SLA_VIOLATION_LATENCY", 80))

        # Patch module-level constants so CloudEnvironment reads them correctly
        import cloud_env_simulator as _cem
        _cem.AVG_INCOMING_REQUESTS    = avg_req
        _cem.AVG_REQUEST_WORKLOAD     = avg_wl
        _cem.REQUEST_WORKLOAD_VARIANCE = wl_var
        _cem.MAX_QUEUE_LENGTH         = max_queue
        _cem.SLA_VIOLATION_LATENCY    = sla_thresh
        _cem.EPISODE_LENGTH           = max_steps

        from cloud_env_simulator import CloudEnvironment
        from dqn_agent import DQNAgent

        env = CloudEnvironment(num_servers=n_servers, avg_proc_rate=proc_power)
        env.server_capacity = max_util

        state_size  = len(env.reset())
        action_size = len(env.action_map)
        training_state["action_map"] = [[a, b] for a, b in env.action_map]
        training_state["n_servers"]  = n_servers

        agent = DQNAgent(
            state_size=state_size, action_size=action_size,
            hidden_dim=hidden_dim, gamma=gamma, lr=lr,
            batch_size=batch_size, buffer_size=buf_size,
            epsilon_start=eps_start, epsilon_min=eps_min,
            n_episodes=episodes, target_update_steps=target_upd,
            device=device_str,
        )

        rewards_per_episode = []

        for episode in range(episodes):
            if not training_state["running"]:
                break
            state        = env.reset()
            total_reward = 0.0
            for _ in range(max_steps):
                action                   = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                agent.train_step()
                state         = next_state
                total_reward += reward
                if done:
                    break

            rewards_per_episode.append(total_reward)
            agent.decay_epsilon()

            # Q-value snapshot every episode for smooth chart updates
            if True:
                with torch.no_grad():
                    sv    = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    raw_q = agent.policy_net(sv).cpu().numpy()[0]
                q_max    = float(raw_q.max())
                q_min    = float(raw_q.min())
                best_act = int(raw_q.argmax())
                amap = env.action_map
                best_pair = amap[best_act] if best_act < len(amap) else (best_act, best_act)
                snapshot = {
                    "ep": episode + 1, "max": round(q_max, 3),
                    "min": round(q_min, 3), "mean": round(float(raw_q.mean()), 3),
                    "spread": round(q_max - q_min, 3), "best": best_act,
                    "best_a": best_pair[0], "best_b": best_pair[1],
                    "routing": True,
                }
                hist = training_state.get("q_history", [])
                hist.append(snapshot)
                if len(hist) > 60:
                    hist = hist[-60:]
                training_state["q_history"] = hist
                training_state["q_values"]  = raw_q.tolist()
                training_state["q_meta"]    = snapshot

            training_state["episode"]  = episode + 1
            training_state["rewards"]  = rewards_per_episode[-200:]
            training_state["epsilon"]  = agent.epsilon

        # Save model and metadata
        os.makedirs("./artifacts", exist_ok=True)
        torch.save(agent.policy_net.state_dict(), "./artifacts/dqn_trained_model.pth")
        np.save("./artifacts/training_rewards.npy", np.array(rewards_per_episode))

        meta = {"state_size": state_size, "action_size": action_size,
                "num_servers": n_servers, "hidden_dim": hidden_dim, "device": device_str}
        with open("./artifacts/dqn_model_meta.json", "w") as f:
            json.dump(meta, f)

        with open(_saved_cfg_path, "w") as f:
            json.dump(cfg, f)

        training_state["done"]    = True
        training_state["running"] = False

    except Exception as e:
        training_state["error"]   = str(e) + "\n" + traceback.format_exc()
        training_state["running"] = False
        training_state["done"]    = True


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(cfg):
    global eval_state
    try:
        episodes   = int(cfg.get("TEST_EPISODE_COUNT", 300))
        max_steps  = int(cfg.get("EPISODE_LENGTH", 500))
        n_servers  = int(cfg.get("SERVER_COUNT", 10))
        avg_req    = int(cfg.get("AVG_INCOMING_REQUESTS", 5))
        avg_wl     = int(cfg.get("AVG_REQUEST_WORKLOAD", 10))
        wl_var     = int(cfg.get("REQUEST_WORKLOAD_VARIANCE", 5))
        max_util   = int(cfg.get("MAX_SERVER_UTILIZATION", 100)) or 100   # never 0
        proc_power = int(cfg.get("PROCESSING_POWER", 10)) or 10           # never 0
        max_queue  = int(cfg.get("MAX_QUEUE_LENGTH", 200))
        sla_thresh = int(cfg.get("SLA_VIOLATION_LATENCY", 80))

        model_path = "./artifacts/dqn_trained_model.pth"
        meta_path  = "./artifacts/dqn_model_meta.json"

        # Load model — derive architecture from weights (ground truth)
        weights    = torch.load(model_path, map_location="cpu", weights_only=True)
        state_size = int(weights["network.0.weight"].shape[1])
        action_size= int(weights["network.4.weight"].shape[0])
        hidden_dim = int(weights["network.0.weight"].shape[0])
        num_servers_from_model = int((np.sqrt(8 * action_size + 1) - 1) / 2)  # action_size == num_servers

        # Patch env constants
        import cloud_env_simulator as _cem
        _cem.AVG_INCOMING_REQUESTS    = avg_req
        _cem.AVG_REQUEST_WORKLOAD     = avg_wl
        _cem.REQUEST_WORKLOAD_VARIANCE = wl_var
        _cem.MAX_QUEUE_LENGTH         = max_queue
        _cem.SLA_VIOLATION_LATENCY    = sla_thresh
        _cem.EPISODE_LENGTH           = max_steps

        from cloud_env_simulator import CloudEnvironment
        from dqn_agent import DQN

        # Validate server count matches model
        if n_servers != num_servers_from_model:
            raise RuntimeError(
                f"SERVER_COUNT mismatch: eval uses {n_servers} servers but model "
                f"was trained with {num_servers_from_model}. Retrain or match the config."
            )
        # Validate state size matches new formula: N*3+2
        expected_state_size = n_servers * 3 + 2
        if state_size != expected_state_size:
            raise RuntimeError(
                f"State size mismatch: model has {state_size} inputs but env produces "
                f"{expected_state_size}. Delete artifacts and retrain."
            )

        dqn_net = DQN(state_size, action_size, hidden_dim)
        dqn_net.load_state_dict(weights)
        dqn_net.eval()
        # Build action_map for label decoding in frontend
        from cloud_env_simulator import CloudEnvironment as _CE
        _env_tmp = _CE(num_servers=n_servers)
        eval_state["action_map"] = [[a, b] for a, b in _env_tmp.action_map]

        methods = ["dqn", "random", "round_robin", "least_connections"]
        all_results = {m: [] for m in methods}
        all_metrics = {m: {"avg_latency": [], "sla_violations": [], "cpu_utilization": [],
                           "rewards": []} for m in methods}
        action_counts = {i: 0 for i in range(action_size)}

        total_runs = len(methods) * episodes
        completed  = 0

        for method in methods:
            rr_counter = 0
            for ep in range(episodes):
                if not eval_state["running"]:
                    break

                env = CloudEnvironment(num_servers=n_servers, avg_proc_rate=proc_power)
                env.server_capacity = max_util

                state         = env.reset()
                total_reward  = 0.0
                ep_latency    = []
                ep_sla        = 0
                ep_cpu        = []

                for step in range(max_steps):
                    queues = env.queues[:n_servers]
                    cpu    = env.cpu_usage[:n_servers]

                    if method == "dqn":
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0)
                            q = dqn_net(state_tensor).cpu().numpy()[0]
                        action = int(np.argmax(q))
                        action_counts[action] = action_counts.get(action, 0) + 1

                    elif method == "random":
                        action = np.random.randint(len(env.action_map))

                    elif method == "round_robin":
                        action = rr_counter % len(env.action_map)
                        rr_counter += 1

                    else:  # least_connections
                        queues = env.queues
                        idx = np.argsort(queues)[:2]  # best 2 servers

                        i, j = int(idx[0]), int(idx[1])
                        pair = (min(i, j), max(i, j))

                        action = env.action_map.index(pair)

                    state, reward, done = env.step(action)
                    total_reward += reward
                    avg_lat = float(np.mean(env.queues[:n_servers]))
                    ep_latency.append(avg_lat)
                    cpu = env.cpu_usage[:n_servers]
                    ep_cpu.append(float(np.mean(cpu)))

                    if avg_lat > sla_thresh:
                        ep_sla += 1
                    if done:
                        break

                all_results[method].append(total_reward)
                all_metrics[method]["avg_latency"].append(np.mean(ep_latency))
                all_metrics[method]["sla_violations"].append(ep_sla)
                all_metrics[method]["cpu_utilization"].append(np.mean(ep_cpu))
                all_metrics[method]["rewards"].append(total_reward)

                completed += 1
                eval_state["progress"] = completed
                eval_state["step_metrics"] = {
                    m: {
                        "rewards":        all_metrics[m]["rewards"],
                        "avg_latency":    all_metrics[m]["avg_latency"],
                        "sla_violations": all_metrics[m]["sla_violations"],
                        "cpu_utilization":all_metrics[m]["cpu_utilization"],
                    } for m in methods
                }

        def summarise(m):
            r = all_results[m]
            l = all_metrics[m]["avg_latency"]
            s = all_metrics[m]["sla_violations"]
            c = all_metrics[m]["cpu_utilization"]
            return {
                "mean_reward": round(float(np.mean(r)), 1),
                "std_reward":  round(float(np.std(r)),  1),
                "avg_latency": round(float(np.mean(l)), 2),
                "sla_violations": int(np.sum(s)),
                "avg_cpu":     round(float(np.mean(c)) * 100, 1),
                "avg_servers": n_servers,
            }

        eval_state["results"]        = {m: summarise(m) for m in methods}
        eval_state["dqn_action_dist"]= action_counts
        eval_state["dqn_action_size"]= action_size
        eval_state["done"]           = True
        eval_state["running"]        = False

        with open("./artifacts/dqn_model_meta.json", "w") as f:
            json.dump({"state_size": state_size, "action_size": action_size,
                       "num_servers": num_servers_from_model, "hidden_dim": hidden_dim}, f)

    except Exception as e:
        eval_state["error"]   = str(e) + "\n" + traceback.format_exc()
        eval_state["running"] = False
        eval_state["done"]    = True


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    model_exists = os.path.exists("./artifacts/dqn_trained_model.pth")
    return render_template("index.html", model_exists=model_exists)

@app.route("/train")
def train_page():
    return render_template("train.html", config=DEFAULT_CFG)

@app.route("/evaluate")
def evaluate_page():
    return render_template("evaluate.html")

@app.route("/api/start_training", methods=["POST"])
def start_training():
    global training_state
    if training_state["running"]:
        return jsonify({"status": "already_running"})
    cfg = request.json or {}
    training_state = {
        "running": True, "episode": 0,
        "total_episodes": int(cfg.get("TRAIN_EPISODE_COUNT", 500)),
        "rewards": [], "epsilon": float(cfg.get("EPSILON_START", 1.0)),
        "done": False, "error": None, "q_values": [], "q_history": [],
        "q_meta": None, "device": str(cfg.get("DEVICE", "cpu")),
        "action_map": [], "n_servers": int(cfg.get("SERVER_COUNT", 10)),
    }
    threading.Thread(target=run_training, args=(cfg,), daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/api/training_status")
def training_status():
    return jsonify({
        "running":        training_state["running"],
        "episode":        training_state["episode"],
        "total_episodes": training_state["total_episodes"],
        "rewards":        training_state["rewards"],
        "epsilon":        training_state["epsilon"],
        "done":           training_state["done"],
        "error":          training_state["error"],
        "q_values":       training_state.get("q_values", []),
        "q_history":      training_state.get("q_history", []),
        "q_meta":         training_state.get("q_meta"),
        "device":         training_state.get("device", "cpu"),
        "action_map":     training_state.get("action_map", []),
        "n_servers":      training_state.get("n_servers", 10),
    })

@app.route("/api/stop_training", methods=["POST"])
def stop_training():
    training_state["running"] = False
    return jsonify({"status": "stopping"})

@app.route("/api/start_evaluation", methods=["POST"])
def start_evaluation():
    global eval_state
    if eval_state["running"]:
        return jsonify({"status": "already_running"})
    cfg = request.json or {}
    episodes_req = int(cfg.get("TEST_EPISODE_COUNT", 300))
    total = episodes_req * 4  # 4 methods
    eval_state = {
        "running": True, "progress": 0, "total": total,
        "done": False, "error": None, "results": None,
        "step_metrics": {}, "dqn_action_dist": {}, "dqn_action_size": 0,
        "action_map": [],
    }
    threading.Thread(target=run_evaluation, args=(cfg,), daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/api/evaluation_status")
def evaluation_status():
    return jsonify({
        "running":         eval_state["running"],
        "progress":        eval_state["progress"],
        "total":           eval_state["total"],
        "done":            eval_state["done"],
        "error":           eval_state["error"],
        "results":         eval_state["results"],
        "step_metrics":    eval_state["step_metrics"],
        "dqn_action_dist": eval_state["dqn_action_dist"],
        "dqn_action_size": eval_state["dqn_action_size"],
        "action_map":      eval_state.get("action_map", []),
    })

@app.route("/api/devices")
def get_devices():
    devices = [{"id": "cpu", "label": "CPU", "available": True, "info": "Always available"}]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.insert(0, {
                "id": f"cuda:{i}", "label": f"GPU {i}: {props.name}",
                "available": True,
                "info": f"{props.total_memory // 1024**2} MB VRAM"
            })
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.insert(0 if not any(d["id"].startswith("cuda") for d in devices) else 1,
                       {"id": "mps", "label": "Apple MPS", "available": True,
                        "info": "Apple Silicon GPU"})
    return jsonify(devices)

@app.route("/api/model_meta")
def model_meta():
    model_path = "./artifacts/dqn_trained_model.pth"
    if not os.path.exists(model_path):
        return jsonify({"has_meta": False})
    try:
        w = torch.load(model_path, map_location="cpu", weights_only=True)
        state_size  = int(w["network.0.weight"].shape[1])
        action_size = int(w["network.4.weight"].shape[0])
        num_servers = int((np.sqrt(8 * action_size + 1) - 1) / 2)  # action_size == num_servers
        # state_size = num_servers * 3 + 2  (cpu, queues, proc_rates, avg_lat, mean_cpu)
        return jsonify({
            "has_meta":    True,
            "state_size":  state_size,
            "action_size": action_size,
            "num_servers": num_servers,
            "hidden_dim":  int(w["network.0.weight"].shape[0]),
        })
    except Exception as e:
        return jsonify({"has_meta": False, "error": str(e)})

@app.route("/api/get_train_config")
def get_train_config():
    return jsonify(DEFAULT_CFG)

if __name__ == "__main__":
    app.run(debug=False, port=5000, host="0.0.0.0", threaded=True)
