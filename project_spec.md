# Super Mario Bros — Reinforcement Learning Project Spec

## 1. Overview

Train a Deep Q-Network (DQN) agent to play **Super Mario Bros** (NES) using the
`gym-super-mario-bros` environment. The project is structured as a **learning-first**
endeavour — every module is designed to teach a core RL concept while producing a
working agent that can clear World 1-1.

### Success Criteria

| Milestone | Definition |
|-----------|-----------|
| **M1 — Loop works** | Agent takes random actions, env renders, reward is logged |
| **M2 — Agent learns** | Mean reward increases over 1 000 episodes |
| **M3 — Clears 1-1** | Agent consistently reaches the flagpole on `SuperMarioBros-1-1-v0` |
| **M4 — Polish** | Training dashboard, saved model, gameplay recording |

---

## 2. Environment

| Detail | Value |
|--------|-------|
| **Package** | [`gym-super-mario-bros`](https://github.com/Kautenja/gym-super-mario-bros) |
| **Env ID** | `SuperMarioBros-1-1-v0` (single level to start) |
| **Action space** | `SIMPLE_MOVEMENT` (7 discrete actions: NOOP, right, right+A, right+B, right+A+B, A, left) via `JoypadSpace` wrapper |
| **Observation** | 240 × 256 × 3 RGB frames |
| **Reward** | Built-in shaped reward (positive for moving right, penalty for dying / time) |
| **Environment Constraints** | `gym==0.21.0`, `stable-baselines3==1.8.0` (compatible with 4-variable step API) |

### Wrappers (applied in order)

The raw environment is **not directly usable** for a CNN-based DQN. We need a
standard Atari-style preprocessing pipeline:

```
Raw env (240×256×3 RGB)
  │
  ├─ 1. JoypadSpace(SIMPLE_MOVEMENT)     # Reduce action space from 256 → 7
  ├─ 2. GrayScaleObservation()            # RGB → grayscale (H×W×1)
  ├─ 3. ResizeObservation(84)             # Downsample to 84×84
  ├─ 4. FrameStack(4)                     # Stack 4 consecutive frames → (4, 84, 84)
  └─ 5. SkipFrame(skip=4)                # Agent acts every 4th frame, repeating action
```

**Why each wrapper matters (learn this!):**

- **Grayscale + Resize**: Cuts input size ~27× (240×256×3 → 84×84×1). Color carries
  almost no decision-relevant info in Mario.
- **Frame stacking**: A single frame is ambiguous — the agent can't tell if Mario is
  moving left or right, or if an enemy is approaching. 4 stacked frames encode
  velocity and trajectory.
- **Frame skipping**: Human reaction time is ~200 ms. Deciding every single frame
  (16 ms) wastes compute and makes credit assignment harder. Acting every 4th frame
  is the standard trick from the original DQN paper.

---

## 3. Algorithm — Deep Q-Network (DQN)

### 3.1 Core Idea

DQN approximates the optimal action-value function Q*(s, a) with a neural network.
Instead of a Q-table (infeasible for image observations), we train a CNN to map
`state → Q-values for each action`, then act greedily.

### 3.2 Key Components

```
┌──────────────────────────────────────────────────────┐
│                    DQN Agent                         │
│                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐  │
│  │  Policy   │    │   Replay     │    │  Target    │  │
│  │  Network  │◄───│   Buffer     │───►│  Network   │  │
│  │  (θ)      │    │  (experience │    │  (θ⁻)      │  │
│  └──────────┘    │   memory)    │    └────────────┘  │
│       │          └──────────────┘         │          │
│       │                                   │          │
│       └─── TD loss = (r + γ max Q(s',a'; θ⁻) - Q(s,a; θ))²  │
└──────────────────────────────────────────────────────┘
```

| Component | Purpose |
|-----------|---------|
| **Policy network** | CNN that outputs Q(s, a) for all actions. Used to **select** actions. |
| **Target network** | Frozen copy of policy network, updated every `N` steps. Provides **stable** TD targets. Without it, training diverges. |
| **Replay buffer** | Stores `(s, a, r, s', done)` transitions. Sampling random mini-batches breaks correlation between consecutive frames. |
| **ε-greedy exploration** | With probability ε, take a random action; otherwise take argmax Q. ε decays over training from ~1.0 → 0.01. |

### 3.3 CNN Architecture

```
Input: (batch, 4, 84, 84)  ← 4 stacked grayscale frames

Conv2d(4, 32, kernel=8, stride=4)  → ReLU → (batch, 32, 20, 20)
Conv2d(32, 64, kernel=4, stride=2) → ReLU → (batch, 64, 9, 9)
Conv2d(64, 64, kernel=3, stride=1) → ReLU → (batch, 64, 7, 7)
Flatten                             →        (batch, 3136)
Linear(3136, 512)                  → ReLU → (batch, 512)
Linear(512, n_actions)             →        (batch, 7)
```

> This is the exact architecture from the **Nature DQN paper** (Mnih et al., 2015).
> It's the gold standard starting point — don't over-engineer it.

### 3.4 Hyperparameters

| Hyperparameter | Value | Notes |
|---------------|-------|-------|
| Learning rate | `2.5e-4` | Adam optimizer |
| Discount (γ) | `0.99` | Standard for episodic tasks |
| Replay buffer size | `100_000` | ~5 GB of transitions. Reduce if RAM-constrained. |
| Batch size | `32` | Sampled from replay buffer each training step |
| Target network sync | Every `10_000` steps | Hard copy θ → θ⁻ |
| ε start | `1.0` | Full exploration initially |
| ε end | `0.02` | Small residual exploration |
| ε decay | Linear over `100_000` steps | Or exponential, your choice |
| Training starts | After `10_000` steps | Fill buffer before learning |
| Train frequency | Every `4` steps | Don't train on every single frame |

---

## 4. Project Structure

```
Super-Mario-RL/
├── project_spec.md          # This file
├── README.md                # Project overview + results
│
├── src/
│   ├── __init__.py
│   ├── environment.py       # Env creation + wrapper pipeline
│   ├── agent.py             # DQN agent (action selection, learning, saving/loading)
│   ├── network.py           # CNN architecture (policy + target nets)
│   ├── replay_buffer.py     # Experience replay buffer
│   ├── train.py             # Main training loop + logging
│   └── config.py            # All hyperparameters in one place
│
├── notebooks/
│   └── analysis.ipynb       # Training curve plots, reward analysis
│
├── checkpoints/             # Saved model weights (.pt files)
│   └── .gitkeep
│
├── logs/                    # TensorBoard / CSV logs
│   └── .gitkeep
│
├── recordings/              # Gameplay videos (.mp4)
│   └── .gitkeep
│
├── requirements.txt         # Python dependencies
└── .gitignore
```

### Module Responsibilities

| Module | Responsibility | Key RL concept it teaches |
|--------|---------------|--------------------------|
| `environment.py` | Build the env, apply wrappers, handle render/record | **Observation preprocessing** — why raw pixels aren't usable |
| `replay_buffer.py` | Store transitions, sample mini-batches | **Experience replay** — breaking temporal correlation |
| `network.py` | Define CNN, forward pass | **Function approximation** — replacing Q-tables with neural nets |
| `agent.py` | ε-greedy action selection, compute TD loss, sync target net | **Temporal difference learning**, **exploration vs exploitation** |
| `train.py` | Episode loop, logging, checkpointing | **Training loop engineering** — the practical side of RL |
| `config.py` | Central hyperparameter store | **Hyperparameter sensitivity** — RL is notoriously brittle |

---

## 5. Training Pipeline

```python
for episode in range(num_episodes):
    state = env.reset()

    while not done:
        # 1. SELECT ACTION (ε-greedy)
        action = agent.select_action(state)

        # 2. STEP ENVIRONMENT
        next_state, reward, done, info = env.step(action)

        # 3. STORE TRANSITION
        agent.replay_buffer.push(state, action, reward, next_state, done)

        # 4. LEARN (if buffer has enough samples)
        if len(agent.replay_buffer) >= batch_size and step % train_freq == 0:
            loss = agent.learn()

        # 5. SYNC TARGET NETWORK
        if step % target_sync == 0:
            agent.sync_target()

        # 6. DECAY EPSILON
        agent.decay_epsilon()

        state = next_state
        step += 1

    # Log episode reward, length, epsilon, loss
```

### Logging (what to track)

| Metric | Why |
|--------|-----|
| Episode reward (total) | Primary performance signal |
| Episode length | Longer = Mario gets further |
| Epsilon value | Verify exploration is decaying |
| TD loss | Should decrease, but RL loss is noisy — don't panic |
| Max Q-value | If it diverges to ±∞, something is wrong |
| `x_pos` from `info` dict | Direct measure of how far Mario gets |

---

## 6. Dependencies & Environment Constraints

```
# requirements.txt
gym==0.21.0                    # Gym API (pinned by gym-super-mario-bros)
gym-super-mario-bros==7.4.0    # Mario environment
nes-py                         # NES emulator backend
stable-baselines3==1.8.0       # RL library (compatible with 4-variable step API)
opencv-python                  # Image transforms / preprocessing
torch                          # Neural network framework
tensorboard                    # Training visualization
```

> **Note on compatibility & environment constraints**: `gym-super-mario-bros` uses the old `gym` API (not
> `gymnasium`) with the classic 4-variable step API (`obs, reward, done, info`). Pin `gym==0.21.0` and
> `stable-baselines3==1.8.0` to ensure direct compatibility with this 4-variable step format without requiring shims.

### Installation Guide (Windows & Kaggle)

To bypass build errors:

Step 1: pip install setuptools==65.5.0 wheel==0.38.4
Step 2: pip install -r requirements.txt --no-build-isolation

---

## 7. Development Phases

### Phase 1 — Environment & Random Agent (Day 1)
- [ ] Set up virtual environment, install dependencies
- [ ] Verify `gym-super-mario-bros` runs and renders
- [ ] Implement wrapper pipeline in `environment.py`
- [ ] Run a random agent for 100 episodes, log rewards
- [ ] **Checkpoint**: You can see Mario stumbling around randomly

### Phase 2 — Core DQN Components (Days 2–3)
- [ ] Implement `replay_buffer.py` with `push()` and `sample()`
- [ ] Implement CNN in `network.py`
- [ ] Implement `agent.py` with ε-greedy selection and `learn()`
- [ ] Unit-test each component in isolation
- [ ] **Checkpoint**: Training loop runs without crashing

### Phase 3 — Training & Debugging (Days 4–6)
- [ ] Train for 10k episodes (~6-12 hours on GPU)
- [ ] Monitor TensorBoard: reward curve, loss, Q-values, epsilon
- [ ] Debug if reward flatlines (common issues below)
- [ ] Tune hyperparameters if needed
- [ ] **Checkpoint**: Mean reward is increasing, agent moves right consistently

### Phase 4 — Polish & Documentation (Day 7)
- [ ] Save best model checkpoint
- [ ] Record gameplay video of trained agent
- [ ] Plot training curves in notebook
- [ ] Write README with results, architecture diagram, what you learned
- [ ] **Checkpoint**: Portfolio-ready project

---

## 8. Common Pitfalls & Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Reward stays flat at ~0 | ε not decaying, or decaying too fast | Print ε every 100 episodes. It should reach ~0.1 by episode 5k. |
| Reward spikes then collapses | Target net sync too frequent | Increase `target_sync` interval (try 10k → 30k) |
| Q-values explode (NaN/Inf) | Learning rate too high, or reward not clipped | Clip rewards to [-1, 1] or reduce LR to 1e-4 |
| Training is extremely slow | No GPU, or wrappers missing SkipFrame | Check `torch.cuda.is_available()`. Ensure SkipFrame(4) is applied. |
| `env.step()` returns wrong shapes | Wrapper order is wrong | Wrappers are applied bottom-up. `SkipFrame` should be outermost. |
| Agent always goes right | That's actually correct! The reward incentivizes rightward movement. But check it also jumps. |
| Out of memory | Replay buffer too large | Reduce buffer to 50k or store observations as `uint8` (not float32) |

---

## 9. Stretch Goals (after M4)

Once the base DQN works, these are natural next steps — pick based on interest:

| Extension | What you'll learn |
|-----------|------------------|
| **Double DQN** | Overestimation bias in Q-learning. Change ~5 lines of code. |
| **Dueling DQN** | Separate state-value and advantage streams. Architecture change only. |
| **Prioritized Experience Replay** | Not all transitions are equally useful. Weighted sampling. |
| **Multi-level training** | Generalization — can one agent play multiple levels? |
| **Reward shaping experiments** | Custom rewards (e.g., bonus for coins, penalty for standing still) |
| **PPO comparison** | On-policy vs off-policy: completely different training dynamics |

---

## 10. References

1. **Mnih et al., 2015** — "Human-level control through deep reinforcement learning" (Nature DQN paper)
   - [Paper](https://www.nature.com/articles/nature14236) — Read sections 1–4 at minimum
2. **gym-super-mario-bros docs** — https://github.com/Kautenja/gym-super-mario-bros
3. **PyTorch DQN tutorial** — https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
4. **Spinning Up in Deep RL** (OpenAI) — https://spinningup.openai.com/ — Best conceptual intro