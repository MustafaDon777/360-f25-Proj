# 360-f25-Project

# **Resilient Data Routing Algorithms for Next Gen Multi-Agent Connected Autonomous Vehicle Networks**

## Multi-Agent Grid Navigation with Deep Actor-Critic (MARL + A2C)

This project implements a Multi-Agent Reinforcement Learning (MARL) solution for decentralized pathfinding in a dynamic, partially observable Grid World environment. Each agent uses a dedicated Deep Actor-Critic network to navigate to an assigned goal while avoiding static walls, moving obstacles, and inter-agent collisions.

The simulation highlights key concepts in MARL, including independent learning with complex, shared environmental risks, and features a path-planning visualization to inspect the model's intent.

-----

## Key Machine Learning Features

  - **Decentralized Multi-Agent RL:** Each agent possesses its own complete Actor-Critic model (`ActorCritic`) and is trained independently based on its local observation and unique reward signal.

  - **Hybrid Observation Space:** The state space combines local, high-dimensional perceptual data (via CNN) with global, low-dimensional contextual data (via vector input).

  - **Complex Reward Shaping:** The environment utilizes an advanced reward structure designed to:

      - Penalize collisions with walls and allies severely ($-0.5$ to $-1.0$).

      - Provide a large positive reward for reaching the goal ($+50.0$).

      - Implement **Distance Shaping** (small continuous penalty based on distance to the goal: $-0.002 \times \text{distance}$).

      - Apply a **Collision Avoidance Penalty** (a "Fear Penalty" of $-0.2$) for proximity (radius 2) to moving obstacles.

The reward logic is contained within the `step` method of the `GridEnvironment`:

```python
            # --- REWARD CALCULATIONS ---
            r = -0.01 # Base penalty
            # ... (omitted boundary/wall check for brevity)
            
            # 2. Moving Away Penalty (Distance Shaping)
            prev_dist = abs(ag['x']-gx) + abs(ag['y']-gy)
            new_dist = abs(nx-gx) + abs(ny-gy)
            if new_dist > prev_dist: r -= 0.1
            
            # ... (omitted collision checks for brevity)
            
            # Goal Reached Reward
            if nx==gx and ny==gy:
                r = 50.0; ag['done'] = True; d = True
            else:
                # Shaping reward (Distance improvement)
                r += -0.002 * new_dist

            # 4. Fear Penalty (Proximity to Red Circle)
            near_danger = False
            for o in self.obstacles:
                if abs(o.x - ag['x']) <= 2 and abs(o.y - ag['y']) <= 2:
                    near_danger = True; break
            if near_danger:
                r -= 0.2
            
            rews.append(r); dones.append(d)
```

  - **Planning Model Visualization:** The `MARL_Trainer` includes a **Path Prediction** function (`predict_paths`) that performs a 5-step greedy rollout of the current policy, offering real-time insight into the agent's learned navigation intent.

-----

## Model Architecture: ActorCritic

The core of the system is a customized **Actor-Critic Network** implemented using PyTorch. This architecture uses a dual-stream input pipeline to process distinct types of environmental information:

### 1\. Perceptual Stream (CNN)

This stream processes the agent's immediate, local $5 \times 5$ visual field, essential for collision avoidance and navigation around local features.

|Channel|Input Shape (Local View)|Description|
|---|---|---|
|**0**|$5 \times 5$ Grid|Static Walls and Out-of-Bounds (Binary)|
|**1**|$5 \times 5$ Grid|Moving Danger/Obstacles (Binary)|
|**2**|$5 \times 5$ Grid|Other Active Agents (Ally Detection)|

The CNN forward pass is defined as:

```python
class ActorCritic(nn.Module):
    # ... (init function omitted)
    def forward(self, grid, vec):
        # CNN Stream (Grid input)
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten CNN output
        
        # Vector Stream
        v = F.relu(self.fc_vec(vec))
        
        # Merge, Common, and Output Heads
        merged = torch.cat((x, v), dim=1)
        common = F.relu(self.fc_common(merged))
        return self.actor(common), self.critic(common)
```

### 2\. Contextual Stream (Vector)

This stream provides the necessary long-range context for goal-directed navigation. The coordinates are normalized to the range $[0, 1]$.

|Vector Index|Description|
|---|---|
|**0, 1**|Normalized current $(x, y)$ position|
|**2, 3**|Normalized goal $(g_x, g_y)$ position|

This is processed by a simple fully connected layer (`fc_vec`).

### 3\. Common & Output Layers

The outputs of the CNN stream and the Vector stream are concatenated (`merged`) and passed through a shared fully-connected layer (`fc_common`). This leads to the final outputs:

  - **Actor Head:** A linear layer outputting 4 logits, representing the policy for actions: **Up (0), Down (1), Left (2), Right (3)**.

  - **Critic Head:** A single linear output representing the estimated State Value $V(s)$.

-----

## Training Algorithm

The training process uses a batch-free, online **Actor-Critic** method.

1.  **Rollout Collection:** Agents explore the environment using the current policy (sampling from the Categorical distribution defined by the Actor head). State-Action pairs, rewards, and state values are collected for a full episode.

2.  **Return Calculation:** Returns ($R$) are calculated using the discounted cumulative reward approach, often associated with a simple Monte Carlo return or a specific variation of A2C/VPG.

    $$\text{Return}_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

    The return calculation in the `MARL_Trainer`'s `train_loop` uses the mask to handle episode termination:

    ```python
    returns = []
    R = 0
    # Iterate backwards through rewards and masks
    for r, m in zip(reversed(rollouts[i]['rews']), reversed(rollouts[i]['masks'])):
        # R = r + gamma * R * m (m=0 if done, R is reset)
        R = r + gamma * R * m 
        returns.insert(0, R)
    ```

3.  **Advantage Estimation:** The Advantage ($A_t$) is estimated by subtracting the Critic's value estimate from the calculated Return:

    $$A_t = \text{Return}_t - V(s_t)$$

4.  **Loss Optimization:**

      - **Policy Loss (Actor):** The policy is updated to maximize the probability of actions taken that resulted in a positive advantage.

        $$\mathcal{L}_{\text{policy}} = - \log(\pi(a_t|s_t)) \cdot A_t$$

      - **Value Loss (Critic):** The Critic is updated to minimize the mean squared error between its value estimate and the calculated Return.

        $$\mathcal{L}_{\text{value}} = \frac{1}{2} A_t^2$$

      - The total loss is a combination of both: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + 0.5 \cdot \mathcal{L}_{\text{value}}$.

    The loss calculation is implemented as:

    ```python
    adv = returns - values
    # Total loss: Policy loss + 0.5 * Value loss (MSE of advantage)
    loss = -(log_probs * adv.detach()).mean() + 0.5 * adv.pow(2).mean() 
    ```

-----

## Environment and Interaction

The `GridEnvironment` class handles all simulation logic:

  - **State:** Defined by the position of agents, goals, walls, and moving obstacles.

  - **Dynamics:** Obstacles move slowly along the X-axis, adding a non-stationary element to the environment.

  - **Decentralized Task:** Each agent $i$ is assigned a unique goal $(g_x, g_y)$ which only it is responsible for reaching. Agents must coordinate implicitly by avoiding each other and the shared, dynamic dangers.

-----

## Setup and Running

### Prerequisites

  - Python 3.13.x "preferred"

  - Pygame

  - PyTorch

#### Easy setup

  - Make sure to have a Python environment running before installing the packages

<!-- end list -->

```bash
python -m venv env
```

```bash
.\env\Scripts\Activate
```

```bash
pip install pygame torch numpy
```

### Execution

1.  Run the main script:

    ```
    python main.py
    ```

2.  **Interaction:**

      - Use the **Menu** to configure the number of agents, obstacles, and the learning rate. (using keyboard arrows)

      - Select **START** to begin the MARL training loop. (using keyboard enter)

      - The simulation will display the agents' movement and the models' predicted paths (the white lines) in real-time.

      - Press **[S]** to stop the training and save the current model weights to `marl_save.pth`.

### Persistence

The model weights are automatically saved to `marl_save.pth` every 20 episodes and upon stopping the simulation, allowing training to resume on the learned policies.

