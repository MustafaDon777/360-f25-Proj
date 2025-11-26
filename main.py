import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import threading
import time
import os
import copy


# CONSTANTS
# ================
grid_size = 32
cell_size = 22 # Slightly larger for better visibility
screen_width = grid_size * cell_size + 320 
screen_height = grid_size * cell_size
fps = 60
SAVE_FILE = "marl_save.pth"

# Colors
BLACK = (12, 12, 15)
WHITE = (220, 220, 220)
GRAY = (50, 50, 60)
GREEN = (50, 255, 80)
RED = (255, 60, 60)
BLUE = (60, 120, 255)
CYAN = (60, 255, 255)
MAGENTA = (255, 60, 255)
ORANGE = (255, 165, 0)
TRANS_YELLOW = (255, 255, 0, 30)
PATH_COLOR = (255, 255, 255, 60) # Transparent white for path
UI_BG = (25, 25, 35)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # choice of processor

# ================
# 1. CONFIGURATION
# ================
class GameConfig:
    def __init__(self):
        self.num_agents = 2
        self.num_obstacles = 6
        self.map_shape = "Cross" 
        self.speed_mode = "Normal"
        self.learning_rate = 0.0005
        
        self.menu_idx = 0
        self.items = ["Agents", "Obstacles", "Shape", "Speed", "L. Rate", "START"]

    def get_sleep_time(self):
        if self.speed_mode == "Fast": return 0.0
        if self.speed_mode == "Normal": return 0.01
        return 0.08

# ================
# 2. AI MODEL
# ================
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # CNN: Input (3 channels, 5x5) -> Wall, Danger, Ally
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.flatten_dim = 32 * 5 * 5
        
        # Vector: x, y, gx, gy
        self.fc_vec = nn.Linear(4, 32)
        
        self.fc_common = nn.Linear(self.flatten_dim + 32, 256)
        self.actor = nn.Linear(256, 4) # Up, Down, Left, Right
        self.critic = nn.Linear(256, 1)

    def forward(self, grid, vec):
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        v = F.relu(self.fc_vec(vec))
        merged = torch.cat((x, v), dim=1)
        common = F.relu(self.fc_common(merged))
        return self.actor(common), self.critic(common)

# ================
# 3. ENVIRONMENT
# ================
class MovingObstacle:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.direction = 1 if random.random() > 0.5 else -1
        self.timer = 0
    
    def move(self, grid_map):
        self.timer += 1
        if self.timer > 6: # Slow movement
            nx = self.x + self.direction
            # Check bounds and walls (0 is empty, 1 is wall)
            if 0 <= nx < grid_size and grid_map[self.y][nx] == 0:
                self.x = nx
            else:
                self.direction *= -1
            self.timer = 0

class GridEnvironment:
    def __init__(self, config):
        self.cfg = config
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.agents = []
        self.obstacles = []
        self.goals = []
        self.step_count = 0
        self.max_steps = 300
        self.setup_map()

    def setup_map(self):
        self.grid.fill(1) # Walls
        mid = grid_size // 2
        w = 5
        
        if self.cfg.map_shape == "Cross":
            self.grid[:, mid-w:mid+w] = 0 
            self.grid[mid-w:mid+w, :] = 0 
        elif self.cfg.map_shape == "T-Shape":
            self.grid[2:2+w*2, :] = 0 
            self.grid[:, mid-w:mid+w] = 0 
            
        ## Random internal walls (commented for simplifying the environment)
        # for _ in range(5):
        #     rx, ry = random.randint(1, grid_size-2), random.randint(1, grid_size-2)
        #     if self.grid[ry][rx] == 0: self.grid[ry][rx] = 1

    def reset(self):
        self.step_count = 0
        self.agents = []
        self.goals = []
        self.obstacles = []
        
        # 1. Spawn Agents (at the Bottom)
        for i in range(self.cfg.num_agents):
            while True:
                ax = random.randint(grid_size//2 - 5, grid_size//2 + 5)
                ay = grid_size - 3
                if 0 <= ax < grid_size and self.grid[ay][ax] == 0:
                    overlap = any(a['x']==ax and a['y']==ay for a in self.agents)
                    if not overlap:
                        self.agents.append({'id': i, 'x': ax, 'y': ay, 'active': True, 'done': False})
                        break
        
        # 2. Spawn UNIQUE Goals (Agent destination)
        possible_goals = []
        # Find all valid spots at Top, Left, Right
        if self.cfg.map_shape == "Cross":
            # Top edge
            for x in range(grid_size//2 - 3, grid_size//2 + 3):
                if self.grid[2][x] == 0: possible_goals.append((x, 2))
            # Left edge
            for y in range(grid_size//2 - 3, grid_size//2 + 3):
                if self.grid[y][2] == 0: possible_goals.append((2, y))
            # Right edge
            for y in range(grid_size//2 - 3, grid_size//2 + 3):
                if self.grid[y][grid_size-3] == 0: possible_goals.append((grid_size-3, y))
        else:
            # T-Shape Top bar (space shape)
            for x in range(3, grid_size-3):
                if self.grid[3][x] == 0: possible_goals.append((x, 3))

        # Shuffle and pick unique
        random.shuffle(possible_goals)
        if len(possible_goals) < self.cfg.num_agents:
            # Fallback if map is too small (rare)
            possible_goals = possible_goals * 2 
            
        self.goals = possible_goals[:self.cfg.num_agents]

        # 3. Spawn Obstacles
        for _ in range(self.cfg.num_obstacles):
            ox = random.randint(5, grid_size-5)
            oy = random.randint(5, grid_size-5)
            if self.grid[oy][ox] == 0:
                self.obstacles.append(MovingObstacle(ox, oy))
                
        return self.get_obs()

    def get_obs(self):
        obs = []
        for i, ag in enumerate(self.agents):
            if not ag['active']:
                obs.append((torch.zeros(3, 5, 5), torch.zeros(4)))
                continue
            
            # 5x5 Grid Construction (Agent vision)
            local = np.zeros((3, 5, 5), dtype=np.float32)
            cx, cy = ag['x'], ag['y']
            
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    gx, gy = cx+dx, cy+dy
                    lx, ly = dx+2, dy+2
                    if 0<=gx<grid_size and 0<=gy<grid_size:
                        # Channel 0: Wall
                        if self.grid[gy][gx] == 1: local[0][ly][lx] = 1.0 
                        # Channel 1: Danger (Obstacles)
                        for o in self.obstacles:
                            if o.x == gx and o.y == gy: local[1][ly][lx] = 1.0
                        # Channel 2: Ally (Another agent)
                        for other in self.agents:
                            if other['id']!=i and other['active'] and other['x']==gx and other['y']==gy:
                                local[2][ly][lx] = 1.0
                    else:
                        local[0][ly][lx] = 1.0 # Out of bounds
            
            gx, gy = self.goals[i]
            vec = np.array([ag['x']/grid_size, ag['y']/grid_size, gx/grid_size, gy/grid_size], dtype=np.float32)
            obs.append((torch.tensor(local), torch.tensor(vec)))
        return obs

    def step(self, actions):
        self.step_count += 1
        rews, dones = [], []
        
        # Move obstacles
        for o in self.obstacles: o.move(self.grid)

        for i, ag in enumerate(self.agents):
            if not ag['active'] or ag['done']:
                rews.append(0); dones.append(True); continue
            
            act = actions[i]
            dx, dy = 0, 0
            match act:
                case 0:
                    dy=-1 # Up
                case 1:
                    dy=1 # Down
                case 2:
                    dx=-1 # Left
                case 3:
                    dx=1 # Right
            
            nx, ny = ag['x']+dx, ag['y']+dy
            gx, gy = self.goals[i]
            
            # --- REWARD CALCULATIONS ---
            r = -0.01 # Base penalty
            d = False
            
            # 1. Distance Logic (Penalty for moving away)
            prev_dist = abs(ag['x']-gx) + abs(ag['y']-gy)
            new_dist = abs(nx-gx) + abs(ny-gy)
            
            # Wall Collision
            if not (0<=nx<grid_size and 0<=ny<grid_size) or self.grid[ny][nx]==1:
                r = -0.5 
                # Stay in place (nx, ny not updated)
                nx, ny = ag['x'], ag['y'] 
            else:
                # 2. Moving Away Penalty (If not a wall hit)
                if new_dist > prev_dist:
                    r -= 0.1
                    
                # 3. Check Collisions
                hit_obs = any(o.x==nx and o.y==ny for o in self.obstacles)
                hit_ag = any(o['id']!=i and o['active'] and o['x']==nx and o['y']==ny for o in self.agents)
                
                if hit_obs:
                    r = -5.0; ag['active'] = False; d = True
                elif hit_ag:
                    r = -1.0 # Hit friend
                    # Stay in place
                    nx, ny = ag['x'], ag['y']
                else:
                    # Valid Move
                    ag['x'], ag['y'] = nx, ny
                    if nx==gx and ny==gy:
                        r = 50.0; ag['done'] = True; d = True
                    else:
                        # Shaping reward (Distance improvement)
                        r += -0.002 * new_dist

            # 4. Fear Penalty (Proximity to Red Circle)
            # Scan immediate area (Radius 2)
            near_danger = False
            for o in self.obstacles:
                if abs(o.x - ag['x']) <= 2 and abs(o.y - ag['y']) <= 2:
                    near_danger = True
                    break
            if near_danger:
                r -= 0.2

            rews.append(r)
            dones.append(d)
        
        if self.step_count >= self.max_steps: dones = [True]*len(dones)
        return self.get_obs(), rews, dones

# ================
# 4. TRAINER
# ================
class MARL_Trainer:
    def __init__(self, config):
        self.cfg = config
        self.env = GridEnvironment(config)
        self.models = [ActorCritic().to(DEVICE) for _ in range(config.num_agents)]
        self.opts = [optim.Adam(m.parameters(), lr=config.learning_rate) for m in self.models]
        self.running = False
        self.data = {}
        self.load_models()

    def save_models(self):
        state = {'models': [m.state_dict() for m in self.models]}
        torch.save(state, SAVE_FILE)

    def load_models(self):
        if os.path.exists(SAVE_FILE):
            try:
                ckpt = torch.load(SAVE_FILE, weights_only=False)
                saved = ckpt['models']
                for i in range(min(len(self.models), len(saved))):
                    self.models[i].load_state_dict(saved[i])
                print("Models loaded.")
            except: pass

    # --- PATH PREDICTION FUNCTION ---
    # Simulates the model forward 10 steps for visualization
    def predict_paths(self):
        paths = {}
        with torch.no_grad():
            for i, ag in enumerate(self.env.agents):
                if not ag['active'] or ag['done']: continue
                
                path = []
                # Clone simple state
                cx, cy = ag['x'], ag['y']
                gx, gy = self.env.goals[i]
                
                # Simulate 5 steps
                for _ in range(5):
                    # Construct pseudo-observation
                    # (Simplified: we use the actual current grid for static walls, 
                    # but we ignore moving obstacles in prediction to simulate 'intent')
                    local = np.zeros((3, 5, 5), dtype=np.float32)
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            tx, ty = cx+dx, cy+dy
                            lx, ly = dx+2, dy+2
                            if 0<=tx<grid_size and 0<=ty<grid_size:
                                if self.env.grid[ty][tx] == 1: local[0][ly][lx] = 1.0
                            else:
                                local[0][ly][lx] = 1.0 # Bounds
                    
                    vec = np.array([cx/grid_size, cy/grid_size, gx/grid_size, gy/grid_size], dtype=np.float32)
                    
                    gt = torch.tensor(local).unsqueeze(0).to(DEVICE)
                    vt = torch.tensor(vec).unsqueeze(0).to(DEVICE)
                    
                    logits, _ = self.models[i](gt, vt)
                    act = torch.argmax(logits).item() # Greedy prediction
                    
                    if act==0: cy-=1
                    elif act==1: cy+=1
                    elif act==2: cx-=1
                    elif act==3: cx+=1
                    
                    path.append((cx, cy))
                
                paths[i] = path
        return paths

    def train_loop(self):
        self.running = True
        ep = 0
        gamma = 0.99
        
        while self.running:
            obs = self.env.reset()
            ep_rw = [0] * self.cfg.num_agents
            rollouts = [{'log_probs':[], 'vals':[], 'rews':[], 'masks':[]} for _ in range(self.cfg.num_agents)]
            
            done = False
            while not done:
                acts = []
                for i in range(self.cfg.num_agents):
                    if self.env.agents[i]['active'] and not self.env.agents[i]['done']:
                        gt, vt = obs[i]
                        gt, vt = gt.unsqueeze(0).to(DEVICE), vt.unsqueeze(0).to(DEVICE)
                        logits, val = self.models[i](gt, vt)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        acts.append(action.item())
                        rollouts[i]['log_probs'].append(dist.log_prob(action))
                        rollouts[i]['vals'].append(val)
                    else:
                        acts.append(0)
                
                next_obs, rews, dones = self.env.step(acts)
                done = all(dones)
                
                for i in range(self.cfg.num_agents):
                    if len(rollouts[i]['vals']) > len(rollouts[i]['rews']):
                        ep_rw[i] += rews[i]
                        rollouts[i]['rews'].append(torch.tensor([rews[i]], dtype=torch.float32).to(DEVICE))
                        rollouts[i]['masks'].append(torch.tensor([1-float(dones[i])], dtype=torch.float32).to(DEVICE))
                
                obs = next_obs
                
                # Get paths every few frames to save CPU
                predicted_paths = {}
                if self.env.step_count % 5 == 0:
                    predicted_paths = self.predict_paths()

                with threading.Lock():
                    self.data = {
                        'agents': copy.deepcopy(self.env.agents),
                        'obstacles': copy.deepcopy(self.env.obstacles),
                        'goals': self.env.goals,
                        'episode': ep,
                        'rewards': ep_rw,
                        'paths': predicted_paths
                    }
                
                if not self.running: break
                time.sleep(self.cfg.get_sleep_time())

            if self.running:
                for i in range(self.cfg.num_agents):
                    if not rollouts[i]['rews']: continue
                    
                    returns = []
                    R = 0
                    for r, m in zip(reversed(rollouts[i]['rews']), reversed(rollouts[i]['masks'])):
                        R = r + gamma * R * m
                        returns.insert(0, R)
                    
                    returns = torch.cat(returns).detach()
                    log_probs = torch.cat(rollouts[i]['log_probs'])
                    values = torch.cat(rollouts[i]['vals']).squeeze()
                    
                    L = min(len(returns), len(values), len(log_probs))
                    returns, values, log_probs = returns[:L], values[:L], log_probs[:L]
                    
                    adv = returns - values
                    loss = -(log_probs * adv.detach()).mean() + 0.5 * adv.pow(2).mean()
                    
                    self.opts[i].zero_grad()
                    loss.backward()
                    self.opts[i].step()

            ep += 1
            if ep % 20 == 0: self.save_models()

    def stop(self):
        self.running = False
        self.save_models()

# ================
# 5. UI & APP
# ================
class MainApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("MARL: Path Prediction & Complex Rewards")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 18)
        self.big_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.config = GameConfig()
        self.mode = "MENU"
        self.trainer = None
        self.last_paths = {}

    def draw_menu(self):
        self.screen.fill(UI_BG)
        title = self.big_font.render("MARL SETUP", True, GREEN)
        self.screen.blit(title, (50, 40))
        
        for i, item in enumerate(self.config.items):
            color = CYAN if i == self.config.menu_idx else WHITE
            val = ""
            if item == "Agents": val = f"< {self.config.num_agents} >"
            elif item == "Obstacles": val = f"< {self.config.num_obstacles} >"
            elif item == "Shape": val = f"< {self.config.map_shape} >"
            elif item == "Speed": val = f"< {self.config.speed_mode} >"
            elif item == "L. Rate": val = f"< {self.config.learning_rate} >"
            elif item == "START": val = "[ ENTER ]"
            
            txt = self.font.render(f"{item:<12} {val}", True, color)
            self.screen.blit(txt, (50, 120 + i * 40))
            
        inst = self.font.render("Arrow Keys to Modify", True, GRAY)
        self.screen.blit(inst, (50, screen_height - 40))

    def handle_menu(self, key):
        if key == pygame.K_UP: self.config.menu_idx = (self.config.menu_idx - 1) % 6
        elif key == pygame.K_DOWN: self.config.menu_idx = (self.config.menu_idx + 1) % 6
        elif key in [pygame.K_LEFT, pygame.K_RIGHT]:
            idx = self.config.menu_idx
            d = 1 if key == pygame.K_RIGHT else -1
            if idx == 0: self.config.num_agents = max(1, min(4, self.config.num_agents + d))
            elif idx == 1: self.config.num_obstacles = max(0, min(15, self.config.num_obstacles + d))
            elif idx == 2: self.config.map_shape = "T-Shape" if self.config.map_shape == "Cross" else "Cross"
            elif idx == 3: 
                m = ["Slow", "Normal", "Fast"]
                self.config.speed_mode = m[(m.index(self.config.speed_mode)+d)%3]
            elif idx == 4:
                lrs = [0.001, 0.0005, 0.0001]
                self.config.learning_rate = lrs[(lrs.index(self.config.learning_rate)+d)%3]
        elif key == pygame.K_RETURN and self.config.menu_idx == 5:
            self.mode = "TRAIN"
            self.trainer = MARL_Trainer(self.config)
            t = threading.Thread(target=self.trainer.train_loop)
            t.daemon = True
            t.start()

    def draw_train(self):
        self.screen.fill(BLACK)
        if not self.trainer or not self.trainer.data: return
        
        d = self.trainer.data
        if d.get('paths'): self.last_paths = d['paths']

        # Draw Walls (Static)
        for y in range(grid_size):
            for x in range(grid_size):
                if self.trainer.env.grid[y][x] == 1:
                    pygame.draw.rect(self.screen, GRAY, (x*cell_size, y*cell_size, cell_size, cell_size))
                else:
                    pygame.draw.rect(self.screen, (20,20,25), (x*cell_size, y*cell_size, cell_size, cell_size), 1)

        # Draw Path Predictions
        for aid, path in self.last_paths.items():
            if len(path) > 1:
                # Convert grid coords to pixel coords
                px_points = [(p[0]*cell_size+cell_size//2, p[1]*cell_size+cell_size//2) for p in path]
                pygame.draw.lines(self.screen, WHITE, False, px_points, 2)

        # Draw Goals
        for i, (gx, gy) in enumerate(d.get('goals', [])):
            col = [BLUE, CYAN, MAGENTA, ORANGE][i % 4]
            pygame.draw.circle(self.screen, col, (gx*cell_size+cell_size//2, gy*cell_size+cell_size//2), 6)
            pygame.draw.circle(self.screen, col, (gx*cell_size+cell_size//2, gy*cell_size+cell_size//2), 10, 1)

        # Draw Obstacles
        for o in d.get('obstacles', []):
            pygame.draw.circle(self.screen, RED, (o.x*cell_size+cell_size//2, o.y*cell_size+cell_size//2), 8)

        # Draw Agents
        for ag in d.get('agents', []):
            if ag['active']:
                cx, cy = ag['x']*cell_size+cell_size//2, ag['y']*cell_size+cell_size//2
                col = [BLUE, CYAN, MAGENTA, ORANGE][ag['id'] % 4]
                
                # Visibility box
                s = pygame.Surface((cell_size*5, cell_size*5), pygame.SRCALPHA)
                s.fill(TRANS_YELLOW)
                self.screen.blit(s, (cx-cell_size*2.5, cy-cell_size*2.5))
                
                pygame.draw.circle(self.screen, col, (cx, cy), 9)
                
                # Line to goal
                gx, gy = d['goals'][ag['id']]
                pygame.draw.line(self.screen, col, (cx, cy), (gx*cell_size+cell_size//2, gy*cell_size+cell_size//2), 1)

        # UI Panel
        ui_x = grid_size * cell_size
        pygame.draw.rect(self.screen, UI_BG, (ui_x, 0, 320, screen_height))
        pygame.draw.line(self.screen, WHITE, (ui_x, 0), (ui_x, screen_height))
        
        info = [
            f"Episode: {d.get('episode',0)}",
            f"Shape: {self.config.map_shape}",
            f"Obs: {self.config.num_obstacles}",
            "--- Rewards ---"
        ]
        
        y = 20
        for line in info:
            self.screen.blit(self.font.render(line, True, WHITE), (ui_x+20, y))
            y += 25
            
        for i, r in enumerate(d.get('rewards', [])):
            col = [BLUE, CYAN, MAGENTA, ORANGE][i % 4]
            self.screen.blit(self.font.render(f"Agent {i+1}: {r:.1f}", True, col), (ui_x+20, y))
            y += 25
            
        self.screen.blit(self.font.render("[S] Stop & Save", True, RED), (ui_x+20, screen_height-40))

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.trainer: self.trainer.stop()
                    pygame.quit(); return
                if event.type == pygame.KEYDOWN:
                    if self.mode == "MENU": self.handle_menu(event.key)
                    elif self.mode == "TRAIN" and event.key == pygame.K_s:
                        self.trainer.stop(); self.mode = "MENU"
            
            if self.mode == "MENU": self.draw_menu()
            else: self.draw_train()
            
            pygame.display.flip()
            self.clock.tick(fps)

# ================
# READ readme.md TO RUN SIMULATION
# ================
if __name__ == "__main__":
    MainApp().run()
