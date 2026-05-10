import os
import traci
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# --- 1. RESEARCH LOGGER CALLBACK ---
class ResearchLoggerCallback(BaseCallback):
    def __init__(self, filename="proposed_stress_log.csv", verbose=0):
        super(ResearchLoggerCallback, self).__init__(verbose)
        self.filename = filename
        self.episode_count = 0
        self.header = ["Episode", "Total_Reward", "Steps", "Value_Loss", "Policy_Loss"]
        
        with open(self.filename, "w", newline="") as f:
            csv.writer(f).writerow(self.header)

    def _on_step(self) -> bool:
        if "episode" in self.locals["infos"][0]:
            self.episode_count += 1
            ep_info = self.locals["infos"][0]["episode"]
            v_loss = self.model.logger.name_to_value.get("train/value_loss", 0)
            p_loss = self.model.logger.name_to_value.get("train/policy_gradient_loss", 0)
            
            with open(self.filename, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.episode_count,
                    round(ep_info["r"], 2),
                    ep_info["l"],
                    round(v_loss, 5),
                    round(p_loss, 5)
                ])
        return True

# --- 2. PROPOSED ENVIRONMENT (INFORMED & COOPERATIVE) ---
class InformedStressEnv(gym.Env):
    def __init__(self, use_gui=False, alpha=0.5):
        super().__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(current_dir, "sumo_traffic_project", "config.sumocfg")
        self.use_gui = use_gui
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.alpha = alpha # Cooperation factor
        
        # Proposed State: [Local_Q1, Local_Q2, Local_Phase, Neighbor_Q, Neighbor_Phase]
        self.observation_space = spaces.Box(low=0, high=100, shape=(2, 5), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2, 2])
        self.prev_actions = np.zeros(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if traci.isLoaded(): traci.close()
        traci.start([self.sumo_binary, "-c", self.config_file, "--start", "--quit-on-end", "--no-warnings"])
        self.prev_actions = np.zeros(2)
        return self._get_obs(), {}

    def _get_obs(self):
        try:
            # J1 Local: E0(West), E1(South)
            q0, q1, p1 = traci.edge.getLastStepHaltingNumber("E0"), traci.edge.getLastStepHaltingNumber("E1"), traci.trafficlight.getPhase("J1")
            # J2 Local: E2(Connector), E3(North)
            q2, q3, p2 = traci.edge.getLastStepHaltingNumber("E2"), traci.edge.getLastStepHaltingNumber("E3"), traci.trafficlight.getPhase("J2")

            # INFORMED STATE: Agents see neighbor's inputs
            obs1 = [q0, q1, p1, q3, p2] # J1 sees J2's heavy North input
            obs2 = [q2, q3, p2, q0, p1] # J2 sees J1's heavy West input
            return np.array([obs1, obs2], dtype=np.float32)
        except: return np.zeros((2, 5), dtype=np.float32)

    def step(self, actions):
        # B. SWITCHING PENALTY
        switch_penalty = 0
        if actions[0] != self.prev_actions[0]: switch_penalty -= 5.0
        if actions[1] != self.prev_actions[1]: switch_penalty -= 5.0
        self.prev_actions = actions.copy()

        traci.trafficlight.setPhase("J1", int(actions[0] * 2))
        traci.trafficlight.setPhase("J2", int(actions[1] * 2))
        
        for _ in range(10): traci.simulationStep()
            
        obs = self._get_obs()
        
        # C. COOPERATIVE REWARD CALCULATION
        # Local "Pain"
        r_j1_local = -(obs[0][0] + obs[0][1])
        r_j2_local = -(obs[1][0] + obs[1][1])
        
        # Mixing Local and Neighbor reward
        reward_j1 = (1 - self.alpha) * r_j1_local + (self.alpha) * r_j2_local
        reward_j2 = (1 - self.alpha) * r_j2_local + (self.alpha) * r_j1_local
        
        total_reward = (reward_j1 + reward_j2) + switch_penalty
        
        # D. INCREASED THRESHOLD (Match Baseline)
        total_halting = np.sum(obs[:, :2])
        terminated = bool(total_halting > 300)
        truncated = bool(traci.simulation.getMinExpectedNumber() <= 0)

        return obs, total_reward, terminated, truncated, {}

# --- 3. TRAINING EXECUTION ---
if __name__ == "__main__":
    env = InformedStressEnv(use_gui=False, alpha=0.5)
    callback = ResearchLoggerCallback(filename="proposed_stress_log.csv")
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        n_steps=1024, # Consistent with baseline stress test
        tensorboard_log="./tensorboard_logs/proposed_stress/"
    )
    
    print("Starting PROPOSED METHOD Stress Test...")
    try:
        model.learn(total_timesteps=60000, callback=callback)
        model.save("ippo_stress_proposed")
    finally:
        env.close()