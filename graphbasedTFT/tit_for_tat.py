import numpy as np


def tft_improved(alpha, r, beta=0):  # improved Tit-for-Tat with parameters alpha, r and beta
    def function(old_coop_degrees, detected_coop_degrees, r):
        old_coop_degrees = np.array(old_coop_degrees)
        detected_coop_degrees = np.array(detected_coop_degrees)
        delta = detected_coop_degrees - old_coop_degrees
        r = np.clip(r + beta * delta, 0.0, 1.0)
        output = alpha * old_coop_degrees + (1 - alpha) * (r + (1 - r) * detected_coop_degrees)
        return np.clip(output,0.0,1.0), r

    return function, r


def tft_inertia(alpha, r):  # improved Tit-for-Tat with parameters alpha, r
    def function(old_coop_degrees, detected_coop_degrees):
        old_coop_degrees = np.array(old_coop_degrees)
        detected_coop_degrees = np.array(detected_coop_degrees)
        output = alpha * old_coop_degrees + (1 - alpha) * (r + (1 - r) * detected_coop_degrees)
        return np.clip(output,0.0,1.0)

    return function



class TFT_improved:
    def __init__(self, alpha, r, beta, n_agents):
        self.alpha = alpha
        self.r_init = r
        self.r = r
        self.beta = beta
        self.n_agents = n_agents
        self.old_coop_degrees = np.zeros(self.n_agents)
        self.algo, _ = tft_improved(alpha, r, beta)

    def reset(self):
        self.old_coop_degrees = np.zeros(self.n_agents)
        self.r = self.r_init

    def act(self, detected_degrees):
        output, r_new = self.algo(self.old_coop_degrees, detected_degrees, self.r)
        self.r = r_new
        self.old_coop_degrees = output
        return np.clip(output,0.0,1.0)


class TFT_inertia:
    def __init__(self, alpha, r, n_agents):
        self.alpha = alpha
        self.r = [r]
        self.n_agents = n_agents
        self.old_coop_degrees = np.zeros(self.n_agents)
        self.algo = tft_inertia(alpha, r)

    def reset(self):
        self.old_coop_degrees = np.zeros(self.n_agents)

    def act(self, detected_degrees):
        output = self.algo(self.old_coop_degrees, detected_degrees)
        self.old_coop_degrees = output
        return np.clip(output,0.0,1.0)

class TFT_improved_beta:
    def __init__(self, alpha, r, beta, gamma, n_agents):
        self.alpha = alpha
        self.r_init = r
        self.r = r
        self.beta = beta
        self.gamma = gamma
        self.n_agents = n_agents
        self.old_coop_degrees = np.zeros(self.n_agents)
        self.algo, _ = tft_improved(alpha, r, beta)

    def reset(self):
        self.old_coop_degrees = np.zeros(self.n_agents)
        self.r = self.r_init

    def act(self, detected_degrees):
        output, r_new = self.algo(self.old_coop_degrees, detected_degrees, self.r)
        self.r = np.minimum(r_new + self.r_init * (np.random.rand() < self.gamma), 1.0)
        self.old_coop_degrees = output
        return np.clip(output,0.0,1.0)

class Nice_algo:
    def __init__(self):
        pass

    def reset(self):
        pass

    def act(self, detected_degrees):
        n_a = len(detected_degrees)
        output = np.ones(n_a)
        return output


class Egoist_algo:
    def __init__(self):
        pass

    def reset(self):
        pass

    def act(self, detected_degrees):
        n_a = len(detected_degrees)
        output = np.zeros(n_a)
        return output


class Traitor_algo:
    def __init__(self, t_traitor):
        self.step_t = 0
        self.t_traitor = t_traitor

    def reset(self):
        self.step_t = 0

    def act(self, detected_degrees):
        n_a = len(detected_degrees)
        if self.step_t <= self.t_traitor:
            output = np.ones(n_a)
        else:
            output = np.zeros(n_a)
        self.step_t += 1
        return output


class LateNice_algo:
    def __init__(self, t_coop):
        self.step_t = 0
        self.t_coop = t_coop

    def reset(self):
        self.step_t = 0

    def act(self, detected_degrees):
        n_a = len(detected_degrees)
        if self.step_t <= self.t_coop:
            output = np.zeros(n_a)
        else:
            output = np.ones(n_a)
        self.step_t += 1
        return output