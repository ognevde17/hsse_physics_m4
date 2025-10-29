
import numpy as np
from typing import List, Dict, Callable
from scipy.integrate import odeint

try:
    from .ball_physics import Ball, Surface, BallDynamics
except ImportError:
    from ball_physics import Ball, Surface, BallDynamics


class Simulation:
    
    def __init__(self, ball: Ball, surface: Surface, dt: float = 0.01, 
                 total_time: float = 10.0, g: float = 9.81):
        self.ball = ball
        self.surface = surface
        self.dt = dt
        self.total_time = total_time
        self.g = g
        
        self.dynamics = BallDynamics(ball, surface, g)
        
        self.time_points: List[float] = []
        self.positions: List[np.ndarray] = []
        self.velocities: List[np.ndarray] = []
        self.angular_velocities: List[np.ndarray] = []
        self.energies: List[float] = []
        self.angular_momenta: List[np.ndarray] = []
        self.is_slipping_history: List[bool] = []
    
    def get_state_vector(self) -> np.ndarray:
        return np.concatenate([
            self.ball.position,
            self.ball.velocity,
            self.ball.angular_velocity
        ])
    
    def set_state_from_vector(self, state: np.ndarray):
        self.ball.position = state[0:2].copy()
        self.ball.velocity = state[2:4].copy()
        self.ball.angular_velocity = state[4:7].copy()
    
    def run(self, walls: List[Dict] = None, restitution: float = 1.0):
        current_time = 0.0
        
        while current_time < self.total_time:
            self.time_points.append(current_time)
            self.positions.append(self.ball.position.copy())
            self.velocities.append(self.ball.velocity.copy())
            self.angular_velocities.append(self.ball.angular_velocity.copy())
            self.energies.append(self.ball.kinetic_energy())
            self.angular_momenta.append(self.ball.angular_momentum().copy())
            self.is_slipping_history.append(self.dynamics.is_slipping)
            
            state = self.get_state_vector()
            t_span = [current_time, current_time + self.dt]
            
            solution = odeint(
                self.dynamics.equations_of_motion,
                state,
                t_span
            )
            
            self.set_state_from_vector(solution[-1])
            
            if walls is not None:
                for wall in walls:
                    self.dynamics.handle_wall_collision(
                        wall['position'], 
                        wall.get('axis', 0),
                        restitution
                    )
            
            speed = np.linalg.norm(self.ball.velocity)
            angular_speed = np.linalg.norm(self.ball.angular_velocity)
            
            if speed < 1e-6 and angular_speed < 1e-6:
                self.ball.velocity = np.zeros(2)
                self.ball.angular_velocity = np.zeros(3)
                
                if abs(self.surface.angle) < 1e-6:
                    break
            
            current_time += self.dt
        
        if len(self.time_points) == 0 or self.time_points[-1] < current_time:
            self.time_points.append(current_time)
            self.positions.append(self.ball.position.copy())
            self.velocities.append(self.ball.velocity.copy())
            self.angular_velocities.append(self.ball.angular_velocity.copy())
            self.energies.append(self.ball.kinetic_energy())
            self.angular_momenta.append(self.ball.angular_momentum().copy())
            self.is_slipping_history.append(self.dynamics.is_slipping)
    
    def get_results(self) -> Dict:
        return {
            'time': np.array(self.time_points),
            'position': np.array(self.positions),
            'velocity': np.array(self.velocities),
            'angular_velocity': np.array(self.angular_velocities),
            'energy': np.array(self.energies),
            'angular_momentum': np.array(self.angular_momenta),
            'is_slipping': np.array(self.is_slipping_history)
        }
    
    def check_energy_conservation(self, tolerance: float = 0.05) -> bool:
        if len(self.energies) < 2:
            return True
        
        initial_energy = self.energies[0]
        
        for i, pos in enumerate(self.positions):
            height_change = -pos[0] * np.sin(self.surface.angle)
            potential_energy = self.ball.mass * self.g * height_change
            total_energy = self.energies[i] + potential_energy
            
            if initial_energy > 0:
                deviation = abs(total_energy - initial_energy) / initial_energy
                if deviation > tolerance and not self.dynamics.is_slipping:
                    return False
        
        return True
    
    def check_angular_momentum_conservation(self, tolerance: float = 0.05) -> bool:
        if len(self.angular_momenta) < 2:
            return True
        
        if abs(self.surface.angle) > 1e-6:
            return True
        
        initial_L = np.linalg.norm(self.angular_momenta[0])
        
        for L in self.angular_momenta:
            L_magnitude = np.linalg.norm(L)
            if initial_L > 0:
                deviation = abs(L_magnitude - initial_L) / initial_L
                if deviation > tolerance:
                    return False
        
        return True


class MultiballSimulation:
    
    def __init__(self, balls: List[Ball], surface: Surface, 
                 dt: float = 0.01, total_time: float = 10.0, g: float = 9.81):
        self.balls = balls
        self.surface = surface
        self.dt = dt
        self.total_time = total_time
        self.g = g
        
        self.dynamics_list = [BallDynamics(ball, surface, g) for ball in balls]
        
        self.time_points: List[float] = []
        self.ball_positions: List[List[np.ndarray]] = [[] for _ in balls]
        self.ball_velocities: List[List[np.ndarray]] = [[] for _ in balls]
    
    def run(self, walls: List[Dict] = None, restitution: float = 1.0):
        current_time = 0.0
        
        while current_time < self.total_time:
            self.time_points.append(current_time)
            for i, ball in enumerate(self.balls):
                self.ball_positions[i].append(ball.position.copy())
                self.ball_velocities[i].append(ball.velocity.copy())
            
            for i, (ball, dynamics) in enumerate(zip(self.balls, self.dynamics_list)):
                state = np.concatenate([
                    ball.position,
                    ball.velocity,
                    ball.angular_velocity
                ])
                
                t_span = [current_time, current_time + self.dt]
                solution = odeint(dynamics.equations_of_motion, state, t_span)
                
                ball.position = solution[-1, 0:2].copy()
                ball.velocity = solution[-1, 2:4].copy()
                ball.angular_velocity = solution[-1, 4:7].copy()
                
                if walls is not None:
                    for wall in walls:
                        dynamics.handle_wall_collision(
                            wall['position'],
                            wall.get('axis', 0),
                            restitution
                        )
            
            for i in range(len(self.balls)):
                for j in range(i + 1, len(self.balls)):
                    self.dynamics_list[i].handle_ball_collision(
                        self.balls[j], 
                        restitution
                    )
            
            current_time += self.dt
        
        self.time_points.append(current_time)
        for i, ball in enumerate(self.balls):
            self.ball_positions[i].append(ball.position.copy())
            self.ball_velocities[i].append(ball.velocity.copy())
    
    def get_results(self) -> Dict:
        return {
            'time': np.array(self.time_points),
            'positions': [np.array(pos) for pos in self.ball_positions],
            'velocities': [np.array(vel) for vel in self.ball_velocities]
        }

