
import numpy as np
from typing import Tuple, Optional


class Ball:
    
    def __init__(self, mass: float, radius: float, position: np.ndarray, 
                 velocity: np.ndarray, angular_velocity: np.ndarray):
        self.mass = mass
        self.radius = radius
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.angular_velocity = np.array(angular_velocity, dtype=float)
        
        self.moment_of_inertia = 0.4 * mass * radius**2
    
    def kinetic_energy(self) -> float:
        translational = 0.5 * self.mass * np.sum(self.velocity**2)
        rotational = 0.5 * self.moment_of_inertia * np.sum(self.angular_velocity**2)
        return translational + rotational
    
    def angular_momentum(self) -> np.ndarray:
        return self.moment_of_inertia * self.angular_velocity
    
    def linear_momentum(self) -> np.ndarray:
        return self.mass * self.velocity


class Surface:
    
    def __init__(self, friction_coeff: float, angle: float = 0.0, 
                 bounds: Optional[Tuple[float, float, float, float]] = None):
        self.friction_coeff = friction_coeff
        self.angle = np.radians(angle)
        self.bounds = bounds
    
    def is_within_bounds(self, position: np.ndarray) -> bool:
        if self.bounds is None:
            return True
        x_min, x_max, y_min, y_max = self.bounds
        x, y = position
        return x_min <= x <= x_max and y_min <= y <= y_max


class BallDynamics:
    
    def __init__(self, ball: Ball, surface: Surface, g: float = 9.81):
        self.ball = ball
        self.surface = surface
        self.g = g
        self.is_slipping = False
    
    def normal_force(self) -> float:
        return self.ball.mass * self.g * np.cos(self.surface.angle)
    
    def check_slipping_condition(self) -> bool:
        v_contact = self.ball.velocity - np.cross(
            self.ball.angular_velocity, 
            np.array([0, 0, -self.ball.radius])
        )[:2]
        
        return np.linalg.norm(v_contact) > 1e-6
    
    def friction_force(self) -> np.ndarray:
        N = self.normal_force()
        
        if self.is_slipping:
            if np.linalg.norm(self.ball.velocity) > 1e-10:
                direction = -self.ball.velocity / np.linalg.norm(self.ball.velocity)
                return self.surface.friction_coeff * N * direction
            return np.zeros(2)
        else:
            return np.zeros(2)
    
    def equations_of_motion(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, vx, vy, wx, wy, wz = state
        
        self.ball.position = np.array([x, y])
        self.ball.velocity = np.array([vx, vy])
        self.ball.angular_velocity = np.array([wx, wy, wz])
        
        N = self.normal_force()
        f_max = self.surface.friction_coeff * N
        
        if abs(self.surface.angle) > 1e-6:
            cos_theta = np.cos(self.surface.angle)
            sin_theta = np.sin(self.surface.angle)
            
            incline_direction = np.array([cos_theta, -sin_theta])
            
            f_required = (2.0/7.0) * self.ball.mass * self.g * sin_theta
            
            if f_required > f_max:
                self.is_slipping = True
                
                v_magnitude = np.sqrt(vx**2 + vy**2)
                
                if v_magnitude > 1e-10:
                    friction_direction = -np.array([vx, vy]) / v_magnitude
                    F_friction = self.surface.friction_coeff * N * friction_direction
                else:
                    F_friction = -self.surface.friction_coeff * N * incline_direction
                
                F_gravity = np.array([0.0, -self.ball.mass * self.g])
                F_normal = N * np.array([-sin_theta, cos_theta])
                
                acceleration = (F_gravity + F_normal + F_friction) / self.ball.mass
                
                r_contact = np.array([0, 0, -self.ball.radius])
                F_friction_3d = np.array([F_friction[0], F_friction[1], 0])
                torque = np.cross(r_contact, F_friction_3d)
                
                angular_acceleration = torque / self.ball.moment_of_inertia
            else:
                self.is_slipping = False
                
                a_magnitude = (5.0/7.0) * self.g * sin_theta
                acceleration = a_magnitude * incline_direction
                
                if self.ball.radius > 1e-10:
                    angular_acceleration = np.array([
                        acceleration[1] / self.ball.radius,
                        -acceleration[0] / self.ball.radius,
                        0.0
                    ])
                else:
                    angular_acceleration = np.zeros(3)
        else:
            v_magnitude = np.sqrt(vx**2 + vy**2)
            
            if v_magnitude > 1e-8:
                self.is_slipping = False
                
                v_direction = np.array([vx, vy]) / v_magnitude
                
                a_max = (2.0/7.0) * self.surface.friction_coeff * self.g
                ax = -a_max * v_direction[0]
                ay = -a_max * v_direction[1]
                
                acceleration = np.array([ax, ay])
                
                if self.ball.radius > 1e-10:
                    angular_acceleration = np.array([
                        ay / self.ball.radius,
                        -ax / self.ball.radius,
                        0.0
                    ])
                else:
                    angular_acceleration = np.zeros(3)
            else:
                self.is_slipping = False
                acceleration = np.zeros(2)
                angular_acceleration = np.zeros(3)
        
        return np.array([vx, vy, acceleration[0], acceleration[1], 
                        angular_acceleration[0], angular_acceleration[1], 
                        angular_acceleration[2]])
    
    def handle_wall_collision(self, wall_position: float, axis: int = 0, 
                            restitution: float = 1.0) -> bool:
        distance = self.ball.position[axis] - wall_position
        
        if axis == 0:
            if abs(distance) <= self.ball.radius:
                self.ball.velocity[axis] = -restitution * self.ball.velocity[axis]
                
                if distance > 0:
                    self.ball.position[axis] = wall_position + self.ball.radius
                else:
                    self.ball.position[axis] = wall_position - self.ball.radius
                
                return True
        elif axis == 1:
            if abs(distance) <= self.ball.radius:
                self.ball.velocity[axis] = -restitution * self.ball.velocity[axis]
                
                if distance > 0:
                    self.ball.position[axis] = wall_position + self.ball.radius
                else:
                    self.ball.position[axis] = wall_position - self.ball.radius
                
                return True
        
        return False
    
    def handle_ball_collision(self, other_ball: Ball, restitution: float = 1.0) -> bool:
        delta_pos = self.ball.position - other_ball.position
        distance = np.linalg.norm(delta_pos)
        
        if distance <= self.ball.radius + other_ball.radius:
            normal = delta_pos / distance
            
            relative_velocity = self.ball.velocity - other_ball.velocity
            
            v_normal = np.dot(relative_velocity, normal)
            
            if v_normal < 0:
                impulse = -(1 + restitution) * v_normal * normal
                impulse *= (self.ball.mass * other_ball.mass) / (self.ball.mass + other_ball.mass)
                
                self.ball.velocity += impulse / self.ball.mass
                other_ball.velocity -= impulse / other_ball.mass
                
                overlap = self.ball.radius + other_ball.radius - distance
                correction = overlap * normal / 2
                self.ball.position += correction
                other_ball.position -= correction
                
                return True
        
        return False

