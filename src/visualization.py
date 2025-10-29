
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from typing import Dict, List, Optional
import matplotlib.patches as mpatches


def plot_trajectory(results: Dict, surface_angle: float = 0.0, 
                    save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = results['position']
    x = positions[:, 0]
    y = positions[:, 1]
    
    ax.plot(x, y, 'b-', linewidth=2, label='Траектория')
    ax.plot(x[0], y[0], 'go', markersize=10, label='Начало')
    ax.plot(x[-1], y[-1], 'ro', markersize=10, label='Конец')
    
    ax.set_xlabel('X (м)', fontsize=12)
    ax.set_ylabel('Y (м)', fontsize=12)
    ax.set_title(f'Траектория движения шара (угол наклона: {surface_angle}°)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_energy(results: Dict, ball_mass: float, g: float = 9.81, 
                surface_angle: float = 0.0, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time = results['time']
    kinetic_energy = results['energy']
    positions = results['position']
    
    angle_rad = np.radians(surface_angle)
    height_change = -positions[:, 0] * np.sin(angle_rad)
    potential_energy = ball_mass * g * height_change
    
    total_energy = kinetic_energy + potential_energy
    
    ax.plot(time, kinetic_energy, 'b-', linewidth=2, label='Кинетическая энергия')
    ax.plot(time, potential_energy, 'r-', linewidth=2, label='Потенциальная энергия')
    ax.plot(time, total_energy, 'g--', linewidth=2, label='Полная энергия')
    
    ax.set_xlabel('Время (с)', fontsize=12)
    ax.set_ylabel('Энергия (Дж)', fontsize=12)
    ax.set_title('Сохранение энергии', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if len(total_energy) > 0 and abs(total_energy[0]) > 1e-10:
        energy_deviation = (np.max(total_energy) - np.min(total_energy)) / abs(total_energy[0]) * 100
        ax.text(0.02, 0.98, f'Отклонение энергии: {energy_deviation:.2f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_velocity(results: Dict, save_path: Optional[str] = None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    time = results['time']
    velocity = results['velocity']
    vx = velocity[:, 0]
    vy = velocity[:, 1]
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    ax1.plot(time, vx, 'b-', linewidth=2, label='vx')
    ax1.plot(time, vy, 'r-', linewidth=2, label='vy')
    ax1.set_xlabel('Время (с)', fontsize=12)
    ax1.set_ylabel('Скорость (м/с)', fontsize=12)
    ax1.set_title('Компоненты скорости', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(time, v_magnitude, 'g-', linewidth=2)
    ax2.set_xlabel('Время (с)', fontsize=12)
    ax2.set_ylabel('|v| (м/с)', fontsize=12)
    ax2.set_title('Модуль скорости', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_angular_velocity(results: Dict, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time = results['time']
    angular_velocity = results['angular_velocity']
    wx = angular_velocity[:, 0]
    wy = angular_velocity[:, 1]
    wz = angular_velocity[:, 2]
    
    ax.plot(time, wx, 'b-', linewidth=2, label='ωx')
    ax.plot(time, wy, 'r-', linewidth=2, label='ωy')
    ax.plot(time, wz, 'g-', linewidth=2, label='ωz')
    
    ax.set_xlabel('Время (с)', fontsize=12)
    ax.set_ylabel('Угловая скорость (рад/с)', fontsize=12)
    ax.set_title('Изменение угловой скорости', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_slipping_regions(results: Dict, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time = results['time']
    is_slipping = results.get('is_slipping', np.zeros_like(time))
    
    colors = ['green' if not slip else 'red' for slip in is_slipping]
    ax.scatter(time, np.zeros_like(time), c=colors, s=50, alpha=0.5)
    
    ax.set_xlabel('Время (с)', fontsize=12)
    ax.set_title('Режимы движения', fontsize=14)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    rolling_patch = mpatches.Patch(color='green', label='Качение без проскальзывания')
    slipping_patch = mpatches.Patch(color='red', label='Проскальзывание')
    ax.legend(handles=[rolling_patch, slipping_patch])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def create_animation(results: Dict, ball_radius: float, 
                     surface_angle: float = 0.0,
                     walls: Optional[List[Dict]] = None,
                     save_path: Optional[str] = None,
                     fps: int = 30):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    positions = results['position']
    time = results['time']
    is_slipping = results.get('is_slipping', np.zeros(len(time)))
    
    x_min, x_max = positions[:, 0].min() - 2*ball_radius, positions[:, 0].max() + 2*ball_radius
    y_min, y_max = positions[:, 1].min() - 2*ball_radius, positions[:, 1].max() + 2*ball_radius
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_aspect('equal')
    
    if abs(surface_angle) > 0.1:
        surface_y = y_min - ball_radius
        surface = Rectangle((x_min, surface_y - 0.5), x_max - x_min, 0.5, 
                           color='brown', alpha=0.5)
        ax.add_patch(surface)
    else:
        ax.axhline(y=0, color='brown', linewidth=2, alpha=0.5)
    
    if walls:
        for wall in walls:
            if wall.get('axis', 0) == 0:
                ax.axvline(x=wall['position'], color='gray', linewidth=3)
            else:
                ax.axhline(y=wall['position'], color='gray', linewidth=3)
    
    ball = Circle((positions[0, 0], positions[0, 1]), ball_radius, 
                  color='blue', alpha=0.7)
    ax.add_patch(ball)
    
    trajectory_line, = ax.plot([], [], 'b--', alpha=0.3, linewidth=1)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    mode_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12)
    
    ax.set_xlabel('X (м)', fontsize=12)
    ax.set_ylabel('Y (м)', fontsize=12)
    ax.set_title(f'Движение шара (угол наклона: {surface_angle}°)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    trajectory_x = []
    trajectory_y = []
    
    def init():
        ball.center = (positions[0, 0], positions[0, 1])
        trajectory_line.set_data([], [])
        time_text.set_text('')
        mode_text.set_text('')
        return ball, trajectory_line, time_text, mode_text
    
    def animate(frame):
        ball.center = (positions[frame, 0], positions[frame, 1])
        
        if is_slipping[frame]:
            ball.set_color('red')
        else:
            ball.set_color('blue')
        
        trajectory_x.append(positions[frame, 0])
        trajectory_y.append(positions[frame, 1])
        trajectory_line.set_data(trajectory_x, trajectory_y)
        
        time_text.set_text(f'Время: {time[frame]:.2f} с')
        mode = 'Проскальзывание' if is_slipping[frame] else 'Качение'
        mode_text.set_text(f'Режим: {mode}')
        
        return ball, trajectory_line, time_text, mode_text
    
    skip_frames = max(1, len(positions) // (fps * 10))
    frames = range(0, len(positions), skip_frames)
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames,
                        interval=1000/fps, blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
    
    return anim


def create_multiball_animation(results: Dict, ball_radii: List[float],
                               walls: Optional[List[Dict]] = None,
                               save_path: Optional[str] = None,
                               fps: int = 30):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    positions_list = results['positions']
    time = results['time']
    n_balls = len(positions_list)
    
    all_positions = np.vstack(positions_list)
    max_radius = max(ball_radii)
    x_min, x_max = all_positions[:, 0].min() - 2*max_radius, all_positions[:, 0].max() + 2*max_radius
    y_min, y_max = all_positions[:, 1].min() - 2*max_radius, all_positions[:, 1].max() + 2*max_radius
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    if walls:
        for wall in walls:
            if wall.get('axis', 0) == 0:
                ax.axvline(x=wall['position'], color='gray', linewidth=3)
            else:
                ax.axhline(y=wall['position'], color='gray', linewidth=3)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_balls))
    balls = []
    for i in range(n_balls):
        ball = Circle((positions_list[i][0, 0], positions_list[i][0, 1]), 
                     ball_radii[i], color=colors[i], alpha=0.7)
        ax.add_patch(ball)
        balls.append(ball)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    ax.set_xlabel('X (м)', fontsize=12)
    ax.set_ylabel('Y (м)', fontsize=12)
    ax.set_title('Движение нескольких шаров', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    def init():
        for i, ball in enumerate(balls):
            ball.center = (positions_list[i][0, 0], positions_list[i][0, 1])
        time_text.set_text('')
        return balls + [time_text]
    
    def animate(frame):
        for i, ball in enumerate(balls):
            ball.center = (positions_list[i][frame, 0], positions_list[i][frame, 1])
        time_text.set_text(f'Время: {time[frame]:.2f} с')
        return balls + [time_text]
    
    skip_frames = max(1, len(time) // (fps * 10))
    frames = range(0, len(time), skip_frames)
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames,
                        interval=1000/fps, blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
    
    return anim


def plot_all_results(results: Dict, ball_mass: float, ball_radius: float,
                    surface_angle: float = 0.0, g: float = 9.81,
                    save_dir: Optional[str] = None):
    import os
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, 'trajectory.png') if save_dir else None
    plot_trajectory(results, surface_angle, save_path)
    
    save_path = os.path.join(save_dir, 'energy.png') if save_dir else None
    plot_energy(results, ball_mass, g, surface_angle, save_path)
    
    save_path = os.path.join(save_dir, 'velocity.png') if save_dir else None
    plot_velocity(results, save_path)
    
    save_path = os.path.join(save_dir, 'angular_velocity.png') if save_dir else None
    plot_angular_velocity(results, save_path)
    
    if 'is_slipping' in results:
        save_path = os.path.join(save_dir, 'slipping_regions.png') if save_dir else None
        plot_slipping_regions(results, save_path)
    
    plt.show()

