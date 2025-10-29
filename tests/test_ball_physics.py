
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ball_physics import Ball, Surface, BallDynamics
from src.simulation import Simulation


class TestBallPhysics(unittest.TestCase):
    
    def test_kinetic_energy(self):
        mass = 2.0
        radius = 0.1
        velocity = np.array([3.0, 4.0])
        angular_velocity = np.array([0.0, 0.0, 10.0])
        
        ball = Ball(mass, radius, np.array([0.0, 0.0]), velocity, angular_velocity)
        
        v_squared = np.sum(velocity**2)
        w_squared = np.sum(angular_velocity**2)
        I = 0.4 * mass * radius**2
        
        E_k_analytical = 0.5 * mass * v_squared + 0.5 * I * w_squared
        
        E_k_computed = ball.kinetic_energy()
        
        self.assertAlmostEqual(E_k_computed, E_k_analytical, places=10,
                              msg="Кинетическая энергия не совпадает с аналитическим решением")
    
    def test_moment_of_inertia(self):
        mass = 1.0
        radius = 0.5
        
        ball = Ball(mass, radius, np.array([0.0, 0.0]), 
                   np.array([0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        
        I_analytical = 0.4 * mass * radius**2
        
        self.assertAlmostEqual(ball.moment_of_inertia, I_analytical, places=10,
                              msg="Момент инерции не совпадает с аналитическим решением")


class TestInclineMotion(unittest.TestCase):
    
    def test_acceleration_no_slip(self):
        mass = 1.0
        radius = 0.1
        angle = 30.0
        g = 9.81
        
        ball = Ball(mass, radius, np.array([0.0, 0.0]), 
                   np.array([0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        
        surface = Surface(friction_coeff=0.7, angle=angle)
        
        sim = Simulation(ball, surface, dt=0.001, total_time=0.5, g=g)
        sim.run()
        
        results = sim.get_results()
        
        angle_rad = np.radians(angle)
        a_analytical = (5.0/7.0) * g * np.sin(angle_rad)
        
        time_idx = np.argmin(np.abs(results['time'] - 0.1))
        v_computed = results['velocity'][time_idx, 0]
        t = results['time'][time_idx]
        a_computed = v_computed / t
        
        relative_error = abs(a_computed - a_analytical) / a_analytical
        
        self.assertLess(relative_error, 0.05,
                       msg=f"Ускорение {a_computed:.3f} не совпадает с аналитическим {a_analytical:.3f}")
    
    def test_final_velocity_no_slip(self):
        mass = 1.0
        radius = 0.1
        angle = 30.0
        distance = 2.0
        g = 9.81
        
        ball = Ball(mass, radius, np.array([0.0, 0.0]), 
                   np.array([0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        
        surface = Surface(friction_coeff=0.7, angle=angle)
        
        sim = Simulation(ball, surface, dt=0.01, total_time=5.0, g=g)
        sim.run()
        
        results = sim.get_results()
        
        distances = results['position'][:, 0]
        idx = np.argmin(np.abs(distances - distance))
        
        v_computed = np.linalg.norm(results['velocity'][idx])
        
        angle_rad = np.radians(angle)
        h = distance * np.sin(angle_rad)
        v_analytical = np.sqrt((10.0/7.0) * g * h)
        
        relative_error = abs(v_computed - v_analytical) / v_analytical
        
        self.assertLess(relative_error, 0.1,
                       msg=f"Скорость {v_computed:.3f} не совпадает с аналитической {v_analytical:.3f}")


class TestEnergyConservation(unittest.TestCase):
    
    def test_energy_conservation_incline(self):
        mass = 1.0
        radius = 0.1
        angle = 30.0
        g = 9.81
        
        ball = Ball(mass, radius, np.array([0.0, 0.0]), 
                   np.array([0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        
        surface = Surface(friction_coeff=0.7, angle=angle)
        
        sim = Simulation(ball, surface, dt=0.01, total_time=2.0, g=g)
        sim.run()
        
        results = sim.get_results()
        
        angle_rad = np.radians(angle)
        kinetic = results['energy']
        height_change = -results['position'][:, 0] * np.sin(angle_rad)
        potential = mass * g * height_change
        total_energy = kinetic + potential
        
        E_initial = total_energy[0]
        
        E_mean = np.mean(total_energy)
        if E_mean > 1e-10:
            energy_deviation = (np.max(total_energy) - np.min(total_energy)) / E_mean
        else:
            energy_deviation = np.max(total_energy) - np.min(total_energy)
        
        self.assertLess(energy_deviation, 0.05,
                       msg=f"Энергия не сохраняется, отклонение {energy_deviation*100:.2f}%")
    
    def test_energy_loss_with_friction(self):
        mass = 0.5
        radius = 0.05
        v0 = 2.0
        g = 9.81
        
        ball = Ball(mass, radius, np.array([0.0, 0.0]), 
                   np.array([v0, 0.0]), 
                   np.array([0.0, 0.0, -v0/radius]))
        
        surface = Surface(friction_coeff=0.3, angle=0.0)
        
        sim = Simulation(ball, surface, dt=0.01, total_time=5.0, g=g)
        sim.run()
        
        results = sim.get_results()
        
        E_initial = results['energy'][0]
        E_final = results['energy'][-1]
        
        self.assertLess(E_final, E_initial,
                       msg="Энергия должна уменьшаться из-за трения")
        
        v_final = np.linalg.norm(results['velocity'][-1])
        self.assertLess(v_final, 0.1,
                       msg=f"Шар должен остановиться, но v_final = {v_final:.3f}")


class TestCollisions(unittest.TestCase):
    
    def test_elastic_wall_collision(self):
        mass = 1.0
        radius = 0.1
        v0 = 2.0
        
        ball = Ball(mass, radius, np.array([0.0, 0.0]), 
                   np.array([v0, 0.0]), np.array([0.0, 0.0, 0.0]))
        
        surface = Surface(friction_coeff=0.05, angle=0.0)
        
        walls = [{'position': 1.0, 'axis': 0}]
        
        sim = Simulation(ball, surface, dt=0.01, total_time=2.0)
        sim.run(walls=walls, restitution=1.0)
        
        results = sim.get_results()
        
        velocities_x = results['velocity'][:, 0]
        
        has_sign_change = np.any(velocities_x[:-1] * velocities_x[1:] < 0)
        
        self.assertTrue(has_sign_change,
                       msg="Скорость должна изменить знак при столкновении со стеной")
    
    def test_momentum_conservation_ball_collision(self):
        from src.simulation import MultiballSimulation
        
        mass = 1.0
        radius = 0.1
        
        ball1 = Ball(mass, radius, np.array([-0.5, 0.0]), 
                    np.array([1.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        
        ball2 = Ball(mass, radius, np.array([0.5, 0.0]), 
                    np.array([-1.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        
        surface = Surface(friction_coeff=0.0, angle=0.0)
        
        p_initial = ball1.linear_momentum() + ball2.linear_momentum()
        
        sim = MultiballSimulation([ball1, ball2], surface, dt=0.001, total_time=1.0)
        sim.run(restitution=1.0)
        
        p_final = ball1.linear_momentum() + ball2.linear_momentum()
        
        momentum_error = np.linalg.norm(p_final - p_initial)
        
        self.assertLess(momentum_error, 0.1,
                       msg=f"Импульс не сохраняется, ошибка {momentum_error:.3f}")


class TestSlippingCondition(unittest.TestCase):
    
    def test_slipping_on_steep_incline(self):
        mass = 1.0
        radius = 0.1
        angle = 60.0
        friction = 0.1
        
        
        ball = Ball(mass, radius, np.array([0.0, 0.0]), 
                   np.array([0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        
        surface = Surface(friction_coeff=friction, angle=angle)
        
        sim = Simulation(ball, surface, dt=0.01, total_time=1.0)
        sim.run()
        
        results = sim.get_results()
        
        has_slipping = np.any(results['is_slipping'])
        
        self.assertTrue(has_slipping,
                       msg="На крутом склоне с малым трением должно быть проскальзывание")


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestBallPhysics))
    suite.addTests(loader.loadTestsFromTestCase(TestInclineMotion))
    suite.addTests(loader.loadTestsFromTestCase(TestEnergyConservation))
    suite.addTests(loader.loadTestsFromTestCase(TestCollisions))
    suite.addTests(loader.loadTestsFromTestCase(TestSlippingCondition))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    print("\n" + "="*70)
    print("СТАТИСТИКА ТЕСТОВ")
    print("="*70)
    print(f"Всего тестов: {result.testsRun}")
    print(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Провалено: {len(result.failures)}")
    print(f"Ошибки: {len(result.errors)}")
    print("="*70)

