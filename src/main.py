
import numpy as np
from ball_physics import Ball, Surface
from simulation import Simulation, MultiballSimulation
from visualization import plot_all_results, create_animation, create_multiball_animation
import matplotlib.pyplot as plt


SPEED_OF_LIGHT = 3e8
MAX_REASONABLE_SPEED = 100
MAX_REASONABLE_MASS = 1000
MIN_REASONABLE_MASS = 0.001
MAX_REASONABLE_RADIUS = 10
MIN_REASONABLE_RADIUS = 0.001
MAX_REASONABLE_TIME = 3600
MAX_FRICTION_COEFF = 2.0
MIN_DENSITY = 10
MAX_DENSITY = 22000


def check_density(mass, radius):
    volume = (4/3) * np.pi * radius**3
    density = mass / volume
    
    if density < MIN_DENSITY:
        return f"Плотность {density:.1f} кг/м³ слишком мала! (< {MIN_DENSITY} кг/м³ - легче воздуха)"
    if density > MAX_DENSITY:
        return f"Плотность {density:.1f} кг/м³ слишком велика! (> {MAX_DENSITY} кг/м³ - больше осмия)"
    
    return True


def check_speed_physical(v):
    if v > SPEED_OF_LIGHT:
        return f"Скорость {v:.2e} м/с больше скорости света ({SPEED_OF_LIGHT:.2e} м/с)! Релятивистские эффекты не учтены."
    if v > MAX_REASONABLE_SPEED:
        return f"Скорость {v:.1f} м/с слишком велика для макроскопического шара! (разумный максимум: {MAX_REASONABLE_SPEED} м/с)"
    return True


def input_float(prompt, default=None, min_val=None, max_val=None, physical_check=None):
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (по умолчанию {default}): ").strip()
                value = float(user_input) if user_input else default
            else:
                value = float(input(f"{prompt}: "))
            
            if min_val is not None and value < min_val:
                print(f"❌ Значение должно быть >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"❌ Значение должно быть <= {max_val}")
                continue
            
            if physical_check is not None:
                check_result = physical_check(value)
                if check_result is not True:
                    print(f"❌ {check_result}")
                    continue
            
            return value
        except ValueError:
            print("❌ Пожалуйста, введите корректное число")


def input_yes_no(prompt, default=True):
    default_str = "д" if default else "н"
    while True:
        response = input(f"{prompt} (д/н, по умолчанию {default_str}): ").strip().lower()
        if not response:
            return default
        if response in ['д', 'да', 'y', 'yes']:
            return True
        if response in ['н', 'нет', 'n', 'no']:
            return False
        print("❌ Введите 'д' или 'н'")


def scenario1_incline():
    print("\n" + "="*70)
    print("СЦЕНАРИЙ 1: Скатывание по наклонной плоскости")
    print("="*70)
    
    print("\n📋 Введите параметры шара:")
    mass = input_float("  Масса (кг)", default=1.0, 
                       min_val=MIN_REASONABLE_MASS, max_val=MAX_REASONABLE_MASS)
    radius = input_float("  Радиус (м)", default=0.1, 
                        min_val=MIN_REASONABLE_RADIUS, max_val=MAX_REASONABLE_RADIUS)
    
    density_check = check_density(mass, radius)
    if density_check is not True:
        print(f"   ⚠️  {density_check}")
        volume = (4/3) * np.pi * radius**3
        density = mass / volume
        print(f"   💡 Справка: вода ≈ 1000 кг/м³, сталь ≈ 7800 кг/м³")
    
    print("\n📋 Введите параметры поверхности:")
    angle = input_float("  Угол наклона (градусы)", default=30.0, min_val=0.0, max_val=89.0)
    
    if angle > 75:
        print(f"   ⚠️  Угол {angle}° очень крутой!")
    
    friction = input_float("  Коэффициент трения", default=0.5, 
                          min_val=0.0, max_val=MAX_FRICTION_COEFF)
    
    if friction > 1.0:
        print(f"   💡 μ = {friction:.2f} > 1 (резина или клейкие поверхности)")
    
    print("\n📋 Параметры симуляции:")
    total_time = input_float("  Время симуляции (сек)", default=3.0, 
                            min_val=0.01, max_val=MAX_REASONABLE_TIME)
    
    ball = Ball(
        mass=mass,
        radius=radius,
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0])
    )
    
    surface = Surface(friction_coeff=friction, angle=angle)
    
    print("\n⏳ Запуск симуляции...")
    sim = Simulation(ball, surface, dt=0.01, total_time=total_time)
    sim.run()
    
    results = sim.get_results()
    
    print("\n✅ Симуляция завершена!")
    print(f"   📍 Начальная позиция: {results['position'][0]}")
    print(f"   📍 Конечная позиция: {results['position'][-1]}")
    print(f"   🚀 Конечная скорость: {np.linalg.norm(results['velocity'][-1]):.2f} м/с")
    print(f"   ⚡ Сохранение энергии: {'✅ Да' if sim.check_energy_conservation() else '❌ Нет'}")
    
    if any(results['is_slipping']):
        print(f"   ⚠️  Был переход в режим проскальзывания!")
    else:
        print(f"   ✅ Качение без проскальзывания")
    
    if input_yes_no("\n📊 Показать графики?", default=True):
        plot_all_results(results, mass, radius, surface_angle=angle)
        plt.show()


def scenario2_slipping():
    print("\n" + "="*70)
    print("СЦЕНАРИЙ 2: Переход в режим проскальзывания")
    print("="*70)
    
    print("\n💡 СПРАВКА:")
    print("   Проскальзывание происходит когда: tan(θ) > (7/2) × μ")
    print("   Для угла 45° нужно μ < 0.20")
    print("   Для угла 60° нужно μ < 0.25")
    
    print("\n📋 Введите параметры шара:")
    mass = input_float("  Масса (кг)", default=1.0, min_val=0.01)
    radius = input_float("  Радиус (м)", default=0.1, min_val=0.01)
    
    print("\n📋 Введите параметры поверхности:")
    angle = input_float("  Угол наклона (градусы)", default=45.0, min_val=10.0, max_val=90.0)
    
    angle_rad = np.radians(angle)
    mu_critical = (2.0/7.0) * np.tan(angle_rad)
    
    print(f"\n💡 Для угла {angle}°:")
    print(f"   Критический μ = {mu_critical:.3f}")
    print(f"   При μ < {mu_critical:.3f} будет проскальзывание")
    print(f"   При μ ≥ {mu_critical:.3f} будет качение без проскальзывания")
    
    friction = input_float(f"  Коэффициент трения (рекомендуется < {mu_critical:.2f})", 
                          default=0.1, min_val=0.0, max_val=1.0)
    
    will_slip = friction < mu_critical
    print(f"\n🔮 Прогноз: {'⚠️  БУДЕТ проскальзывание' if will_slip else '✅ Качение без проскальзывания'}")
    
    print("\n📋 Параметры симуляции:")
    total_time = input_float("  Время симуляции (сек)", default=2.0, min_val=0.1)
    
    ball = Ball(
        mass=mass,
        radius=radius,
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0])
    )
    
    surface = Surface(friction_coeff=friction, angle=angle)
    
    print("\n⏳ Запуск симуляции...")
    sim = Simulation(ball, surface, dt=0.01, total_time=total_time)
    sim.run()
    
    results = sim.get_results()
    
    print("\n✅ Симуляция завершена!")
    print(f"   📍 Пройденное расстояние: {results['position'][-1][0]:.2f} м")
    print(f"   🚀 Конечная скорость: {np.linalg.norm(results['velocity'][-1]):.2f} м/с")
    
    slipping_count = np.sum(results['is_slipping'])
    total_count = len(results['is_slipping'])
    slipping_percent = (slipping_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"\n📊 АНАЛИЗ РЕЖИМА ДВИЖЕНИЯ:")
    if slipping_count > 0:
        print(f"   ⚠️  Проскальзывание: {slipping_percent:.1f}% времени")
        print(f"   ✅ Качение: {100-slipping_percent:.1f}% времени")
    else:
        print(f"   ✅ Качение без проскальзывания на всём протяжении")
    
    if will_slip:
        a_slip = 9.81 * (np.sin(angle_rad) - friction * np.cos(angle_rad))
        print(f"\n📐 Теоретическое ускорение (проскальзывание): {a_slip:.2f} м/с²")
    else:
        a_roll = (5.0/7.0) * 9.81 * np.sin(angle_rad)
        print(f"\n📐 Теоретическое ускорение (качение): {a_roll:.2f} м/с²")
    
    if input_yes_no("\n📊 Показать графики?", default=True):
        plot_all_results(results, mass, radius, surface_angle=angle)
        plt.show()


def scenario3_horizontal():
    print("\n" + "="*70)
    print("СЦЕНАРИЙ 3: Качение по горизонтальной плоскости")
    print("="*70)
    
    print("\n📋 Введите параметры шара:")
    mass = input_float("  Масса (кг)", default=0.5, 
                       min_val=MIN_REASONABLE_MASS, max_val=MAX_REASONABLE_MASS)
    radius = input_float("  Радиус (м)", default=0.05, 
                        min_val=MIN_REASONABLE_RADIUS, max_val=MAX_REASONABLE_RADIUS)
    
    density_check = check_density(mass, radius)
    if density_check is not True:
        print(f"   ⚠️  {density_check}")
    
    print("\n📋 Начальная скорость:")
    vx = input_float("  Скорость по X (м/с)", default=3.0, 
                     min_val=-MAX_REASONABLE_SPEED, max_val=MAX_REASONABLE_SPEED,
                     physical_check=lambda v: check_speed_physical(abs(v)))
    vy = input_float("  Скорость по Y (м/с)", default=2.0, 
                     min_val=-MAX_REASONABLE_SPEED, max_val=MAX_REASONABLE_SPEED,
                     physical_check=lambda v: check_speed_physical(abs(v)))
    
    v_total = np.sqrt(vx**2 + vy**2)
    if v_total > 50:
        print(f"   ⚠️  Высокая скорость {v_total:.1f} м/с ({v_total*3.6:.1f} км/ч)!")
    
    print("\n📋 Параметры поверхности:")
    friction = input_float("  Коэффициент трения", default=0.3, 
                          min_val=0.0, max_val=MAX_FRICTION_COEFF)
    
    print("\n📋 Параметры симуляции:")
    total_time = input_float("  Время симуляции (сек)", default=5.0, 
                            min_val=0.01, max_val=MAX_REASONABLE_TIME)
    
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    wx = vy / radius if radius > 0 else 0.0
    wy = -vx / radius if radius > 0 else 0.0
    
    ball = Ball(
        mass=mass,
        radius=radius,
        position=np.array([0.0, 0.0]),
        velocity=np.array([vx, vy]),
        angular_velocity=np.array([wx, wy, 0.0])
    )
    
    surface = Surface(friction_coeff=friction, angle=0.0)
    
    print("\n⏳ Запуск симуляции...")
    sim = Simulation(ball, surface, dt=0.01, total_time=total_time)
    sim.run()
    
    results = sim.get_results()
    
    print("\n✅ Симуляция завершена!")
    print(f"   🚀 Начальная скорость: {v_magnitude:.2f} м/с")
    print(f"   🚀 Конечная скорость: {np.linalg.norm(results['velocity'][-1]):.2f} м/с")
    print(f"   📏 Пройденное расстояние: {np.linalg.norm(results['position'][-1] - results['position'][0]):.2f} м")
    
    if input_yes_no("\n📊 Показать графики?", default=True):
        plot_all_results(results, mass, radius, surface_angle=0.0)
        plt.show()


def scenario4_wall_collisions():
    print("\n" + "="*70)
    print("СЦЕНАРИЙ 4: Столкновения со стенами")
    print("="*70)
    
    print("\n📋 Введите параметры шара:")
    mass = input_float("  Масса (кг)", default=0.5, min_val=0.01)
    radius = input_float("  Радиус (м)", default=0.05, min_val=0.01)
    
    print("\n📋 Начальная скорость:")
    vx = input_float("  Скорость по X (м/с)", default=2.0)
    vy = input_float("  Скорость по Y (м/с)", default=1.5)
    
    print("\n📋 Параметры поверхности:")
    friction = input_float("  Коэффициент трения", default=0.1, min_val=0.0, max_val=1.0)
    
    print("\n📋 Границы области (стены):")
    boundary = input_float("  Размер области (±метры)", default=2.0, min_val=0.5)
    
    print("\n📋 Параметры столкновений:")
    restitution = input_float("  Коэффициент восстановления (0-1)", default=0.9, min_val=0.0, max_val=1.0)
    
    print("\n📋 Параметры симуляции:")
    total_time = input_float("  Время симуляции (сек)", default=10.0, min_val=0.1)
    
    wx = vy / radius if radius > 0 else 0.0
    wy = -vx / radius if radius > 0 else 0.0
    
    ball = Ball(
        mass=mass,
        radius=radius,
        position=np.array([0.0, 0.0]),
        velocity=np.array([vx, vy]),
        angular_velocity=np.array([wx, wy, 0.0])
    )
    
    surface = Surface(friction_coeff=friction, angle=0.0, bounds=[-boundary, boundary, -boundary, boundary])
    
    walls = [
        {'position': boundary, 'axis': 0},
        {'position': -boundary, 'axis': 0},
        {'position': boundary, 'axis': 1},
        {'position': -boundary, 'axis': 1}
    ]
    
    print("\n⏳ Запуск симуляции...")
    sim = Simulation(ball, surface, dt=0.01, total_time=total_time)
    sim.run(walls=walls, restitution=restitution)
    
    results = sim.get_results()
    
    print("\n✅ Симуляция завершена!")
    print(f"   ⚡ Начальная энергия: {results['energy'][0]:.3f} Дж")
    print(f"   ⚡ Конечная энергия: {results['energy'][-1]:.3f} Дж")
    print(f"   📉 Потеря энергии: {(1 - results['energy'][-1]/results['energy'][0])*100:.1f}%")
    
    if input_yes_no("\n📊 Показать графики?", default=True):
        plot_all_results(results, mass, radius, surface_angle=0.0)
        
        if input_yes_no("🎬 Показать анимацию?", default=True):
            anim = create_animation(results, radius, surface_angle=0.0, walls=walls)
            plt.show()
        else:
            plt.show()


def scenario5_multiple_balls():
    print("\n" + "="*70)
    print("СЦЕНАРИЙ 5: Столкновения нескольких шаров")
    print("="*70)
    
    print("\n📋 Количество шаров:")
    n_balls = int(input_float("  Введите количество шаров (2-5)", default=2, min_val=2, max_val=5))
    
    balls = []
    
    for i in range(n_balls):
        print(f"\n🔵 Параметры шара {i+1}:")
        mass = input_float(f"  Масса (кг)", default=1.0, min_val=0.01)
        radius = input_float(f"  Радиус (м)", default=0.1, min_val=0.01)
        
        print(f"  Начальная позиция:")
        x = input_float(f"    X (м)", default=float(i - n_balls//2))
        y = input_float(f"    Y (м)", default=0.0)
        
        print(f"  Начальная скорость:")
        vx = input_float(f"    Vx (м/с)", default=1.0 if i % 2 == 0 else -1.0)
        vy = input_float(f"    Vy (м/с)", default=0.0)
        
        ball = Ball(
            mass=mass,
            radius=radius,
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            angular_velocity=np.array([0.0, 0.0, 0.0])
        )
        balls.append(ball)
    
    print("\n📋 Параметры среды:")
    friction = input_float("  Коэффициент трения", default=0.05, min_val=0.0, max_val=1.0)
    boundary = input_float("  Размер области (±метры)", default=3.0, min_val=0.5)
    restitution = input_float("  Коэффициент восстановления (0-1)", default=1.0, min_val=0.0, max_val=1.0)
    total_time = input_float("  Время симуляции (сек)", default=8.0, min_val=0.1)
    
    surface = Surface(friction_coeff=friction, angle=0.0)
    
    walls = [
        {'position': boundary, 'axis': 0},
        {'position': -boundary, 'axis': 0},
        {'position': boundary, 'axis': 1},
        {'position': -boundary, 'axis': 1}
    ]
    
    print("\n⏳ Запуск симуляции...")
    sim = MultiballSimulation(balls, surface, dt=0.01, total_time=total_time)
    sim.run(walls=walls, restitution=restitution)
    
    results = sim.get_results()
    
    print("\n✅ Симуляция завершена!")
    print(f"   🔵 Количество шаров: {len(balls)}")
    print(f"   ⏱️  Время симуляции: {results['time'][-1]:.2f} с")
    
    if input_yes_no("\n📊 Показать анимацию?", default=True):
        radii = [ball.radius for ball in balls]
        anim = create_multiball_animation(results, radii, walls=walls)
        plt.show()


def scenario6_custom():
    print("\n" + "="*70)
    print("СЦЕНАРИЙ 6: Пользовательский сценарий")
    print("="*70)
    
    print("\n📋 Тип поверхности:")
    print("  1. Наклонная плоскость")
    print("  2. Горизонтальная плоскость")
    
    surface_type = int(input_float("  Выберите (1-2)", default=1, min_val=1, max_val=2))
    
    print("\n📋 Введите параметры шара:")
    mass = input_float("  Масса (кг)", default=1.0, min_val=0.01)
    radius = input_float("  Радиус (м)", default=0.1, min_val=0.01)
    
    print("\n📋 Начальное положение:")
    x0 = input_float("  X (м)", default=0.0)
    y0 = input_float("  Y (м)", default=0.0)
    
    print("\n📋 Начальная скорость:")
    vx0 = input_float("  Vx (м/с)", default=0.0)
    vy0 = input_float("  Vy (м/с)", default=0.0)
    
    print("\n📋 Параметры поверхности:")
    angle = input_float("  Угол наклона (градусы)", default=30.0 if surface_type == 1 else 0.0, min_val=0.0, max_val=90.0)
    friction = input_float("  Коэффициент трения", default=0.5, min_val=0.0, max_val=1.0)
    
    add_walls = input_yes_no("\n🧱 Добавить стены?", default=False)
    
    walls = None
    restitution = 1.0
    
    if add_walls:
        boundary = input_float("  Размер области (±метры)", default=5.0, min_val=0.5)
        restitution = input_float("  Коэффициент восстановления (0-1)", default=0.9, min_val=0.0, max_val=1.0)
        walls = [
            {'position': boundary, 'axis': 0},
            {'position': -boundary, 'axis': 0},
            {'position': boundary, 'axis': 1},
            {'position': -boundary, 'axis': 1}
        ]
    
    total_time = input_float("\n⏱️  Время симуляции (сек)", default=5.0, min_val=0.1)
    
    wx = vy0 / radius if radius > 0 else 0.0
    wy = -vx0 / radius if radius > 0 else 0.0
    
    ball = Ball(
        mass=mass,
        radius=radius,
        position=np.array([x0, y0]),
        velocity=np.array([vx0, vy0]),
        angular_velocity=np.array([wx, wy, 0.0])
    )
    
    surface = Surface(friction_coeff=friction, angle=angle)
    
    print("\n⏳ Запуск симуляции...")
    sim = Simulation(ball, surface, dt=0.01, total_time=total_time)
    sim.run(walls=walls, restitution=restitution)
    
    results = sim.get_results()
    
    print("\n✅ Симуляция завершена!")
    print(f"   📍 Конечная позиция: {results['position'][-1]}")
    print(f"   🚀 Конечная скорость: {np.linalg.norm(results['velocity'][-1]):.2f} м/с")
    print(f"   ⚡ Сохранение энергии: {'✅ Да' if sim.check_energy_conservation() else '❌ Нет'}")
    
    if input_yes_no("\n📊 Показать графики?", default=True):
        plot_all_results(results, mass, radius, surface_angle=angle)
        
        if input_yes_no("🎬 Показать анимацию?", default=False):
            anim = create_animation(results, radius, surface_angle=angle, walls=walls)
            plt.show()
        else:
            plt.show()


def main():
    print("\n" + "="*70)
    print("🎯  МОДЕЛИРОВАНИЕ ДВИЖЕНИЯ ШАРА ПО ПОВЕРХНОСТИ")
    print("="*70)
    
    while True:
        print("\n📋 Доступные сценарии:")
        print("  1. Скатывание по наклонной плоскости (без проскальзывания)")
        print("  2. Переход в режим проскальзывания ⚠️")
        print("  3. Качение по горизонтальной плоскости")
        print("  4. Столкновения со стенами")
        print("  5. Столкновения нескольких шаров")
        print("  6. Пользовательский сценарий")
        print("  0. Выход")
        
        try:
            choice = int(input_float("\nВыберите сценарий (0-6)", default=1, min_val=0, max_val=6))
            
            if choice == 0:
                print("\n👋 До свидания!")
                break
            elif choice == 1:
                scenario1_incline()
            elif choice == 2:
                scenario2_slipping()
            elif choice == 3:
                scenario3_horizontal()
            elif choice == 4:
                scenario4_wall_collisions()
            elif choice == 5:
                scenario5_multiple_balls()
            elif choice == 6:
                scenario6_custom()
            
            if not input_yes_no("\n🔄 Запустить другой сценарий?", default=True):
                print("\n👋 До свидания!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Программа прервана. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            if not input_yes_no("Попробовать снова?", default=True):
                break


if __name__ == "__main__":
    main()
