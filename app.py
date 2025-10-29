
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.ball_physics import Ball, Surface
from src.simulation import Simulation, MultiballSimulation
from src.visualization import plot_trajectory, plot_energy, plot_velocity, plot_angular_velocity, plot_slipping_regions, create_animation, create_multiball_animation
import io
import os
import tempfile

st.set_page_config(
    page_title="Моделирование движения шара",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

SPEED_OF_LIGHT = 3e8
MAX_REASONABLE_SPEED = 100.0
MAX_REASONABLE_MASS = 1000.0
MIN_REASONABLE_MASS = 0.001
MAX_REASONABLE_RADIUS = 10.0
MIN_REASONABLE_RADIUS = 0.001
MAX_REASONABLE_TIME = 3600.0
MAX_FRICTION_COEFF = 2.0
MIN_DENSITY = 10.0
MAX_DENSITY = 22000.0


def check_density(mass, radius):
    volume = (4/3) * np.pi * radius**3
    density = mass / volume
    
    if density < MIN_DENSITY:
        return False, f"⚠️ Плотность {density:.1f} кг/м³ слишком мала!"
    if density > MAX_DENSITY:
        return False, f"⚠️ Плотность {density:.1f} кг/м³ слишком велика!"
    
    return True, f"✅ Плотность: {density:.1f} кг/м³"


def show_animation(results, ball_radius, surface_angle=0.0, walls=None):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmp:
        anim = create_animation(results, ball_radius, surface_angle, walls, save_path=tmp.name, fps=15)
        plt.close()
        
        if os.path.exists(tmp.name):
            st.image(tmp.name)
            os.unlink(tmp.name)


def main():
    
    st.title("⚽ Моделирование движения шара по поверхности")
    st.markdown("---")
    
    st.sidebar.title("📋 Выбор сценария")
    
    scenario = st.sidebar.radio(
        "Выберите сценарий:",
        [
            "1️⃣ Скатывание по наклонной",
            "2️⃣ Проскальзывание",
            "3️⃣ Качение по горизонтали",
            "4️⃣ Столкновения со стенами",
            "5️⃣ Несколько шаров"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Подсказка:** Все параметры имеют физически обоснованные ограничения")
    
    if "1️⃣" in scenario:
        scenario_1_incline()
    elif "2️⃣" in scenario:
        scenario_2_slipping()
    elif "3️⃣" in scenario:
        scenario_3_horizontal()
    elif "4️⃣" in scenario:
        scenario_4_walls()
    elif "5️⃣" in scenario:
        scenario_5_multiball()


def scenario_1_incline():
    st.header("1️⃣ Скатывание по наклонной плоскости")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Параметры шара")
        mass = st.slider("Масса (кг)", MIN_REASONABLE_MASS, MAX_REASONABLE_MASS, 1.0, 
                        help="От 1 грамма до 1 тонны")
        radius = st.slider("Радиус (м)", MIN_REASONABLE_RADIUS, MAX_REASONABLE_RADIUS, 0.1, 
                          help="От 1 мм до 10 м")
        
        is_valid, message = check_density(mass, radius)
        if is_valid:
            st.success(message)
        else:
            st.warning(message)
            st.info("💡 Вода ≈ 1000 кг/м³, сталь ≈ 7800 кг/м³")
    
    with col2:
        st.subheader("🏔️ Параметры поверхности")
        angle = st.slider("Угол наклона (градусы)", 0.0, 89.0, 30.0,
                         help="От горизонтали до вертикали")
        
        if angle > 75:
            st.warning("⚠️ Очень крутой склон!")
        
        friction = st.slider("Коэффициент трения", 0.0, MAX_FRICTION_COEFF, 0.5,
                            help="0 = лёд, 1 = резина")
        
        if friction > 1.0:
            st.info("💡 μ > 1 для резины или клейких поверхностей")
    
    st.subheader("⏱️ Параметры симуляции")
    total_time = st.slider("Время симуляции (сек)", 0.1, float(MAX_REASONABLE_TIME), 3.0)
    
    if st.button("🚀 Запустить симуляцию", type="primary"):
        run_simulation_incline(mass, radius, angle, friction, total_time)


def scenario_2_slipping():
    st.header("2️⃣ Переход в режим проскальзывания")
    
    st.info("💡 **Справка:** Проскальзывание происходит когда tan(θ) > (7/2) × μ")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Параметры шара")
        mass = st.slider("Масса (кг)", MIN_REASONABLE_MASS, MAX_REASONABLE_MASS, 1.0, key="slip_mass")
        radius = st.slider("Радиус (м)", MIN_REASONABLE_RADIUS, MAX_REASONABLE_RADIUS, 0.1, key="slip_radius")
    
    with col2:
        st.subheader("🏔️ Параметры поверхности")
        angle = st.slider("Угол наклона (градусы)", 10.0, 89.0, 45.0, key="slip_angle")
        
        angle_rad = np.radians(angle)
        mu_critical = (2.0/7.0) * np.tan(angle_rad)
        
        st.info(f"📐 Критический μ для {angle}°: **{mu_critical:.3f}**")
        st.write(f"• μ < {mu_critical:.3f} → проскальзывание")
        st.write(f"• μ ≥ {mu_critical:.3f} → качение")
        
        friction = st.slider("Коэффициент трения", 0.0, MAX_FRICTION_COEFF, 0.1, key="slip_friction")
        
        if friction < mu_critical:
            st.error("🔮 Прогноз: **БУДЕТ проскальзывание!**")
        else:
            st.success("🔮 Прогноз: **Качение без проскальзывания**")
    
    total_time = st.slider("Время симуляции (сек)", 0.1, float(MAX_REASONABLE_TIME), 2.0, key="slip_time")
    
    if st.button("🚀 Запустить симуляцию", type="primary", key="slip_run"):
        run_simulation_slipping(mass, radius, angle, friction, total_time)


def scenario_3_horizontal():
    st.header("3️⃣ Качение по горизонтальной плоскости")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Параметры шара")
        mass = st.slider("Масса (кг)", MIN_REASONABLE_MASS, MAX_REASONABLE_MASS, 0.5, key="hor_mass")
        radius = st.slider("Радиус (м)", MIN_REASONABLE_RADIUS, MAX_REASONABLE_RADIUS, 0.05, key="hor_radius")
        
        st.subheader("🚀 Начальная скорость")
        vx = st.slider("Скорость по X (м/с)", -MAX_REASONABLE_SPEED, MAX_REASONABLE_SPEED, 3.0, key="hor_vx")
        vy = 0.0
        
        st.info("ℹ️ На горизонтальной поверхности Y координата (высота) не меняется, поэтому vy = 0")
        st.metric("Скорость", f"{abs(vx):.2f} м/с", f"{abs(vx)*3.6:.1f} км/ч")
        
        if abs(vx) > 50:
            st.warning(f"⚠️ Высокая скорость!")
    
    with col2:
        st.subheader("🏔️ Параметры поверхности")
        friction = st.slider("Коэффициент трения", 0.0, MAX_FRICTION_COEFF, 0.3, key="hor_friction")
        
        st.subheader("⏱️ Время симуляции")
        total_time = st.slider("Время (сек)", 0.1, float(MAX_REASONABLE_TIME), 5.0, key="hor_time")
    
    if st.button("🚀 Запустить симуляцию", type="primary", key="hor_run"):
        run_simulation_horizontal(mass, radius, vx, vy, friction, total_time)


def scenario_4_walls():
    st.header("4️⃣ Столкновения со стенами")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Параметры шара")
        mass = st.slider("Масса (кг)", MIN_REASONABLE_MASS, MAX_REASONABLE_MASS, 0.5, key="wall_mass")
        radius = st.slider("Радиус (м)", MIN_REASONABLE_RADIUS, MAX_REASONABLE_RADIUS, 0.05, key="wall_radius")
        
        st.subheader("🚀 Начальная скорость")
        vx = st.slider("Скорость по X (м/с)", -MAX_REASONABLE_SPEED, MAX_REASONABLE_SPEED, 2.0, key="wall_vx")
        vy = 0.0
        
        st.info("ℹ️ Движение на горизонтали: vy = 0")
    
    with col2:
        st.subheader("🧱 Параметры стен")
        boundary = st.slider("Размер области (±метры)", 0.5, 10.0, 2.0, key="wall_boundary")
        
        st.subheader("💥 Столкновения")
        restitution = st.slider("Коэффициент восстановления", 0.0, 1.0, 0.9,
                               help="0 = неупругий, 1 = упругий", key="wall_rest")
        
        friction = st.slider("Коэффициент трения", 0.0, MAX_FRICTION_COEFF, 0.1, key="wall_friction")
        
        st.subheader("⏱️ Время")
        total_time = st.slider("Время (сек)", 0.1, float(MAX_REASONABLE_TIME), 10.0, key="wall_time")
    
    if st.button("🚀 Запустить симуляцию", type="primary", key="wall_run"):
        walls = [
            {'position': boundary, 'axis': 0},
            {'position': -boundary, 'axis': 0},
            {'position': boundary, 'axis': 1},
            {'position': -boundary, 'axis': 1}
        ]
        run_simulation_walls(mass, radius, vx, vy, friction, boundary, walls, restitution, total_time)


def scenario_5_multiball():
    st.header("5️⃣ Столкновения нескольких шаров")
    
    n_balls = st.slider("Количество шаров", 2, 5, 2, key="multi_n")
    
    st.warning("⚠️ Упрощенная версия: все шары с одинаковыми параметрами")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Параметры шаров")
        mass = st.slider("Масса (кг)", MIN_REASONABLE_MASS, MAX_REASONABLE_MASS, 1.0, key="multi_mass")
        radius = st.slider("Радиус (м)", MIN_REASONABLE_RADIUS, MAX_REASONABLE_RADIUS, 0.1, key="multi_radius")
    
    with col2:
        st.subheader("🌐 Параметры среды")
        friction = st.slider("Коэффициент трения", 0.0, MAX_FRICTION_COEFF, 0.05, key="multi_friction")
        boundary = st.slider("Размер области (±м)", 1.0, 10.0, 3.0, key="multi_boundary")
        restitution = st.slider("Коэффициент восстановления", 0.0, 1.0, 1.0, key="multi_rest")
        total_time = st.slider("Время (сек)", 0.1, float(MAX_REASONABLE_TIME), 8.0, key="multi_time")
    
    if st.button("🚀 Запустить симуляцию", type="primary", key="multi_run"):
        run_simulation_multiball(n_balls, mass, radius, friction, boundary, restitution, total_time)


def run_simulation_incline(mass, radius, angle, friction, total_time):
    with st.spinner("⏳ Выполняется симуляция..."):
        ball = Ball(mass, radius, [0, 0], [0, 0], [0, 0, 0])
        surface = Surface(friction_coeff=friction, angle=angle)
        sim = Simulation(ball, surface, dt=0.01, total_time=total_time)
        sim.run()
        
        results = sim.get_results()
        
        st.success("✅ Симуляция завершена!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Конечная скорость", f"{np.linalg.norm(results['velocity'][-1]):.2f} м/с")
        with col2:
            st.metric("Пройденное расстояние", f"{results['position'][-1][0]:.2f} м")
        with col3:
            energy_conserved = "✅ Да" if sim.check_energy_conservation() else "❌ Нет"
            st.metric("Сохранение энергии", energy_conserved)
        
        if any(results['is_slipping']):
            st.warning("⚠️ Был переход в режим проскальзывания!")
        else:
            st.success("✅ Качение без проскальзывания")
        
        display_plots(results, mass, radius, angle, "incline")


def run_simulation_slipping(mass, radius, angle, friction, total_time):
    with st.spinner("⏳ Выполняется симуляция..."):
        ball = Ball(mass, radius, [0, 0], [0, 0], [0, 0, 0])
        surface = Surface(friction_coeff=friction, angle=angle)
        sim = Simulation(ball, surface, dt=0.01, total_time=total_time)
        sim.run()
        
        results = sim.get_results()
        
        st.success("✅ Симуляция завершена!")
        
        slipping_count = np.sum(results['is_slipping'])
        total_count = len(results['is_slipping'])
        slipping_percent = (slipping_count / total_count) * 100 if total_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Конечная скорость", f"{np.linalg.norm(results['velocity'][-1]):.2f} м/с")
        with col2:
            st.metric("Пройденное расстояние", f"{results['position'][-1][0]:.2f} м")
        with col3:
            st.metric("Проскальзывание", f"{slipping_percent:.1f}% времени")
        
        if slipping_count > 0:
            st.warning(f"⚠️ Проскальзывание: {slipping_percent:.1f}% времени")
        else:
            st.success("✅ Качение без проскальзывания")
        
        display_plots(results, mass, radius, angle, "slipping")


def run_simulation_horizontal(mass, radius, vx, vy, friction, total_time):
    with st.spinner("⏳ Выполняется симуляция..."):
        wx = 0.0
        wy = -vx / radius if radius > 0 else 0.0
        
        ball = Ball(mass, radius, [0, 0], [vx, vy], [wx, wy, 0.0])
        surface = Surface(friction_coeff=friction, angle=0.0)
        sim = Simulation(ball, surface, dt=0.01, total_time=total_time)
        sim.run()
        
        results = sim.get_results()
        
        st.success("✅ Симуляция завершена!")
        
        v_initial = abs(vx)
        v_final = np.linalg.norm(results['velocity'][-1])
        distance = np.linalg.norm(results['position'][-1] - results['position'][0])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Начальная скорость", f"{v_initial:.2f} м/с")
        with col2:
            st.metric("Конечная скорость", f"{v_final:.2f} м/с")
        with col3:
            st.metric("Пройденное расстояние", f"{distance:.2f} м")
        
        display_plots(results, mass, radius, 0.0, "horizontal")


def run_simulation_walls(mass, radius, vx, vy, friction, boundary, walls, restitution, total_time):
    with st.spinner("⏳ Выполняется симуляция..."):
        wx = 0.0
        wy = -vx / radius if radius > 0 else 0.0
        
        ball = Ball(mass, radius, [0, 0], [vx, vy], [wx, wy, 0.0])
        surface = Surface(friction_coeff=friction, angle=0.0, bounds=[-boundary, boundary, -boundary, boundary])
        sim = Simulation(ball, surface, dt=0.01, total_time=total_time)
        sim.run(walls=walls, restitution=restitution)
        
        results = sim.get_results()
        
        st.success("✅ Симуляция завершена!")
        
        E_initial = results['energy'][0]
        E_final = results['energy'][-1]
        energy_loss = (1 - E_final/E_initial)*100 if E_initial > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Начальная энергия", f"{E_initial:.3f} Дж")
        with col2:
            st.metric("Конечная энергия", f"{E_final:.3f} Дж")
        with col3:
            st.metric("Потеря энергии", f"{energy_loss:.1f}%")
        
        display_plots(results, mass, radius, 0.0, "walls", walls=walls)


def run_simulation_multiball(n_balls, mass, radius, friction, boundary, restitution, total_time):
    with st.spinner("⏳ Выполняется симуляция..."):
        balls = []
        
        for i in range(n_balls):
            x = (i - n_balls//2) * 0.5
            vx = 1.0 if i % 2 == 0 else -1.0
            ball = Ball(mass, radius, [x, 0], [vx, 0], [0, 0, 0])
            balls.append(ball)
        
        surface = Surface(friction_coeff=friction, angle=0.0)
        
        walls = [
            {'position': boundary, 'axis': 0},
            {'position': -boundary, 'axis': 0},
            {'position': boundary, 'axis': 1},
            {'position': -boundary, 'axis': 1}
        ]
        
        sim = MultiballSimulation(balls, surface, dt=0.01, total_time=total_time)
        sim.run(walls=walls, restitution=restitution)
        
        results = sim.get_results()
        
        st.success("✅ Симуляция завершена!")
        st.metric("Количество шаров", n_balls)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.rainbow(np.linspace(0, 1, n_balls))
        for i, positions in enumerate(results['positions']):
            ax.plot(positions[:, 0], positions[:, 1], 
                   label=f'Шар {i+1}', color=colors[i], alpha=0.7)
            ax.plot(positions[0, 0], positions[0, 1], 'o', color=colors[i], markersize=10)
        
        ax.axvline(x=boundary, color='gray', linewidth=2, linestyle='--')
        ax.axvline(x=-boundary, color='gray', linewidth=2, linestyle='--')
        ax.axhline(y=boundary, color='gray', linewidth=2, linestyle='--')
        ax.axhline(y=-boundary, color='gray', linewidth=2, linestyle='--')
        
        ax.set_xlabel('X (м)', fontsize=12)
        ax.set_ylabel('Y (м)', fontsize=12)
        ax.set_title('Траектории шаров', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')
        
        st.pyplot(fig)
        plt.close()


def display_plots(results, mass, radius, angle, prefix, walls=None):
    st.subheader("📊 Результаты симуляции")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Траектория", "Энергия", "Скорость", "Угловая скорость", "Режимы", "🎬 Анимация"])
    
    with tab1:
        fig = plot_trajectory(results, surface_angle=angle)
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        fig = plot_energy(results, mass, surface_angle=angle)
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        fig = plot_velocity(results)
        st.pyplot(fig)
        plt.close()
    
    with tab4:
        fig = plot_angular_velocity(results)
        st.pyplot(fig)
        plt.close()
    
    with tab5:
        if 'is_slipping' in results:
            fig = plot_slipping_regions(results)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Данные о проскальзывании недоступны")
    
    with tab6:
        st.info("🎬 Создание анимации (может занять несколько секунд)...")
        show_animation(results, radius, angle, walls)


if __name__ == "__main__":
    main()

