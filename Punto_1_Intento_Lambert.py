    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    # Constants
    mu_sun = 1.327e11  # km³/s²
    days_to_sec = 86400

    # Initial conditions (from problem)
    r_earth = np.array([-1.3112578338e8, -7.5507634192e7])
    v_earth = np.array([-14.330508015, -25.964546881])
    r_mars = np.array([-2.309032001186e8, 9.4520480368e7])
    v_mars = np.array([-8.3495579277, -20.320213209])

    # Time settings
    t_max = 250 * days_to_sec  # ~8 months (Hohmann transfer time)
    dt = 3600  # 1-hour steps

    # Equations of motion
    def equations_of_motion(t, y):
        r = y[:2]
        v = y[2:]
        r_norm = np.linalg.norm(r)
        drdt = v
        dvdt = -mu_sun * r / r_norm**3
        return np.concatenate((drdt, dvdt))

    # RK4 integrator
    def rk4_step(t, y, dt, func):
        k1 = func(t, y)
        k2 = func(t + dt/2, y + dt/2 * k1)
        k3 = func(t + dt/2, y + dt/2 * k2)
        k4 = func(t + dt, y + dt * k3)
        return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    # Objective: Minimize distance to Mars at t_max
    def objective(v0_guess):
        y0 = np.concatenate((r_earth, v_earth + v0_guess))
        t, y = 0, y0.copy()
        while t < t_max:
            y = rk4_step(t, y, dt, equations_of_motion)
            t += dt
        return np.linalg.norm(y[:2] - r_mars)

    # Initial guess (Hohmann transfer delta-v in prograde direction)
    v_earth_unit = v_earth / np.linalg.norm(v_earth)  # Earth's velocity direction
    v0_guess_hohmann = 2.95 * v_earth_unit  # Prograde burn

    # Optimize
    result = minimize(
        objective,
        v0_guess_hohmann,
        method='Nelder-Mead',  # Robust to noisy objectives
        options={'maxiter': 1000, 'xatol': 1e-4}
    )
    v0_optimal = result.x
    print(f"Optimal Δv (relative to Earth): {v0_optimal} km/s")
    print(f"Magnitude: {np.linalg.norm(v0_optimal):.2f} km/s")

    # Simulate final trajectory
    y0 = np.concatenate((r_earth, v_earth + v0_optimal))
    t, y = 0, y0.copy()
    trajectory = [y0[:2]]
    while t < t_max:
        y = rk4_step(t, y, dt, equations_of_motion)
        trajectory.append(y[:2])
        t += dt
    trajectory = np.array(trajectory)

    # Plot
    plt.figure(figsize=(12, 10))
    plt.plot(r_earth[0], r_earth[1], 'bo', label='Earth (departure)')
    plt.plot(r_mars[0], r_mars[1], 'ro', label='Mars (arrival)')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'g-', label='Spacecraft')
    plt.plot(0, 0, 'yo', label='Sun')
    plt.xlabel('x (km)'); plt.ylabel('y (km)')
    plt.title('Earth-to-Mars Transfer Trajectory')
    plt.legend(); plt.grid(); plt.axis('equal')
    plt.show()

    # Verify Mars arrival
    final_pos = trajectory[-1]
    distance_to_mars = np.linalg.norm(final_pos - r_mars)
    print(f"Final distance to Mars: {distance_to_mars:.2f} km")
