def homogenous_poisson(intensity,T)
"""Simulates a homogenous Poisson Process starting at 0 and ending at time T
params:
    intensity: value of the instensity \lambda
    T: Time of end of simulation
returns:
    t_array: list of instants
"""
    t_array = [0]
    while True:
        u = np.random.rand()
        w = -np.log(u) / l
        t_array.append(t_array[-1] + w)
        if t_array[-1] > T:
            return t_array[1:-1]

