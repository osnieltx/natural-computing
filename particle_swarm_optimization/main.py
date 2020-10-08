import math
import random
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from math import sqrt, sin, pi
from matplotlib import pyplot as plt
from typing import Callable, List, Iterator
plt.style.use('dark_background')

from matplotlib.animation import FuncAnimation, FFMpegWriter


def problem(x: list):
    try:
        return Decimal(math.trunc(pow(2, -2 * pow((x[0]-Decimal(0.1))/Decimal(0.9), 2)) * pow(Decimal(sin(5 * Decimal(pi) * x[0])), 6) * 100)/100)
    except OverflowError:
        return 0.


def norm(vector: Iterator): return sqrt(sum(v_i ** 2 for v_i in vector))


@dataclass
class Particle:
    position: List[int]
    velocity: List[float]
    performance: float
    best_position: List[int]
    best_performance: float = None
    best_neighbor: "Particle" = None
    f: Callable = None

    def __init__(self, position, v_max, solution_space_dimension, f: Callable):
        # Initialize random velocity based on v_max
        velocity_vector = [Decimal(random.random() * 10) for _ in range(solution_space_dimension)]
        velocity_module = norm(velocity_vector)
        while - v_max > velocity_module or velocity_module > v_max:
            velocity_vector = [Decimal(random.random() * 10) for _ in range(solution_space_dimension)]
            velocity_module = norm(velocity_vector)
        self.velocity = velocity_vector

        self.v_max = v_max
        self.f = f
        self.set_position(position)

    def set_position(self, p):
        self.position = p
        self.performance = self.f(p)
        if not self.best_performance or self.performance > self.best_performance:
            self.best_performance = self.performance
            self.best_position = self.position

    def normalize_velocity(self):
        while norm(self.velocity) > self.v_max:
            self.velocity = [v_i + (-1 if v_i > 0 else 1) * Decimal(0.0001) for v_i in self.velocity]
        self.velocity = [Decimal(math.trunc(v * (10 ** 17)) / (10 ** 17)) for v in self.velocity]


def optimize(n_neighbors: int, max_it: int, individual_phi: float, neighborhood_phi: float, v_max: float, f: Callable,
             particles: List[List[int]], velocity_component=List):
    solution_space_dimension = len(particles[0])

    particles = [Particle(p, v_max, solution_space_dimension, f) for p in particles]
    history = [deepcopy(particles)]

    # Time loop
    for t in range(max_it):
        for p_i, particle in enumerate(particles):
            neigbors = [np for np_i, np in enumerate(particles) if np_i != p_i]
            neigbors.sort(key=lambda neighbor: norm(p - n for p, n in zip(particle.position, neighbor.position)))
            neigbors = neigbors[:n_neighbors]
            for neighbor in neigbors:
                if not particle.best_neighbor or neighbor.best_performance > particle.best_neighbor.performance:
                    particle.best_neighbor = neighbor

            op_components = zip(particle.position, particle.velocity, particle.best_position,
                                (particle.best_neighbor.best_position if particle.best_neighbor else particle.position))
            particle.velocity = [velocity_component[t] * v_i + individual_phi * (bp_i - p_i) + neighborhood_phi * (bnp_i - p_i)
                                 for p_i, v_i, bp_i, bnp_i in op_components]
            particle.normalize_velocity()
            particle.set_position([p_i + v_i for p_i, v_i in zip(particle.position, particle.velocity)])
        history.append(deepcopy(particles))

    return history

max_it = 100
v_max = Decimal(.1)
velocity_component = [Decimal(i/1000) for i in range(0, 1000, 1000//max_it)][::-1]
n_neighbors, n_particles = 9, 30

optimization_by_individual_phi = []
for individual_phi in range(0, 11, 2):
    individual_phi = Decimal(individual_phi/10.)
    particles = [[Decimal((random.random() * 4) - 2)] for i in range(n_particles)]
    print('--------------------------------------------')
    print(f'Analysing individual_phi {individual_phi:.2}')
    optimization_by_neighborhood_phi = []
    for neighborhood_phi in range(0, 11, 2):
        neighborhood_phi = Decimal(neighborhood_phi/10)
        print('-------------------------------')
        print(f'Analysing group_phi {neighborhood_phi}')
        optimization = []
        for i in range(10):
            a_optimization = optimize(n_neighbors, max_it, individual_phi, neighborhood_phi, v_max, problem,
                                      particles, velocity_component)[-1]
            optimization.append(max(p.best_performance for p in a_optimization))
            if (i+1) % 25 == 0:
                print(f"{i+1}% completed.")
        optimization_by_neighborhood_phi.append(sum(optimization) / 10)
    optimization_by_individual_phi.append(optimization_by_neighborhood_phi)

print(optimization_by_individual_phi)
#
# fig = plt.figure()
# ax = plt.axes(xlim=(-2, 2), ylim=(-.5, 1.5))
# x_values = [Decimal(x/1000) for x in range(-2000, 2000, 4)]
# base_function_line, = ax.plot(x_values, [problem([x]) for x in x_values], color='white')
# scat = ax.scatter([], [], color='green')
# # best_scat = ax.scatter([], [], color='lightgreen')
# quiver = ax.quiver([], [], [], [])
# tm = ax.text(-2, 1.6, '')
#
#
# def init():
#     scat.set_offsets([])
#     quiver.set_offsets([])
#     # best_scat.set_offsets([])
#     return scat, quiver, tm
#
#
# def animate(i):
#     # best_scat.set_offsets([[p.best_position[0], p.best_performance] for p in optimization_history[i]])
#     scat.set_offsets([[p.position[0], p.performance] for p in optimization_history[i]])
#
#     global quiver
#     quiver.remove()
#     quiver = ax.quiver([float(p.position[0]) for p in optimization_history[i]],
#                        [float(p.performance) for p in optimization_history[i]],
#                        [float(p.velocity[0]) for p in optimization_history[i]],
#                        [0 for p in optimization_history[i]], color='green')
#
#     tm.set_text(f'time {i}')
#
#     return scat, quiver, tm
#
# anim = FuncAnimation(fig, animate, init_func=init, frames=max_it, repeat=True, interval=300)
#
# mywriter = FFMpegWriter()
# anim.save('particle_swarm_optimization.mp4', writer=mywriter)
#
# for t in optimization_history:
#     print(sum(abs(p.velocity[0]) for p in t)/len(t))
#
#
#
#
#
#
