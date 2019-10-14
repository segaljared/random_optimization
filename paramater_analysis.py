import numpy
import time
import matplotlib.pyplot as plot


def run_parameter_analysis(create_problem, optimizer, parameter_values, random_seed, title, log_scale=False):
    fitness = []
    fitness_min = []
    fitness_max = []
    elapsed_time = []
    for value in parameter_values:
        numpy.random.seed(random_seed)
        total_time = 0
        parameter_fitness = []
        for i in range(0, 10):
            initial_state, problem = create_problem()
            start = time.time()
            _, best_fitness, _ = optimizer(problem, initial_state, value)
            total_time += time.time() - start
            parameter_fitness.append(best_fitness)
        fitness.append(numpy.mean(parameter_fitness))
        fitness_min.append(numpy.min(parameter_fitness))
        fitness_max.append(numpy.max(parameter_fitness))
        elapsed = total_time / 10
        elapsed_time.append(elapsed * 1000)
    plot.ion()
    plot.clf()
    plot.title(title)
    fig, axis_1 = plot.subplots()
    axis_1.set_xlabel('Parameter values')
    if log_scale:
        axis_1.set_xscale('log')
    axis_1.set_ylabel('Fitness', color='darkorange')
    axis_1.plot(parameter_values, fitness, color='darkorange', lw=2)
    axis_1.fill_between(parameter_values, fitness_min, fitness_max, alpha=0.2, color='darkorange', lw=2)

    axis_1.tick_params(axis='y', labelcolor='darkorange')

    axis_2 = axis_1.twinx()
    if log_scale:
        axis_2.set_xscale('log')
    axis_2.set_ylabel('Time (ms)', color='navy')
    axis_2.plot(parameter_values, elapsed_time, color='navy', lw=2)
    axis_2.tick_params(axis='y', labelcolor='navy')

    fig.tight_layout()
    plot.pause(0.001)
    plot.savefig('%s_param_analysis' % title.replace(' ', '_').lower())

