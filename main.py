import mlrose
import numpy
import time
import neural_net_optimizer_digits
import paramater_analysis
import matplotlib.pyplot as plot


def run_complexity_analysis_four_peaks(random_seed=4664779):
    def create_problem(size):
        initial_state = numpy.random.randint(2, size=size)
        problem = mlrose.DiscreteOpt(size, mlrose.FourPeaks())
        return initial_state, problem
    
    def run_random_hill_climbing(init_state_problem):
        initial_state, problem = init_state_problem
        return mlrose.random_hill_climb(problem, init_state=initial_state, max_attempts=2*problem.length, restarts=5, curve=True)

    def run_simulated_annealing(init_state_problem):
        initial_state, problem = init_state_problem
        return mlrose.simulated_annealing(problem, init_state=initial_state, max_attempts=problem.length, curve=True)

    def run_genetic_algorithm(init_state_problem):
        _, problem = init_state_problem
        return mlrose.genetic_alg(problem, pop_size=4*problem.length, mutation_prob=0.1, curve=True)

    def run_mimic(init_state_problem):
        _, problem = init_state_problem
        return mlrose.mimic(problem, keep_pct=0.1, pop_size=7*problem.length, curve=True)

    sizes = [3, 5, 10, 15, 20, 25, 30, 40, 60, 80]
    local_optima = sizes
    global_optima = []
    for size in sizes:
        T = numpy.ceil(size * 0.1)
        global_optima.append(2 * size - T - 1)
    run_complexity_anaysis_and_plot('four_peaks', create_problem, random_seed, run_random_hill_climbing, run_simulated_annealing, run_genetic_algorithm, run_mimic, sizes, global_optima, local_optima=local_optima)

def run_complexity_analysis_alternating_bits(random_seed=4664779):
    def create_problem(size):
        initial_state = numpy.random.randint(2, size=size)
        problem = mlrose.DiscreteOpt(size, mlrose.FlipFlop())
        return initial_state, problem
    
    def run_random_hill_climbing(init_state_problem):
        initial_state, problem = init_state_problem
        return mlrose.random_hill_climb(problem, init_state=initial_state, max_attempts=problem.length, restarts=5, curve=True)

    def run_simulated_annealing(init_state_problem):
        initial_state, problem = init_state_problem
        return mlrose.simulated_annealing(problem, init_state=initial_state, max_attempts=int(0.5*problem.length), curve=True)

    def run_genetic_algorithm(init_state_problem):
        _, problem = init_state_problem
        return mlrose.genetic_alg(problem, pop_size=4*problem.length, mutation_prob=0.05, curve=True)

    def run_mimic(init_state_problem):
        _, problem = init_state_problem
        return mlrose.mimic(problem, keep_pct=0.2, pop_size=10*problem.length, curve=True)

    sizes = [3, 5, 10, 15, 20, 25, 30, 40, 60, 80]
    global_optima = []
    for size in sizes:
        global_optima.append(size - 1)
    run_complexity_anaysis_and_plot('alternating_bits', create_problem, random_seed, run_random_hill_climbing, run_simulated_annealing, run_genetic_algorithm, run_mimic, sizes, global_optima)

def run_complexity_analysis_counting_ones(random_seed=4664779):
    def create_problem(size):
        initial_state = numpy.random.randint(2, size=size)
        problem = mlrose.DiscreteOpt(size, mlrose.OneMax())
        return initial_state, problem
    
    def run_random_hill_climbing(init_state_problem):
        initial_state, problem = init_state_problem
        return mlrose.random_hill_climb(problem, init_state=initial_state, max_attempts=problem.length, restarts=1, curve=True)

    def run_simulated_annealing(init_state_problem):
        initial_state, problem = init_state_problem
        return mlrose.simulated_annealing(problem, init_state=initial_state, max_attempts=problem.length, curve=True)

    def run_genetic_algorithm(init_state_problem):
        _, problem = init_state_problem
        return mlrose.genetic_alg(problem, mutation_prob=0.05, curve=True)

    def run_mimic(init_state_problem):
        _, problem = init_state_problem
        return mlrose.mimic(problem, keep_pct=0.2, pop_size=5*problem.length, curve=True)

    sizes = [3, 5, 10, 15, 20, 25, 30, 40, 60, 80]
    global_optima = []
    for size in sizes:
        global_optima.append(size - 1)
    run_complexity_anaysis_and_plot('counting_ones', create_problem, random_seed, run_random_hill_climbing, run_simulated_annealing, run_genetic_algorithm, run_mimic, sizes, global_optima)

def run_complexity_anaysis_and_plot(problem_name, create_problem, random_seed, run_random_hill_climbing, run_simulated_annealing, run_genetic_algorithm, run_mimic, sizes, global_optima, local_optima=None):
    rhc_fitness, rhc_elapsed_time, rhc_iterations = run_complexity_analysis(create_problem, sizes, run_random_hill_climbing, random_seed)
    print('Fitness: {}'.format(rhc_fitness))
    print('Time: {}'.format(array_to_string(rhc_elapsed_time, '{:.0f} ms')))
    print('Iterations: {}'.format(rhc_iterations))
    sa_fitness, sa_elapsed_time, sa_iterations = run_complexity_analysis(create_problem, sizes, run_simulated_annealing, random_seed)
    print('Fitness: {}'.format(sa_fitness))
    print('Time: {}'.format(array_to_string(sa_elapsed_time, '{:.0f} ms')))
    print('Iterations: {}'.format(sa_iterations))
    ga_fitness, ga_elapsed_time, ga_iterations = run_complexity_analysis(create_problem, sizes, run_genetic_algorithm, random_seed)
    print('Fitness: {}'.format(ga_fitness))
    print('Time: {}'.format(array_to_string(ga_elapsed_time, '{:.0f} ms')))
    print('Iterations: {}'.format(ga_iterations))
    mimic_fitness, mimic_elapsed_time, mimic_iterations = run_complexity_analysis(create_problem, sizes, run_mimic, random_seed)
    print('Fitness: {}'.format(mimic_fitness))
    print('Time: {}'.format(array_to_string(mimic_elapsed_time, '{:.0f} ms')))
    print('Iterations: {}'.format(mimic_iterations))
    plot.ion()
    plot.clf()
    plot.title('Fitness')
    plot.ylabel('fitness')
    plot.xlabel('size (# of bits)')
    plot.plot(sizes, rhc_fitness, marker='.', color='deepskyblue', label='RHC')
    plot.plot(sizes, sa_fitness, marker='*', color='lightskyblue', label='SA')
    plot.plot(sizes, ga_fitness, marker='+', color='navy', label='GA')
    plot.plot(sizes, mimic_fitness, marker='x', color='royalblue', label='MIMIC')
    if local_optima is not None:
        plot.plot(sizes, local_optima, color='red', label='local optima')
    plot.plot(sizes, global_optima, color='green', label='global optima')
    plot.legend(loc='best')
    plot.pause(0.001)
    plot.savefig('%s_fitness' % problem_name)

    plot.clf()
    plot.title('Time')
    plot.ylabel('time (ms)')
    plot.xlabel('size (# of bits)')
    plot.plot(sizes, rhc_elapsed_time, marker='.', color='deepskyblue', label='RHC')
    plot.plot(sizes, sa_elapsed_time, marker='*', color='lightskyblue', label='SA')
    plot.plot(sizes, ga_elapsed_time, marker='+', color='navy', label='GA')
    plot.plot(sizes, mimic_elapsed_time, marker='x', color='royalblue', label='MIMIC')
    plot.legend(loc='best')
    plot.pause(0.001)
    plot.savefig('%s_time' % problem_name)

    plot.clf()
    plot.title('Iterations')
    plot.ylabel('iterations')
    plot.xlabel('size (# of bits)')
    plot.plot(sizes, rhc_iterations, marker='.', color='deepskyblue', label='RHC')
    plot.plot(sizes, sa_iterations, marker='*', color='lightskyblue', label='SA')
    plot.plot(sizes, ga_iterations, marker='+', color='navy', label='GA')
    plot.plot(sizes, mimic_iterations, marker='x', color='royalblue', label='MIMIC')
    plot.legend(loc='best')
    plot.pause(0.001)
    plot.savefig('%s_iterations' % problem_name)

def run_complexity_analysis(create_problem, sizes, optimizer, random_seed):
    fitness = []
    elapsed_time = []
    iterations = []
    for size in sizes:
        numpy.random.seed(random_seed)
        total_time = 0
        size_fitness = 0
        size_iterations = 0
        print('Size: %d' % size)
        for i in range(0, 10):
            problem = create_problem(size)
            start = time.time()
            _, best_fitness, fitness_curve = optimizer(problem)
            total_time += time.time() - start
            size_fitness += best_fitness
            size_iterations += len(fitness_curve)
        fitness.append(size_fitness / 10.0)
        elapsed_time.append(total_time * 100)       # average (total / 10) in ms, so * 1000
        iterations.append(size_iterations / 10.0)
    return fitness, elapsed_time, iterations

def run_parameter_analysis_four_peaks(size=20, random_seed=4664779):
    def create_problem():
        initial_state = numpy.random.randint(2, size=size)
        problem = mlrose.DiscreteOpt(size, mlrose.FourPeaks())
        return initial_state, problem
    run_random_hill_climbing_parameter_analysis(create_problem, 'Four Peaks', random_seed)
    run_simulated_annealing_parameter_analysis(create_problem, 'Four Peaks', random_seed)
    run_genetic_algorithm_parameter_analysis(create_problem, 'Four Peaks', random_seed)
    run_mimic_parameter_analysis(create_problem, 'Four Peaks', random_seed)

def run_parameter_analysis_alternating_bits(size=20, random_seed=4664779):
    def create_problem():
        initial_state = numpy.random.randint(2, size=size)
        problem = mlrose.DiscreteOpt(size, mlrose.FlipFlop())
        return initial_state, problem
    name = 'Alternating Bits'
    run_random_hill_climbing_parameter_analysis(create_problem, name, random_seed)
    run_simulated_annealing_parameter_analysis(create_problem, name, random_seed)
    run_genetic_algorithm_parameter_analysis(create_problem, name, random_seed)
    run_mimic_parameter_analysis(create_problem, name, random_seed)

def run_parameter_analysis_one_max(size=20, random_seed=4664779):
    def create_problem():
        initial_state = numpy.random.randint(2, size=size)
        problem = mlrose.DiscreteOpt(size, mlrose.OneMax())
        return initial_state, problem
    name = 'Counting Ones'
    run_random_hill_climbing_parameter_analysis(create_problem, name, random_seed)
    run_simulated_annealing_parameter_analysis(create_problem, name, random_seed)
    run_genetic_algorithm_parameter_analysis(create_problem, name, random_seed)
    run_mimic_parameter_analysis(create_problem, name, random_seed)

def run_random_hill_climbing_parameter_analysis(create_problem, problem_title, random_seed):
    def random_hill_climb_max_attempts(problem, initial_state, attempts_percentage):
        attempts = int(problem.length * attempts_percentage)
        return mlrose.random_hill_climb(problem, init_state=initial_state, curve=False, max_attempts=attempts)

    paramater_analysis.run_parameter_analysis(create_problem,
                                              random_hill_climb_max_attempts,
                                              numpy.linspace(0.2, 2.0, 9),
                                              random_seed,
                                              '%s RHC Max Attempts' % problem_title)
    
    def random_hill_climb_restarts(problem, initial_state, restarts):
        return mlrose.random_hill_climb(problem, init_state=initial_state, curve=False, restarts=int(restarts))

    paramater_analysis.run_parameter_analysis(create_problem, 
                                              random_hill_climb_restarts,
                                              numpy.arange(0, 11),
                                              random_seed,
                                              '%s RHC Restarts' % problem_title)

def run_simulated_annealing_parameter_analysis(create_problem, problem_title, random_seed):
    def simulated_annealing_max_attempts(problem, initial_state, attempts_percentage):
        attempts = int(problem.length * attempts_percentage)
        return mlrose.simulated_annealing(problem, init_state=initial_state, curve=False, max_attempts=attempts)

    paramater_analysis.run_parameter_analysis(create_problem,
                                              simulated_annealing_max_attempts,
                                              numpy.linspace(0.2, 2.0, 9),
                                              random_seed,
                                              '%s SA Max Attempts' % problem_title)
    
    def simulated_annealing_decay(problem, initial_state, decay):
        if decay == 'Geom':
            schedule = mlrose.GeomDecay()
        if decay == 'Exp':
            schedule = mlrose.ExpDecay()
        if decay == 'Arith':
            schedule = mlrose.ArithDecay()
        return mlrose.simulated_annealing(problem, init_state=initial_state, curve=False, schedule=schedule)

    paramater_analysis.run_parameter_analysis(create_problem, 
                                              simulated_annealing_decay,
                                              ['Geom', 'Exp', 'Arith'],
                                              random_seed,
                                              '%s SA Decay Schedule' % problem_title)

def run_genetic_algorithm_parameter_analysis(create_problem, problem_title, random_seed):
    def genetic_algorithm_pop_size(problem, initial_state, pop_multiplier):
        pop_size = problem.length * pop_multiplier
        print('GA w/ pop size %d' % pop_size)
        return mlrose.genetic_alg(problem, curve=False, pop_size=int(pop_size))

    paramater_analysis.run_parameter_analysis(create_problem,
                                              genetic_algorithm_pop_size,
                                              numpy.arange(1, 10),
                                              random_seed,
                                              '%s GA Population Size' % problem_title)
    
    def genetic_algorithm_mutation_prob(problem, initial_state, mutation_prob):
        print('GA w/ mutation prob %f' % mutation_prob)
        return mlrose.genetic_alg(problem, curve=False, mutation_prob=mutation_prob)

    paramater_analysis.run_parameter_analysis(create_problem, 
                                              genetic_algorithm_mutation_prob,
                                              numpy.logspace(-3, -0.7, 10),
                                              random_seed,
                                              '%s GA Mutation Prob' % problem_title,
                                              log_scale=True)

def run_mimic_parameter_analysis(create_problem, problem_title, random_seed):
    def mimic_pop_size(problem, initial_state, pop_multiplier):
        pop_size = problem.length * pop_multiplier
        print('MIMIC w/ pop size %d' % pop_size)
        return mlrose.mimic(problem, curve=False, pop_size=int(pop_size))

    paramater_analysis.run_parameter_analysis(create_problem,
                                              mimic_pop_size,
                                              numpy.arange(1, 10),
                                              random_seed,
                                              '%s MIMIC Population Size' % problem_title)
    
    def mimic_keep_pct(problem, initial_state, keep_pct):
        print('MIMIC w/ keep percentage %f' % keep_pct)
        return mlrose.mimic(problem, curve=False, keep_pct=keep_pct)

    paramater_analysis.run_parameter_analysis(create_problem, 
                                              mimic_keep_pct,
                                              numpy.logspace(-2, -0.4, 10),
                                              random_seed,
                                              '%s MIMIC Keep Percentage' % problem_title,
                                              log_scale=True)


def array_to_string(array, formatting):
    formatted_string = '['
    for i, element in enumerate(array):
        formatted_string += formatting.format(element)
        if i + 1 < len(array):
            formatted_string += ', '
    formatted_string += ']'
    return formatted_string

def main():
    run_parameter_analysis_four_peaks()
    run_parameter_analysis_one_max()
    run_parameter_analysis_alternating_bits()

    # after running the above, I ran the following with parameters gleaned from the first three functions
    run_complexity_analysis_four_peaks()
    run_complexity_analysis_counting_ones()
    run_complexity_analysis_alternating_bits()

    nn = neural_net_optimizer_digits.NeuralNetOptimizerDigits()
    nn.find_parameters_simulated_annealing()
    nn.find_parameters_random_hill_climbing()
    nn.find_parameters_genetic_algorithm()

    nn.gradient_descent()
    nn.optimize_random_hill_climbing()
    nn.optimize_simulated_annealing()
    nn.optimize_genetic_algoirthm()


if __name__ == "__main__":
    main()
