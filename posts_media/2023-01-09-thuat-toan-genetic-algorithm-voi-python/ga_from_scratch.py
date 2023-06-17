import numpy as np
import random
from random import gauss, randrange


def create_individual(num_gens, upper_limit, lower_limit):
    return [round(random.random() * (upper_limit - lower_limit) + lower_limit, 1)
            for x in range(num_gens)]


def population(number_of_individuals, number_of_genes, upper_limit, lower_limit):
    return [create_individual(number_of_genes, upper_limit, lower_limit)
            for x in range(number_of_individuals)]


def fitness_calculation(individual):
    return sum(individual)


def roulette(cum_sum, chance):
    variable = list(cum_sum.copy())
    variable.append(chance)
    variable = sorted(variable)

    return int(variable.index(chance))


def individual_selection(generation, method='fittest-half'):
    selected_individuals = {}
    selected_range = int(len(generation['individuals']) // 2)

    generation['normalized_fitness'] = sorted(
        [generation['fitness'][x] / sum(generation['fitness'])
         for x in range(len(generation['fitness']))], reverse=True
    )

    # calculate cumulative sum of normalized fitness array
    generation['cum_sum'] = np.array(generation['normalized_fitness']).cumsum()

    if method == 'roulette-wheel':
        # select half of population
        selected_individuals = []

        for x in range(selected_range):
            selected_individuals.append(
                roulette(generation['cum_sum'], random.random()))

            # check if there are some duplicated individuals
            while len(set(selected_individuals)) != len(selected_individuals):
                selected_individuals[x] = roulette(
                    generation['cum_sum'], random.random())

        selected_individuals = {
            'individuals': [generation['individuals'][selected_individuals[idx]] for idx in range(selected_range)],
            'fitness': [generation['fitness'][selected_individuals[idx]] for idx in range(selected_range)]
        }

    elif method == 'fittest-half':
        selected_individuals = {
            'individuals': [generation['individuals'][idx] for idx in range(selected_range)],
            'fitness': [generation['fitness'][idx] for idx in range(selected_range)]
        }

    elif method == 'random':
        random_inds = random.sample(
            range(len(generation['individuals'])), selected_range)
        selected_individuals = {
            'individuals': [generation['individuals'][idx] for idx in random_inds],
            'fitness': [generation['fitness'][idx] for idx in random_inds]
        }

    return selected_individuals


def pairing(elite, selected_inds, method='weighted-random'):
    individuals = [elite['individuals']] + selected_inds['individuals']
    fitness = [elite['fitness']] + selected_inds['fitness']
    parent = []

    pairing_len = len(individuals) // 2
    if method == 'random':

        for x in range(pairing_len):
            parent.append([
                individuals[random.randint(0, len(individuals) - 1)],
                individuals[random.randint(0, len(individuals) - 1)]
            ])

            while parent[x][0] == parent[x][1]:
                parent[x][1] = individuals[random.randint(0, len(individuals) - 1)]

    elif method == 'weighted-random':
        normalized_fitness = sorted(
            [fitness[x] / sum(fitness) for x in range(pairing_len)], reverse=True
        )
        cum_sum = np.array(normalized_fitness).cumsum()

        for x in range(pairing_len):
            parent.append(
                [individuals[roulette(cum_sum, random.random())],
                 individuals[roulette(cum_sum, random.random())]]
            )
            while parent[x][0] == parent[x][1]:
                parent[x][1] = individuals[roulette(cum_sum, random.random())]

    return parent


def mating(parents, method='single-point'):
    offsprings = []
    if method == 'single-point':
        pivot_point = random.randint(1, len(parents[0]))
        offsprings = [parents[0]
                      [0:pivot_point] + parents[1][pivot_point:], parents[1]
                      [0:pivot_point] + parents[0][pivot_point:]]

    if method == 'multiple-points':
        pivot_point_1 = random.randint(1, len(parents[0] - 1))
        pivot_point_2 = random.randint(1, len(parents[0]))

        while pivot_point_2 < pivot_point_1:
            pivot_point_2 = random.randint(1, len(parents[0]))
        offsprings = \
            [parents[0][0:pivot_point_1] + parents[1][pivot_point_1:pivot_point_2] +
             [parents[0][pivot_point_2:]], [parents[1][0:pivot_point_1] +
                                            parents[0][pivot_point_1:pivot_point_2] +
                                            [parents[1][pivot_point_2:]]]]

    return offsprings


def mutation(individual, upper_limit, lower_limit, muatation_rate=2,
             method='Reset', standard_deviation=0.001):
    gene = [np.random.randint(0, 7)]

    for x in range(muatation_rate - 1):
        gene.append(np.random.randint(0, 7))
        while len(set(gene)) < len(gene):
            gene[x] = np.random.randint(0, 7)
    mutated_individual = individual.copy()

    if method == 'Gauss':
        for x in range(muatation_rate):
            mutated_individual[x] = round(individual[x] + gauss(0, standard_deviation), 1)
    if method == 'Reset':
        for x in range(muatation_rate):
            mutated_individual[x] = round(np.random.random() * (upper_limit - lower_limit) + lower_limit, 1)

    return mutated_individual


def next_generation(gen, upper_limit, lower_limit):
    elit = {}
    next_gen = {}
    elit['individuals'] = gen['individuals'].pop(-1)
    elit['fitness'] = gen['fitness'].pop(-1)

    selected = individual_selection(gen)
    parents = pairing(elit, selected)
    offsprings = [[[mating(parents[x]) for x in range(len(parents))][y][z] for z in range(2)] \
                  for y in range(len(parents))]

    offsprings1 = [offsprings[x][0] for x in range(len(parents))]
    offsprings2 = [offsprings[x][1] for x in range(len(parents))]

    unmutated = selected['individuals'] + offsprings1 + offsprings2
    mutated = [mutation(unmutated[x], upper_limit, lower_limit) for x in range(len(gen['individuals']))]

    unsorted_individuals = mutated + [elit['individuals']]
    unsorted_next_gen = [fitness_calculation(mutated[x]) for x in range(len(mutated))]
    unsorted_fitness = [unsorted_next_gen[x] for x in range(len(gen['fitness']))] + [elit['fitness']]
    sorted_next_gen = sorted([[unsorted_individuals[x], unsorted_fitness[x]] \
                              for x in range(len(unsorted_individuals))], key=lambda x: x[1])

    next_gen['individuals'] = [sorted_next_gen[x][0] for x in range(len(sorted_next_gen))]
    next_gen['fitness'] = [sorted_next_gen[x][1] for x in range(len(sorted_next_gen))]

    gen['individuals'].append(elit['individuals'])
    gen['fitness'].append(elit['fitness'])

    return next_gen


def fitness_similarity_check(max_fitness, num_similarity):
    similarity = 0
    for n in range(1, len(max_fitness) - 1):
        if max_fitness[n - 1] == max_fitness[n]:
            similarity += 1
        else:
            similarity = 0

    if similarity == num_similarity + 1:
        return True

    return False


if __name__ == '__main__':
    result_file = 'ga_result.txt'

    def first_generation(pop):
        fitness = [fitness_calculation(pop[x]) for x in range(len(pop))]
        sorted_fitness = sorted([[pop[x], fitness[x]] for x in range(len(pop))], key=lambda x: x[1])
        population = [sorted_fitness[x][0] for x in range(len(sorted_fitness))]
        fitness = [sorted_fitness[x][1] for x in range(len(sorted_fitness))]

        return {'individuals': population, 'fitness': fitness}


    pop = population(number_of_individuals=5, number_of_genes=10, upper_limit=1, lower_limit=0)
    gen = [first_generation(pop)]
    fitness_avg = np.array([sum(gen[0]['fitness']) / len(gen[0]['fitness'])])
    fitness_max = np.array([max(gen[0]['fitness'])])

    res = open(result_file, 'a')
    res.write('\n' + str(gen) + '\n')
    res.close()

    finish = False
    while not finish:
        if max(fitness_max) > 4.5 or max(fitness_avg) > 4 or fitness_similarity_check(fitness_max, 50):
            break

        gen.append(next_generation(gen[-1], 1, 0))
        fitness_avg = np.append(fitness_avg, sum(gen[-1]['fitness']) / len(gen[-1]['fitness']))
        fitness_max = np.append(fitness_max, max(gen[-1]['fitness']))
        res = open(result_file, 'a')
        res.write('\n' + str(gen[-1]) + '\n')
        res.close()
