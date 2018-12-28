import numpy as np
import matplotlib.pyplot as plt
import sys


class Population:

    population_list = []
    board_size = 0
    pop_size = 0
    max_evals = 0
    pm = 0
    pc = 0

    ''' Initlize all paremeters for the algorithm '''

    def __init__(self, population_size=1000, board_size=8, evals=10 ** 5):

        self.pm = 4 / population_size
        self.pc = 0.37
        self.pop_size = population_size
        self.max_evals = evals
        self.board_size = board_size
        for i in range(population_size):
            self.population_list.append(np.random.permutation(board_size))

        print(self.fitness([0, 1, 2, 3, 4, 5, 6, 7]))
        print(self.fitness([3, 0, 4, 7, 1, 6, 2, 5]))
        print(self.fitness([0, 7, 2, 5, 4, 1, 3, 6]))

    ''' fitness function get a specific board with queens already spotted in their places
        and calculate it's value by counting how many queens are on the same line.
        the fitness return value can accumulate at most as the number of queens on the given board. 
        with board of size 8 we can get 0 - 28 because 8 choose 2 = 28
    '''

    def fitness(self, phenotype):

        t1 = 0
        t2 = 0
        f1 = []
        f2 = []
        size = len(phenotype)

        for i in range(size):
            f1.append(phenotype[i] - i)
            f2.append(size - (phenotype[i] - i))

        while len(f1) > 0:
            p1 = f1.pop()
            if f1.count(p1) > 0:
                t1 = t1 + f1.count(p1) * 2
                f1 = list(filter(lambda x: x != p1, f1))

        while len(f2) > 0:
            p2 = f2.pop()
            if f2.count(p2) > 0:
                t2 = t2 + f2.count(p2) * 2
                f2 = list(filter(lambda x: x != p2, f2))

        fit = t1 + t2
        return fit

    def decode(self, genom):
        return genom

# Two point cross over implementation
    def crossover(self, perm1, perm2):
        n = len(perm1)
        index1 = np.random.randint(0, n - 1, dtype=int)
        index2 = np.random.randint(0, n - 1, dtype=int)
        if index1 > index2:
            index1, index2 = index2, index1
        child1 = perm1[:index1] + perm2[index1:index2] + perm1[index2:]
        child2 = perm2[:index1] + perm1[index1:index2] + perm2[index2:]
        return child1, child2


    def mutation(self, perm):
        n = len(perm)
        rnd = np.random.randint(0, n - 1, dtype=int)
        if rnd < self.pm:
            rand_idx1 = np.random.randint(0, n - 1, dtype=int)
            rand_idx2 = np.random.randint(0, n - 1, dtype=int)
            while rand_idx2 == rand_idx1:
                rand_idx2 = np.random.randint(0, n - 1, dtype=int)
            result = np.copy(perm)
            result[rand_idx1], result[rand_idx2] = result[rand_idx2], result[rand_idx1]
            return result
        return perm

    ''' select the best 2 individual in the population (by fitness) to be the parents of the next generation
        then select the next best 2 again and again
    '''

    def select(self):
        print()



    ''' display the chess board at the end of the algorithm '''

    def display(self):
        print()

    ''' The GA algorithm get for NQueens problem '''

    def GA(self, n, max_evals, decodefct, selectfct, fitnessfct, seed=None):
        eval_cntr = 0
        history = []
        #
        # GA params
        mu = 1000
        # Probability of Crossover
        pc = 0.37
        # Probability of Mutation
        pm = 4 / n
        #    kXO = 1 # 1-point Xover
        local_state = np.random.RandomState(seed)
        Genome = local_state.randint(2, size=(n, mu))
        Phenotype = []
        for k in range(mu):
            Phenotype.append(decodefct(Genome[:, [k]]))
            fitness = fitnessfct(Phenotype)
        eval_cntr += mu
        fcurr_best = fmax = np.max(fitness)
        xmax = Genome[:, [np.argmax(fitness)]]
        history.append(fmax)

        # while (eval_cntr < max_evals):

        return xmax, fmax, history


# Defualt args are for 8 x 8 board #
population8 = Population()
chess = np.zeros((8, 8))
chess[1::2, 0::2] = 1
chess[0::2, 1::2] = 1
plt.imshow(chess, cmap='binary')
plt.show()
input()
# population16 = Population(20,16,10**5)
# xMax, fMax, hist = population8.GA()
