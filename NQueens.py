# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
from collections import OrderedDict

class Population:

    # Initlize GA algorithm params
    def __init__(self, population_size=100, board_size=32, evals=10**5):

        self.pm          = 0.2 #2 / board_size
        self.pc          = 0.47 #0.37
        self.pop_size    = population_size
        self.max_evals   = evals
        self.board_size  = board_size
        self.generations = 0
        self.imp_cntr    = 0

        # GUI #
        self.chess = np.zeros((N, N))
        self.chess[0::2, 1::2] = 1
        self.chess[1::2, 0::2] = 1
    
    # Fitness function check the diagonals of given permutation.
    # it counts every two collide queens, the max number permutation can accumulate is N - 1
    # assume N is the board size
    # in this implementation we want to find the minimum collide queens.
    def fitness(self, perm):
        t1 = 0
        t2 = 0
        f1 = []
        f2 = []
        n  = len(perm)
        for i in range(n): 
            f1.append(perm[i] + i)
            f2.append((n-i+perm[i]))
        
        f1 = sorted(f1)
        f2 = sorted(f2)

        for i in range(1, n) :
            if f1[i] == f1[i-1]:
                t1 += 1
            if f2[i] == f2[i-1]:
                t2 += 1
        fit = t1 + t2

        return fit


    # Order 2 crossover functoin, by given two permutation(parents), the function randomize interval from
    # one parent and build a new permutation with the randomize interval permutation and the other parent 
    # elements in their order.
    def crossover(self, perm1, perm2):
        N = len(perm1)
        h = np.random.randint(1, N)
        l = np.random.randint(0, h)
        
        p1 = perm1[l:h]
        p2 = []
        for i in perm2:
            if i not in p1:
                p2.append(i)
        newb = N - len(perm1[l:])
        
        if newb :
            child1 = np.concatenate((p2[:newb], p1[:], p2[newb:]), axis=0)
        else :
            child1 = np.concatenate((p1[:], p2[newb:]), axis=0)
        p2 = []
        p1 = perm2[l:h]
        for i in perm1:
            if i not in p1:
                p2.append(i)
        newb = N - len(perm2[l:])
        if newb :
            child2 = np.concatenate((p2[:newb], p1[:], p2[newb:]), axis=0)
        else :
            child2 = np.concatenate((p1[:], p2[newb:]), axis=0)
    
        return child1, child2


    # Radomize two diffrent indexes and swap their value in perm ( given permutation )
    def mutation(self, perm):
        N = len(perm)
        rnd = np.random.uniform(0,1)
        if rnd < self.pm:
            # mutation methodes, uncomment desire:

            # ----------swap mutation:-----------
            h = np.random.randint(1, N - 1)
            l = np.random.randint(0, h)       
            result = np.copy(perm)
            result[l], result[h] = result[h], result[l]
            return result

            # -----Reverse section mutation:-----
            # result = list(perm)
            # h = np.random.randint(1, N - 1)
            # l = np.random.randint(0, h)
            # result[l : h] = reversed(result[l : h])
            # return result

            # ----------shufle mutation:---------
            # result = np.copy(perm)
            # np.random.shuffle(result)
            # return result

        return perm

    # Select randomize number from 0 to the first quarter of the population number
    # this function must be called after population_list is sorted by the fitness function
    # by that, the chosen individual is one of the strongest ammong it's enviroment.
    def select(self, population_list):
        index = np.random.randint(0, int(self.pop_size*0.2))
        return population_list[index]


    # display the chess board at the end of the algorithm 
    def display(self,perm):
        for i in range(self.board_size) :
            for j in range(self.board_size) :
                if perm[j] == self.board_size - i - 1 :
                    print("Q",end=" ")
                    self.chess[i,j] = 2
                else :
                    print("-",end=" ")
            print()
        plt.matshow(self.chess)


    # GA Algorithm implementation for the N Queens problem
    def run(self, seed=None):
        fmax = sys.maxsize
        xmax = []
        history = []
        population_list = []
        fitness_list = []
        local_state = np.random.RandomState(seed)
        for i in range(self.pop_size):
            population_list.append(np.random.permutation(self.board_size))
            fitness_list.append(self.fitness(population_list[i]))
        fcurr_best = fmax = np.min(fitness_list)
        eval_cntr = self.pop_size

        history.append(fmax)
        xmax = population_list[min(fitness_list)]

        while (eval_cntr < self.max_evals):

            newPopulationList = []
            # Crossover or mutation activation on half of the population
            for i in range(1, int(self.pop_size / 2)):
                parent1 = self.select(population_list)
                parent2 = self.select(population_list)
                if local_state.uniform(0, 1) < self.pc:
                    xChild1, xChild2 = self.crossover(parent1, parent2)
                else:
                    xChild1 = np.copy(parent1)
                    xChild2 = np.copy(parent2)

                xChild1 = self.mutation(xChild1)
                xChild2 = self.mutation(xChild2)
                newPopulationList.append(xChild1)
                newPopulationList.append(xChild2)

            # Elitist is a merge list of all the parents and their children.
            elitist = np.concatenate((newPopulationList, population_list), axis=0)
            # The whole big population is sorted before killing the 'weak' half of the population.
            elitist = sorted(elitist, key=self.fitness)
            # Calculate number of different parents for next generation.
            elitset = np.copy(elitist)
            dic = OrderedDict()
            for x in elitset:
                key = (x[0],x[-1])
                if key not in dic:
                    dic[key] = x[1:-1]
                else:
                    val = dic[key]
                    dic[key] = [a+b for a,b in zip(val,x[1:-1])]
            elitset = [[k[0]] + v + [k[1]] for k,v in dic.items()]
            # Cut the population back to the original param-pop_size, killing the 'weak' population.
            population_list = elitist[:self.pop_size]
            # Generations counter.
            self.generations += 1
            # Best permutation in the current interation.                
            fcurr_best = self.fitness(population_list[0])
            # Count how many calls to fitness function has been made. 
            eval_cntr += 2*self.pop_size + 1
            # Check wheter the best new permutation is better then current best. 
            if fcurr_best <= fmax:
                fmax = fcurr_best
                xmax = population_list[0]
                self.imp_cntr += 1
            # Append best new permutation to history.
            history.append(fcurr_best)
            # Success rate is the precentage of improvments, optimal is 1/5
            success_rate = "{:.5f}".format((self.imp_cntr / eval_cntr) * 100)
            # Pring algorithm process.
            print("Generation: " + str(self.generations) + " || Fitness: " + str(fmax) + " || Parents variety : " + str(len(elitset)))
            # If the optimum has found befor the algorithm iteration limit.
            if fmax == 0:
                break
            # Uncomment next line to print permutetion progress.
            #print(str(xmax) + " fitness: " + str(fmax))
        return xmax, fmax, history, eval_cntr, success_rate, self.generations


if __name__ == "__main__":
 
    N = int(sys.argv[1])
    GA_N_Queen = Population(100, N, 10**6)
    random_data = os.urandom(1)
    randomSeed = int.from_bytes(random_data, byteorder="big")
    start = time.time()
    xMax, fMax, hist, evals, success, generations = GA_N_Queen.run(randomSeed)
    end = time.time()
    GA_N_Queen.display(xMax)
    print('Solved in ' + str(evals) + ' iterations' +
          '\nGenerations: ' + str(generations) +
          '\nImprovement rate: ' + str(success) + ' %' +
          '\nBoard size = ' + str(N) + 'X' + str(N) +
          '\nFitness = ' + str(fMax) + 
          '\nThe permutation : ' + str(xMax) +
          '\nSeed loaded: ' + str(randomSeed) +
          '\nTime elapsed: ' + str((end - start)) + 'sec')
    plt.show()