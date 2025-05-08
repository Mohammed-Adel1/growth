import random
import math
import numpy as np
from swap import SwapMove, swap_predict, swap_extended
from collections import namedtuple
import time

class GrowthOptimizerParams:
    def __init__(self,
                 p1=5,
                 p2=0.001,
                 p3=0.3):
        """
        Parameters for the Growth Optimizer algorithm
        
        Args:
            p1: Number of top solutions to consider for learning
            p2: Probability of accepting worse solutions
            p3: Probability of applying reflection to a dimension
        """
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

def select_id(popsize, i, k):
    """
    Generate k random integer values within [0, popsize-1] that don't include i and don't repeat
    
    Args:
        popsize: Population size
        i: Index to exclude
        k: Number of unique indices to select
        
    Returns:
        List of k unique random indices
    """
    if k <= popsize - 1:
        vec = list(range(0, i)) + list(range(i+1, popsize))
        r = []
        for _ in range(k):
            n = len(vec) - 1
            t = random.randint(0, n)
            r.append(vec[t])
            vec.pop(t)
        return r
    return []

def growth_optimizer(state, params, timeout_callback=None, verbose_callback=None):
    """
    Implementation of the Growth Optimizer algorithm
    
    Args:
        state: Solver state containing the model and solutions
        params: GrowthOptimizerParams instance
        timeout_callback: Function to check if time limit has been reached
        verbose_callback: Function to print progress updates
    """
    model = state.model
    popsize = state.config.POPULATION_SIZE
    dimension = state.L * 3  # Each lecture has 3 dimensions: room, day, slot
    max_fes = state.config.MAX_ITERATIONS
    p1 = params.p1
    p2 = params.p2
    p3 = params.p3
    
    # Initialize population of solutions
    population = [state.current_solution]
    fitness = [state.current_cost]
    
    # Create additional solutions for the population
    for _ in range(1, popsize):
        new_solution = state.current_solution.__class__(model)
        new_solution.copy_from(state.current_solution)
        
        # Perturb the solution with random swaps
        for _ in range(10):  # Apply 10 random swaps to create diversity
            mv = state.generate_swap_move()
            swap_extended(new_solution, mv, strategy='if_feasible')
        
        cost = new_solution.compute_total_cost()
        population.append(new_solution)
        fitness.append(cost)
    
    # Track best solution and evaluation count
    fes = popsize
    
    # Main optimization loop
    iter_count = 0
    while fes < max_fes:
        if timeout_callback and timeout_callback():
            break
            
        # Sort population by fitness
        sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])
        best_idx = sorted_indices[0]
        
        # Learning phase
        for i in range(popsize):
            # Select reference solutions
            worst_idx = sorted_indices[random.randint(popsize - p1, popsize - 1)]
            better_idx = sorted_indices[random.randint(1, min(p1, popsize - 1))]
            
            random_indices = select_id(popsize, i, 2)
            l1 = random_indices[0]
            l2 = random_indices[1]
            
            # Generate new solution through learning
            new_solution = population[i].__class__(model)
            new_solution.copy_from(population[i])
            
            # Apply a series of targeted swaps based on the GO algorithm's learning mechanism
            for _ in range(3):  # Apply a few swaps based on learning
                mv = state.generate_swap_move()
                
                # Adjust swap probabilities based on fitness
                sf = fitness[i] / max(fitness)
                if random.random() < sf:
                    swap_extended(new_solution, mv, strategy='if_feasible')
            
            # Evaluate the new solution
            new_fitness = new_solution.compute_total_cost()
            fes += 1
            
            # Update if better or with probability p2
            if new_fitness < fitness[i]:
                fitness[i] = new_fitness
                population[i] = new_solution
            elif random.random() < p2 and i != best_idx:
                fitness[i] = new_fitness
                population[i] = new_solution
            
            # Update global best
            if new_fitness < state.best_cost:
                state.best_cost = new_fitness
                state.best_solution.copy_from(new_solution)
                
                if verbose_callback:
                    verbose_callback(iter_count, 0, new_fitness, new_fitness,
                                     state.best_cost, 0)
            
            if fes >= max_fes:
                break
        
        # Reflection phase
        if fes < max_fes:
            for i in range(popsize):
                new_solution = population[i].__class__(model)
                new_solution.copy_from(population[i])
                
                # Apply reflection with probability p3 to some lectures
                lectures_to_modify = random.sample(range(state.L), max(1, int(state.L * p3)))
                
                for l_idx in lectures_to_modify:
                    # Get a reference solution from the top p1 solutions
                    ref_idx = sorted_indices[random.randint(0, min(p1-1, popsize-1))]
                    ref_solution = population[ref_idx]
                    
                    # Create a move that mimics reflection
                    if population[i].assignments[l_idx] and ref_solution.assignments[l_idx]:
                        # With some probability, move towards a reference solution's assignment
                        if random.random() < 0.5:
                            ref_assignment = ref_solution.assignments[l_idx]
                            mv = SwapMove(l_idx, ref_assignment.r, ref_assignment.d, ref_assignment.s)
                            swap_extended(new_solution, mv, strategy='if_feasible')
                        else:
                            # Otherwise do a random move
                            mv = state.generate_swap_move()
                            swap_extended(new_solution, mv, strategy='if_feasible')
                    
                # Add some randomness with decreasing probability over time
                af = 0.01 + (0.1 - 0.01) * (1 - fes / max_fes)
                if random.random() < af:
                    for _ in range(3):  # Do a few completely random swaps
                        mv = state.generate_swap_move()
                        swap_extended(new_solution, mv, strategy='if_feasible')
                
                # Evaluate the new solution
                new_fitness = new_solution.compute_total_cost()
                fes += 1
                
                # Update if better or with probability p2
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    population[i] = new_solution
                elif random.random() < p2 and i != best_idx:
                    fitness[i] = new_fitness
                    population[i] = new_solution
                
                # Update global best
                if new_fitness < state.best_cost:
                    state.best_cost = new_fitness
                    state.best_solution.copy_from(new_solution)
                    
                    if verbose_callback:
                        verbose_callback(iter_count, 0, new_fitness, new_fitness,
                                       state.best_cost, 0)
                
                if fes >= max_fes:
                    break
        
        # Update current solution with the best in population
        best_idx = fitness.index(min(fitness))
        state.current_solution.copy_from(population[best_idx])
        state.current_cost = fitness[best_idx]
        
        iter_count += 1
        
        # Periodic reporting
        if verbose_callback and iter_count % 10 == 0:
            verbose_callback(iter_count, 0, state.current_cost, min(fitness),
                           state.best_cost, 0)
    
    return state.best_cost