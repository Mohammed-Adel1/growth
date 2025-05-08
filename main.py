# main.py
from model_parser import TimetableModel
from solution import *
from feasible_solution_finder import FeasibleSolutionFinder, FeasibleSolutionFinderConfig
from heuristic_solver_state import HeuristicSolverState
from growth_optimizer import GrowthOptimizerParams, growth_optimizer

import time
from config import *

# Create a CONFIG object from the config variables
CONFIG = type('Config', (object,), {
    'COMP': COMP,
    'NUM_SWARMS': NUM_SWARMS,
    'POPULATION_SIZE': POPULATION_SIZE,
    'MAX_ITERATIONS': MAX_ITERATIONS,
    'LIMIT': LIMIT,
    'R_CLOUD': R_CLOUD,
    'TIME': TIME,
    'GD_ITER': GD_ITER,
    'INPUT': INPUT,
    'OUTPUT': OUTPUT
})


def timeout_callback_factory(seconds):
    start = time.time()
    return lambda: time.time() - start > seconds


def verbose_callback(iteration, idle, current, local_best, global_best, temperature):
    print(f"Iter {iteration} | Idle {idle} | Curr {current} | Local Best {local_best} "
          f"| Global Best {global_best}")


def main():
    # === Load model ===
    model = TimetableModel()
    model.parse(INPUT)
    print("Model loaded.")

    # === Prepare initial feasible solution ===
    finder = FeasibleSolutionFinder()
    config = FeasibleSolutionFinderConfig()
    solution = Solution(model)

    print("Finding initial feasible solution...")
    if not finder.find(config, solution):
        print("Failed to find an initial feasible solution.")
        return

    initial_cost = solution.compute_total_cost()
    print("Initial feasible solution cost:", initial_cost)

    # === Prepare solver state ===
    best_solution = Solution(model)
    best_solution.copy_from(solution)

    state = HeuristicSolverState(model=model,
                                  current_solution=solution,
                                  best_solution=best_solution,
                                  current_cost=initial_cost,
                                  best_cost=initial_cost,
                                  config=CONFIG)
    
    state.methods_name = ["Growth Optimizer"]
    state.method = 0

    # === Configure Growth Optimizer ===
    params = GrowthOptimizerParams(
        p1=5,    # Number of top solutions to consider
        p2=0.001, # Probability of accepting worse solutions
        p3=0.3    # Probability of applying reflection
    )

    # === Run Growth Optimizer ===
    print("Running Growth Optimizer...")
    timeout = timeout_callback_factory(TIME)  # Use TIME from config.py
    growth_optimizer(state, params, timeout_callback=timeout, verbose_callback=verbose_callback)

    print("\nFinal best cost:", state.best_cost)
    print("Final best solution:")
    print(state.best_solution.to_string())

    with open(OUTPUT, "w") as f:
        f.write(state.best_solution.to_string())


if __name__ == "__main__":
    main()