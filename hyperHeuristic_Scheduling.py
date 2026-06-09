import random
from hyperHeuristic_Rules import Task, Processor, Scheduler, SEQUENCING_RULES, SCHEDULING_RULES

# ------------------------------------------------------------------ #
# Problem setup                                                        #
# ------------------------------------------------------------------ #
random.seed(42)
tasks = [Task(task_id=i, workload=random.randint(50, 300)) for i in range(1, 21)]
processors = [
    Processor(proc_id=1, speed=10, power_usage=50),
    Processor(proc_id=2, speed=20, power_usage=70),
    Processor(proc_id=3, speed=15, power_usage=60),
    Processor(proc_id=4, speed=25, power_usage=80),
]
scheduler = Scheduler(tasks, processors)

# ------------------------------------------------------------------ #
# GA parameters                                                        #
# ------------------------------------------------------------------ #
# A chromosome is a list of (sequencing_rule, scheduling_rule) pairs,
# one pair per phase. Tasks are split into NUM_PHASES equal subsets,
# and each phase applies its own rule pair to its subset.
#
# Example chromosome (NUM_PHASES=2):
#   [('short_job_first', 'min_min'), ('long_job_first', 'max_min')]
NUM_PHASES = 2
POPULATION_SIZE = 10
MUTATION_RATE = 0.2
NUM_GENERATIONS = 20


# ------------------------------------------------------------------ #
# GA operators                                                         #
# ------------------------------------------------------------------ #
def evaluate(chromosome):
    """Apply the chromosome to a fresh processor state; return makespan."""
    for p in processors:
        p.reset()
    for t in tasks:
        t.assigned_processor = None

    phase_size = len(tasks) // NUM_PHASES
    for i, (seq_rule, sched_rule) in enumerate(chromosome):
        start = i * phase_size
        # Last phase takes any remainder tasks
        end = start + phase_size if i < NUM_PHASES - 1 else len(tasks)
        subset = tasks[start:end]
        ordered = scheduler.sequencing_tasks(seq_rule, subset)
        scheduler.schedule_tasks(sched_rule, ordered)

    return max(p.get_total_workload() for p in processors)


def random_chromosome():
    return [
        (random.choice(SEQUENCING_RULES), random.choice(SCHEDULING_RULES))
        for _ in range(NUM_PHASES)
    ]


def select(population, fitnesses, n):
    """Return the n chromosomes with the lowest makespan."""
    ranked = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
    return [population[i] for i in ranked[:n]]


def crossover(parent1, parent2):
    """Single-point crossover at a random phase boundary."""
    point = random.randint(1, NUM_PHASES - 1) if NUM_PHASES > 1 else 1
    return parent1[:point] + parent2[point:]


def mutate(chromosome):
    """Independently randomise each rule in each phase with probability MUTATION_RATE."""
    return [
        (
            random.choice(SEQUENCING_RULES) if random.random() < MUTATION_RATE else seq,
            random.choice(SCHEDULING_RULES) if random.random() < MUTATION_RATE else sched,
        )
        for seq, sched in chromosome
    ]


# ------------------------------------------------------------------ #
# Main GA loop                                                         #
# ------------------------------------------------------------------ #
def genetic_algorithm():
    population = [random_chromosome() for _ in range(POPULATION_SIZE)]
    best_ever, best_ever_fitness = None, float("inf")

    for generation in range(NUM_GENERATIONS):
        fitnesses = [evaluate(chromo) for chromo in population]

        # Track global best (elitism)
        gen_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] < best_ever_fitness:
            best_ever_fitness = fitnesses[gen_best_idx]
            best_ever = population[gen_best_idx]

        print(f"\n--- Generation {generation + 1} ---")
        for i, (chromo, fit) in enumerate(zip(population, fitnesses)):
            marker = " *" if chromo == best_ever else ""
            print(f"  [{i+1:2d}] {chromo} -> Makespan: {fit:.4f}{marker}")
        print(f"  Best so far: Makespan {best_ever_fitness:.4f}")

        survivors = select(population, fitnesses, POPULATION_SIZE // 2)

        # Carry the global best unchanged into the next generation (elitism),
        # then fill the rest via crossover + mutation.
        next_gen = [best_ever]
        while len(next_gen) < POPULATION_SIZE:
            p1, p2 = random.sample(survivors, 2)
            next_gen.append(mutate(crossover(p1, p2)))

        population = next_gen

    return best_ever, best_ever_fitness


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    best_chromosome, best_makespan = genetic_algorithm()

    print(f"\n=== Best chromosome: {best_chromosome} ===")
    print(f"=== Makespan: {best_makespan:.4f} ===")

    # Re-apply the best chromosome and show the final task assignment
    evaluate(best_chromosome)

    print("\n--- Final Task Assignment ---")
    for p in processors:
        task_ids = [t.task_id for t in p.tasks]
        print(
            f"  Processor {p.proc_id} (speed={p.speed:2d}): "
            f"tasks={task_ids}, workload={p.get_total_workload():.4f}s"
        )
    print(f"Makespan: {best_makespan:.4f}s")
