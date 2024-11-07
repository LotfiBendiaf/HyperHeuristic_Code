import random

# Define Task and Processor classes
class Task:
    def __init__(self, task_id, workload, priority=1, deadline=None):
        self.task_id = task_id
        self.workload = workload  # Computation time in cycles or seconds
        self.priority = priority
        self.deadline = deadline
        self.assigned_processor = None

class Processor:
    def __init__(self, proc_id, speed, power_usage):
        self.proc_id = proc_id
        self.speed = speed  # Instructions per second or computational power
        self.power_usage = power_usage
        self.tasks = []  # Tasks assigned to this processor
        self.total_workload = 0  # Total workload assigned to this processor

    def execute_task(self, task):
        execution_time = task.workload / self.speed
        # print(f"Task {task.task_id} executed on Processor {self.proc_id} in {execution_time:.2f} seconds.")
        self.tasks.append(task)
        self.total_workload += execution_time  # Add the task workload to the processor's total workload

    def get_total_workload(self):
        return self.total_workload  # Return the total workload of the processor

# Define Scheduler class with both Min-Min and Max-Min heuristics
class Scheduler:
    def __init__(self, tasks, processors):
        self.tasks = tasks
        self.processors = processors
    
    def schedule_tasks(self, policy, subset):
        match policy:
            case "first_in_first_out":
                self.fcfs_schedule(subset)
            case "round_robin":
                self.round_robin_schedule(subset)
            case "priority_based":
                self.priority_based_schedule(subset)
            case "min_min":
                self.min_min_schedule(subset)
            case "min_queued_elements":
                self.min_queued_elements(subset)
            case "max_min":
                self.max_min_schedule(subset)
            case "short_job_first":
                self.sjf_schedule(subset)
            case "long_job_first":
                self.ljf_schedule(subset)
            case "load_balancing":
                self.load_balancing_schedule(subset)
            case "greedy_best_fit":
                self.greedy_best_fit_schedule(subset)
            case "weighted_round_robin":
                self.weighted_round_robin_schedule(subset)
            case "threshold_based":
                self.threshold_based_schedule(subset)
            case _:
                raise ValueError(f"Unknown scheduling policy: {policy}")
 
    
    def assign_task_to_processor(self, task, processor):
        task.assigned_processor = processor.proc_id
        processor.execute_task(task)

    # Function to calculate the total workload on a processor
    def select_best_processor(self, task):
        # Select processor with the least total workload
        return min(self.processors, key=lambda p: p.get_total_workload())
    
    # FCFS heuristic (First-Come, First-Served)
    def fcfs_schedule(self, task_subset):
        for task in task_subset:
            best_processor = self.select_best_processor(task)
            self.assign_task_to_processor(task, best_processor)

    # Simple Round-Robin scheduling across processors
    def round_robin_schedule(self, task_subset):
        proc_idx = 0
        for task in task_subset:
            self.assign_task_to_processor(task, self.processors[proc_idx])
            proc_idx = (proc_idx + 1) % len(self.processors)

    # SJF heuristic (Shortest Job First)
    def sjf_schedule(self, task_subset):
        sorted_tasks = sorted(task_subset, key=lambda t: t.workload)
        for task in sorted_tasks:
            best_processor = self.select_best_processor(task)
            self.assign_task_to_processor(task, best_processor)

    # LJF heuristic (Longest Job First)
    def ljf_schedule(self, task_subset):
        sorted_tasks = sorted(task_subset, key=lambda t: t.workload, reverse=True)
        for task in sorted_tasks:
            best_processor = self.select_best_processor(task)
            self.assign_task_to_processor(task, best_processor)

    # Priority-Based Scheduling (High Priority First)
    def priority_based_schedule(self, task_subset):
        sorted_tasks = sorted(task_subset, key=lambda t: t.priority, reverse=True)
        for task in sorted_tasks:
            best_processor = self.select_best_processor(task)
            self.assign_task_to_processor(task, best_processor)

    # Select processor with the least number of tasks or fastest
    def min_queued_elements(self, task_subset):
        for task in task_subset:
            best_processor = min(self.processors, key=lambda p: len(p.tasks))
            self.assign_task_to_processor(task, best_processor)

    # Load Balancing Schedule (Least Total Workload)
    def load_balancing_schedule(self, task_subset):
        for task in task_subset:
            best_processor = min(self.processors, key=lambda p: sum(t.workload for t in p.tasks))
            self.assign_task_to_processor(task, best_processor)

    # Weighted Round-Robin scheduling across processors based on speed
    def weighted_round_robin_schedule(self, task_subset):
        total_speed = sum(p.speed for p in self.processors)
        weights = [p.speed / total_speed for p in self.processors]
        proc_idx = 0
        for task in task_subset:
            self.assign_task_to_processor(task, self.processors[proc_idx])
            proc_idx = (proc_idx + int(1 / weights[proc_idx])) % len(self.processors)

    # Greedy Best-Fit heuristic (chooses processor that finishes the task the quickest)
    def greedy_best_fit_schedule(self, task_subset):
        for task in task_subset:
            best_processor = min(self.processors, key=lambda p: task.workload / p.speed)
            self.assign_task_to_processor(task, best_processor)

    # Threshold based heuristic
    def threshold_based_schedule(self, task_subset, threshold=200):
        for task in task_subset:
            if task.workload > threshold:
                # Assign to the highest-speed processor
                best_processor = max(self.processors, key=lambda p: p.speed)
            else:
                # Assign to the least-loaded processor (by number of tasks)
                best_processor = min(self.processors, key=lambda p: len(p.tasks))
            
            self.assign_task_to_processor(task, best_processor)
        
    # Min-Min heuristic
    def min_min_schedule(self, task_subset):
        while task_subset:
            min_task = None
            min_time = float('inf')
            best_processor = None

            # For each task, find the processor that would give the maximum completion time
            for task in task_subset:
                for processor in self.processors:
                    # Calculate the completion time if the task is assigned to the processor
                    time_to_complete = task.workload / processor.speed
                    
                    # Find the maximum time considering the current load on the processor
                    current_total_time = processor.get_total_workload() + time_to_complete
                    
                    if current_total_time < min_time:
                        min_time = current_total_time
                        min_task = task
                        best_processor = processor
            
            # Assign the task with the minimum completion time to the selected processor
            self.assign_task_to_processor(min_task, best_processor)
            task_subset.remove(min_task)

    # Max-Min heuristic
    def max_min_schedule(self, task_subset):
        while task_subset:
            tasks_min = []       # Store the minimum times for each task
            processors_min = []  # Store the corresponding processors for those minimum times

            # For each task, find the processor with the minimum completion time
            for task in task_subset:
                min_time = float('inf')  # Reset minimum time for each task
                best_processor = None
                
                for processor in self.processors:
                    # Calculate the completion time if the task is assigned to this processor
                    time_to_complete = task.workload / processor.speed
                    current_total_time = processor.get_total_workload() + time_to_complete
                    
                    # Find the processor with the minimum time for this task
                    if current_total_time < min_time:
                        min_time = current_total_time
                        best_processor = processor
                
                # Store the task's minimum time and the corresponding processor
                tasks_min.append((task, min_time))
                processors_min.append(best_processor)

            # Now, find the task with the maximum of the minimum completion times
            max_task, max_time = max(tasks_min, key=lambda x: x[1])  # Get the task with the max of minimum times
            best_processor = processors_min[tasks_min.index((max_task, max_time))]  # Get its best processor

            # Assign the selected task to the best processor
            self.assign_task_to_processor(max_task, best_processor)
            
            # Remove the task from the task subset
            task_subset.remove(max_task)

# Create 20 tasks with random workloads
tasks = [Task(task_id=i, workload=random.randint(50, 300)) for i in range(1, 51)]


# Create 4 processors with varying speeds
processors = [
    Processor(proc_id=1, speed=10, power_usage=50),
    Processor(proc_id=2, speed=20, power_usage=70),
    Processor(proc_id=3, speed=15, power_usage=60),
    Processor(proc_id=4, speed=25, power_usage=80)
]
    
# Create 20 tasks with random workloads
tasks = [Task(task_id=i, workload=random.randint(50, 300)) for i in range(1, 21)]

# Create 4 processors with varying speeds
processors = [
    Processor(proc_id=1, speed=10, power_usage=50),
    Processor(proc_id=2, speed=20, power_usage=70),
    Processor(proc_id=3, speed=15, power_usage=60),
    Processor(proc_id=4, speed=25, power_usage=80)
]

# Initialize the scheduler
scheduler = Scheduler(tasks, processors)

# Divide tasks: first 10 for Min-Min, remaining 10 for Max-Min
first_half_tasks = tasks[:10]
second_half_tasks = tasks[10:]

# Example sequence of heuristics
available_heuristics = ['first_in_first_out', 'min_min', 'max_min', 'min_queued_elements', 
                        'round_robin', 'short_job_first', 'long_job_first', 'priority_based', 'load_balancing']

# Genetic Algorithm Parameters
population_size = 10  # Number of sequences in population
mutation_rate = 0.2   # Chance of mutation
num_generations = 20  # Number of generations to evolve
sequence_length = 4   # Length of each heuristic sequence


# A function that applies a sequence of scheduling heuristics and returns the makespan
def evaluate_sequence(scheduler, sequence, tasks, processors):
    task_subset = tasks[:]  # Make a copy of tasks
    for heuristic in sequence:
        scheduler.schedule_tasks(policy=heuristic, subset=task_subset)
    makespan = max(processor.get_total_workload() for processor in processors)
    return makespan  # Calculate the makespan


# Create an initial population of random sequences
def initialize_population():
    population = []
    for _ in range(population_size):
        sequence = random.sample(available_heuristics, sequence_length)
        population.append(sequence)
    return population


# Selection: Choose the top N sequences based on fitness (lowest makespan)
def select(population, fitnesses, num_selected):
    selected_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:num_selected]
    return [population[i] for i in selected_indices]


# Crossover: Combine two sequences to create a new sequence
def crossover(parent1, parent2):
    crossover_point = random.randint(1, sequence_length - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


# Mutation: Randomly swap two heuristics in a sequence or replace one with a random heuristic
def mutate(sequence):
    if random.random() < mutation_rate:
        if random.random() < 0.5:
            # Swap two elements
            i, j = random.sample(range(sequence_length), 2)
            sequence[i], sequence[j] = sequence[j], sequence[i]
        else:
            # Replace one element with a random heuristic
            sequence[random.randint(0, sequence_length - 1)] = random.choice(available_heuristics)
    return sequence


# Main Genetic Algorithm
def genetic_algorithm(scheduler, tasks, processors):
    # Initialize population
    population = initialize_population()
    
    for generation in range(num_generations):
        print(f"\n--- Generation {generation + 1} ---")
        
        # Evaluate the fitness (makespan) of each sequence
        fitnesses = []
        for i, seq in enumerate(population):
            makespan = evaluate_sequence(scheduler, seq, tasks, processors)
            fitnesses.append(makespan)
            print(f"Sequence {i + 1}: {seq} -> Makespan: {makespan:.2f}")
        
        # Select the best sequences
        selected_population = select(population, fitnesses, population_size // 2)
        
        # Create the next generation through crossover and mutation
        next_generation = selected_population[:]
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_generation.append(child)
        
        # Replace old population with the new one
        population = next_generation
    
    # Final evaluation of the last generation
    print(f"\n--- Final Generation ---")
    fitnesses = []
    for i, seq in enumerate(population):
        makespan = evaluate_sequence(scheduler, seq, tasks, processors)
        fitnesses.append(makespan)
        print(f"Sequence {i + 1}: {seq} -> Makespan: {makespan:.2f}")
    
    # Return the best sequence from the final population
    best_sequence = population[fitnesses.index(min(fitnesses))]
    print(f"\nBest sequence: {best_sequence} with Makespan: {min(fitnesses):.2f}")
    
    return best_sequence


# Example usage (assuming you have Scheduler class and methods defined):
scheduler = Scheduler(tasks, processors)
best_sequence = genetic_algorithm(scheduler, tasks, processors)