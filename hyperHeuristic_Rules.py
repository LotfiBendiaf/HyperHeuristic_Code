# Define Task and Processor classes
class Task:
    def __init__(self, task_id, workload, priority=1, deadline=None):
        self.task_id = task_id
        self.workload = workload  # Computation time in cycles 
        self.priority = priority
        self.deadline = deadline
        self.assigned_processor = None

    def __repr__(self):
        return f"Task(id={self.task_id}, workload={self.workload}, priority={self.priority})"

class Processor:
    def __init__(self, proc_id, speed, power_usage=10):
        self.proc_id = proc_id
        self.speed = speed  # cycles per second
        self.power_usage = power_usage
        self.tasks = []  # Tasks assigned to this processor
        self.total_workload = 0  # Total workload assigned to this processor

    def execute_task(self, task):
        # execution_time = task.workload * self.speed
        self.tasks.append(task)
        # self.total_workload += execution_time  # Add the task workload to the processor's total workload

    def get_total_workload(self):
        self.total_workload = 0
        for task in self.tasks:
            self.total_workload += task.workload * self.speed
        
        return self.total_workload

class Scheduler:
    def __init__(self, tasks, processors):
        self.tasks = tasks
        self.processors = processors

    def sequencing_tasks(self, policy, subset):
        match policy:
            case "firstIn_firstOut":
                return self.fcfs_schedule(subset)
            case "lastIn_firstOut":
                return self.lcfs_schedule(subset)
            case "priority_based":
                return self.priority_based_schedule(subset)
            case "min_priority":
                return self.min_priority_schedule(subset)
            case "short_job_first":
                return self.sjf_schedule(subset)
            case "long_job_first":
                return self.ljf_schedule(subset)
            case _:
                raise ValueError(f"Unknown sequencing policy: {policy}")
            
    def schedule_tasks(self, policy, subset):
        match policy:
            # case "round_robin":
            #     self.round_robin_schedule(subset)
            case "min_min":
                self.min_min_schedule(subset)
            case "min_queued_elements":
                self.min_queued_elements(subset)
            case "max_min":
                self.max_min_schedule(subset)
            case "load_balancing":
                self.load_balancing_schedule(subset)
            case "greedy_best_fit":
                self.greedy_best_fit_schedule(subset)
            # case "weighted_round_robin":
            #     self.weighted_round_robin_schedule(subset)
            case "threshold_based":
                self.threshold_based_schedule(subset)
            case _:
                raise ValueError(f"Unknown scheduling policy: {policy}")

    ### Base Functions ###
    def assign_task_to_processor(self, task, processor):
        task.assigned_processor = processor.proc_id
        processor.execute_task(task)

    # Function to calculate the total workload on a processor
    def select_best_processor(self, task):
        # Select processor with the least total workload
        return min(self.processors, key=lambda p: p.get_total_workload())
    
    ### Sequencing Rules ###
    # SJF heuristic (Shortest Job First)
    def sjf_schedule(self, task_subset):
        sorted_tasks = sorted(task_subset, key=lambda t: t.workload)
        return sorted_tasks

    # LJF heuristic (Longest Job First)
    def ljf_schedule(self, task_subset):
        sorted_tasks = sorted(task_subset, key=lambda t: t.workload, reverse=True)
        return sorted_tasks

    # Priority-Based Scheduling (High Priority First)
    def priority_based_schedule(self, task_subset):
        sorted_tasks = sorted(task_subset, key=lambda t: t.priority, reverse=True)
        return sorted_tasks

    # Priority-Based Scheduling (Low Priority First)
    def min_priority_schedule(self, task_subset):
        sorted_tasks = sorted(task_subset, key=lambda t: t.priority)
        return sorted_tasks

    # FCFS heuristic (First-Come, First-Served)
    def fcfs_schedule(self, task_subset):
        sorted_tasks = sorted(task_subset, key=lambda t: t.task_id)
        return sorted_tasks

    # LCFS heuristic (Last-Come, First-Served)
    def lcfs_schedule(self, task_subset):
        sorted_tasks = sorted(task_subset, key=lambda t: t.task_id, reverse=True)
        return sorted_tasks

    # Scheduling Rules
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

    # Simple Round-Robin scheduling across processors
    def round_robin_schedule(self, task_subset):
        proc_idx = 0
        for task in task_subset:
            self.assign_task_to_processor(task, self.processors[proc_idx])
            proc_idx = (proc_idx + 1) % len(self.processors)

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

# Define execution costs and processor speeds
execution_cost = [10, 55, 108, 98, 9, 77, 81, 59, 32, 30, 39, 45, 18, 130, 120, 60, 25, 71]
tasks = [Task(i, execution_cost[i]) for i in range(len(execution_cost))]

processor_speed = [1, 2, 4, 8]
processors = [Processor(j, processor_speed[j]) for j in range(len(processor_speed))]

# Initialize the scheduler
scheduler = Scheduler(tasks, processors)

sequencing_rules = ['firstIn_firstOut', 'lastIn_firstOut', 'short_job_first', 'long_job_first']
scheduling_rules = ['min_min', 'max_min', 'min_queued_elements', 'threshold_based']

# Define a subset of tasks for demonstration
sep1 = 10  # Number of tasks to consider in the subset
task_subset1 = tasks[:sep1]
task_subset2 = tasks[sep1:]

# Apply heuristic on the task subset
print(f"\nRunning heuristic on the first {sep1} tasks:")
phase_one = scheduler.sequencing_tasks(policy=sequencing_rules[0], subset=task_subset1)
scheduler.schedule_tasks(policy=scheduling_rules[0], subset=phase_one)
# Task IDs assigned to each processor
for processor in processors:
    print(f"Total workload {processor.get_total_workload():.2f}, ", end="")
    print(f"Tasks assigned to Processor {processor.proc_id}: ", end="")
    
    # Print all task IDs assigned to the processor
    task_ids = [task.task_id for task in processor.tasks]
    print(task_ids if task_ids else "No tasks assigned")

# Calculate and print the makespan (maximum total workload across all processors)
makespan = max(processor.get_total_workload() for processor in processors)
print(f"The makespan: {makespan:.2f}")

print(f"\nRunning heuristic on the remaining tasks:")
phase_two = scheduler.sequencing_tasks(policy=sequencing_rules[2], subset=task_subset2)
scheduler.schedule_tasks(policy=scheduling_rules[1], subset=phase_two)
# Task IDs assigned to each processor
for processor in processors:
    print(f"Total workload {processor.get_total_workload():.2f}, ", end="")
    print(f"Tasks assigned to Processor {processor.proc_id}: ", end="")
    
    # Print all task IDs assigned to the processor
    task_ids = [task.task_id for task in processor.tasks]
    print(task_ids if task_ids else "No tasks assigned")

# Calculate and print the makespan (maximum total workload across all processors)
makespan = max(processor.get_total_workload() for processor in processors)
print(f"The makespan: {makespan:.2f}")