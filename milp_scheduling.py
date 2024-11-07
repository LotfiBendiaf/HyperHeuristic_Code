import pulp
import random

# Define Task and Processor classes with integer workload and execution cost
class Task:
    def __init__(self, task_id, workload):
        self.task_id = task_id
        self.workload = workload

class Processor:
    def __init__(self, proc_id, speed):
        self.proc_id = proc_id
        self.speed = speed  # Instructions per second or computational power

# Define execution costs and create tasks
execution_cost = [10, 55, 108, 98, 9, 77, 81, 59, 32, 30, 39, 45, 18, 130, 120, 60, 25, 71]
tasks = [Task(i, execution_cost[i]) for i in range(len(execution_cost))]

# Define processors with integer speeds
processors_speeds = [1, 2, 4, 8]
processors = [Processor(j, processors_speeds[j]) for j in range(len(processors_speeds))]  # 3 processors with random speeds

# Initialize the MILP problem
prob = pulp.LpProblem("Multiprocessor_Scheduling", pulp.LpMinimize)

# Decision variables: x_ij is 1 if task i is assigned to processor j, 0 otherwise
x = {(i.task_id, j.proc_id): pulp.LpVariable(f"x_{i.task_id}_{j.proc_id}", 0, 1, pulp.LpBinary)
     for i in tasks for j in processors}

# Integer variables for the workload (makespan) on each processor
T = {j.proc_id: pulp.LpVariable(f"T_{j.proc_id}", 0, None, pulp.LpInteger) for j in processors}

# Objective: Minimize the maximum workload (makespan) with integer makespan variable
makespan = pulp.LpVariable("makespan", 0, None, pulp.LpInteger)
prob += makespan, "Minimize_makespan"

# Constraint 1: Each task must be assigned to exactly one processor
for i in tasks:
    prob += pulp.lpSum(x[i.task_id, j.proc_id] for j in processors) == 1, f"TaskAssignment_{i.task_id}"

# Constraint 2: The total workload of each processor is the sum of the workloads of the tasks assigned to it
for j in processors:
    prob += T[j.proc_id] == pulp.lpSum(x[i.task_id, j.proc_id] * i.workload * j.speed for i in tasks), f"Workload_{j.proc_id}"

# Constraint 3: The makespan is at least the total workload on each processor
for j in processors:
    prob += makespan >= T[j.proc_id], f"Makespan_{j.proc_id}"

# Solve the problem
prob.solve()

# Output the results
if pulp.LpStatus[prob.status] == "Optimal":
    print(f"Optimal Makespan: {pulp.value(makespan):.0f}")
    for j in processors:
        assigned_tasks = [i.task_id for i in tasks if pulp.value(x[i.task_id, j.proc_id]) > 0.5]
        print(f"Processor {j.proc_id} is assigned tasks: {assigned_tasks}")
        print(f"  Total workload on Processor {j.proc_id}: {pulp.value(T[j.proc_id]):.0f}")
else:
    print(f"Solver status: {pulp.LpStatus[prob.status]}")