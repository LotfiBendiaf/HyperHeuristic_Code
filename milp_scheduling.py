import pulp

from hyperHeuristic_Scheduling import processors, tasks

SOLVER_TIME_LIMIT_SECONDS = 30


def solve_exact_milp(time_limit=SOLVER_TIME_LIMIT_SECONDS):
    """Solve the same assignment model used by the hyper-heuristic exactly."""
    prob = pulp.LpProblem("Multiprocessor_Scheduling", pulp.LpMinimize)

    x = {
        (task.task_id, proc.proc_id): pulp.LpVariable(
            f"x_{task.task_id}_{proc.proc_id}", 0, 1, pulp.LpBinary
        )
        for task in tasks
        for proc in processors
    }
    processor_load = {
        proc.proc_id: pulp.LpVariable(f"T_{proc.proc_id}", lowBound=0)
        for proc in processors
    }
    makespan = pulp.LpVariable("makespan", lowBound=0)

    prob += makespan, "Minimize_makespan"

    for task in tasks:
        prob += (
            pulp.lpSum(x[task.task_id, proc.proc_id] for proc in processors) == 1,
            f"TaskAssignment_{task.task_id}",
        )

    for proc in processors:
        prob += (
            processor_load[proc.proc_id]
            == pulp.lpSum(
                x[task.task_id, proc.proc_id] * (task.workload / proc.speed)
                for task in tasks
            ),
            f"Workload_{proc.proc_id}",
        )
        prob += makespan >= processor_load[proc.proc_id], f"Makespan_{proc.proc_id}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    return prob, x, processor_load, makespan


if __name__ == "__main__":
    prob, x, processor_load, makespan = solve_exact_milp()
    status = pulp.LpStatus[prob.status]

    print(f"Solver status: {status}")
    print(f"Time limit: {SOLVER_TIME_LIMIT_SECONDS}s")
    if pulp.value(makespan) is not None:
        label = "Optimal Makespan" if status == "Optimal" else "Best MILP Makespan"
        print(f"{label}: {pulp.value(makespan):.4f}s")
        for proc in processors:
            assigned_tasks = [
                task.task_id
                for task in tasks
                if pulp.value(x[task.task_id, proc.proc_id]) > 0.5
            ]
            print(
                f"Processor {proc.proc_id} (speed={proc.speed:2d}): "
                f"tasks={assigned_tasks}, workload={pulp.value(processor_load[proc.proc_id]):.4f}s"
            )
