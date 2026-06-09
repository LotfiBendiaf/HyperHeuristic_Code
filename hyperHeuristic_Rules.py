class Task:
    def __init__(self, task_id, workload, priority=1, deadline=None):
        self.task_id = task_id
        self.workload = workload  # computation time in cycles
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
        self.tasks = []

    def execute_task(self, task):
        self.tasks.append(task)

    def get_total_workload(self):
        # execution time = workload / speed  (cycles / cycles-per-second = seconds)
        return sum(t.workload / self.speed for t in self.tasks)

    def reset(self):
        self.tasks = []


class Scheduler:
    def __init__(self, tasks, processors):
        self.tasks = tasks
        self.processors = processors

    # ------------------------------------------------------------------ #
    # Sequencing rules — return a reordered copy of subset               #
    # ------------------------------------------------------------------ #
    def sequencing_tasks(self, policy, subset):
        match policy:
            case "firstIn_firstOut":
                return sorted(subset, key=lambda t: t.task_id)
            case "lastIn_firstOut":
                return sorted(subset, key=lambda t: t.task_id, reverse=True)
            case "priority_based":
                return sorted(subset, key=lambda t: t.priority, reverse=True)
            case "min_priority":
                return sorted(subset, key=lambda t: t.priority)
            case "short_job_first":
                return sorted(subset, key=lambda t: t.workload)
            case "long_job_first":
                return sorted(subset, key=lambda t: t.workload, reverse=True)
            case _:
                raise ValueError(f"Unknown sequencing policy: {policy}")

    # ------------------------------------------------------------------ #
    # Scheduling rules — assign subset tasks to processors               #
    # ------------------------------------------------------------------ #
    def schedule_tasks(self, policy, subset):
        match policy:
            case "min_min":
                self._min_min(subset)
            case "max_min":
                self._max_min(subset)
            case "min_queued_elements":
                self._min_queued_elements(subset)
            case "load_balancing":
                self._load_balancing(subset)
            case "greedy_best_fit":
                self._greedy_best_fit(subset)
            case "threshold_based":
                self._threshold_based(subset)
            case _:
                raise ValueError(f"Unknown scheduling policy: {policy}")

    def _assign(self, task, processor):
        task.assigned_processor = processor.proc_id
        processor.execute_task(task)

    def _least_loaded(self):
        return min(self.processors, key=lambda p: p.get_total_workload())

    # Assign each task to the processor with the fewest queued tasks
    def _min_queued_elements(self, subset):
        for task in subset:
            self._assign(task, min(self.processors, key=lambda p: len(p.tasks)))

    # Assign each task to the processor with the lowest total execution time
    def _load_balancing(self, subset):
        for task in subset:
            self._assign(task, self._least_loaded())

    # Assign each task to the processor that minimises (current_load + new_task_time)
    def _greedy_best_fit(self, subset):
        for task in subset:
            best = min(
                self.processors,
                key=lambda p: p.get_total_workload() + task.workload / p.speed,
            )
            self._assign(task, best)

    # Heavy tasks go to the fastest processor; light tasks to the least-loaded one.
    # The threshold is the average workload of the current subset (adaptive).
    def _threshold_based(self, subset):
        if not subset:
            return
        threshold = sum(t.workload for t in subset) / len(subset)
        for task in subset:
            if task.workload > threshold:
                best = max(self.processors, key=lambda p: p.speed)
            else:
                best = self._least_loaded()
            self._assign(task, best)

    # Iteratively schedule the (task, processor) pair with the smallest completion time
    def _min_min(self, subset):
        remaining = list(subset)
        while remaining:
            min_task, min_time, best_proc = None, float("inf"), None
            for task in remaining:
                for proc in self.processors:
                    t = proc.get_total_workload() + task.workload / proc.speed
                    if t < min_time:
                        min_time, min_task, best_proc = t, task, proc
            self._assign(min_task, best_proc)
            remaining.remove(min_task)

    # Iteratively schedule the task whose best-case completion time is largest
    def _max_min(self, subset):
        remaining = list(subset)
        while remaining:
            task_best = []
            for task in remaining:
                best_time, best_proc = float("inf"), None
                for proc in self.processors:
                    t = proc.get_total_workload() + task.workload / proc.speed
                    if t < best_time:
                        best_time, best_proc = t, proc
                task_best.append((task, best_time, best_proc))

            # pick the task with the maximum of its per-task minimum times
            task_best.sort(key=lambda x: x[1], reverse=True)
            max_task, _, best_proc = task_best[0]
            self._assign(max_task, best_proc)
            remaining.remove(max_task)


# Exported rule name lists used by the GA
SEQUENCING_RULES = [
    "firstIn_firstOut",
    "lastIn_firstOut",
    "short_job_first",
    "long_job_first",
    "priority_based",
    "min_priority",
]

SCHEDULING_RULES = [
    "min_min",
    "max_min",
    "min_queued_elements",
    "threshold_based",
    "load_balancing",
    "greedy_best_fit",
]
