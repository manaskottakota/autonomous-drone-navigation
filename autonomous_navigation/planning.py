"""Grid A* route planner used only as a map-aware conventional baseline."""
from __future__ import annotations

import heapq

import numpy as np

from autonomous_navigation.obstacle_env import ObstacleShape, obstacle_parts


def astar_route(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: tuple[ObstacleShape, ...],
    cruise_z: float,
    resolution: float = 0.1,
    padding: float = 0.32,
) -> list[np.ndarray] | None:
    """Plan an 8-connected XY route around known boxes at a fixed cruise height."""
    lower, upper = -1.8, 1.8
    count = int(round((upper - lower) / resolution)) + 1
    def point(node: tuple[int, int]) -> np.ndarray:
        return np.array([lower + node[0] * resolution, lower + node[1] * resolution, cruise_z])

    def blocked(a: np.ndarray, b: np.ndarray, margin: float = padding) -> bool:
        for obstacle in (part for item in obstacles for part in obstacle_parts(item)):
            center = np.asarray(obstacle.center)
            half = np.asarray(obstacle.half_extents)
            if not (center[2] - half[2] - margin <= cruise_z <= center[2] + half[2] + margin):
                continue
            lo = center[:2] - half[:2] - margin
            hi = center[:2] + half[:2] + margin
            delta = b[:2] - a[:2]
            t_min, t_max = 0.0, 1.0
            for axis in range(2):
                if abs(delta[axis]) < 1e-10:
                    if a[axis] < lo[axis] or a[axis] > hi[axis]:
                        t_min, t_max = 1.0, 0.0
                        break
                    continue
                t1, t2 = (lo[axis] - a[axis]) / delta[axis], (hi[axis] - a[axis]) / delta[axis]
                t_min, t_max = max(t_min, min(t1, t2)), min(t_max, max(t1, t2))
            if t_min <= t_max and t_max >= 0 and t_min <= 1:
                return True
        return False

    def attach_node(endpoint: np.ndarray) -> tuple[int, int] | None:
        candidates = [(i, j) for i in range(count) for j in range(count)]
        candidates.sort(key=lambda node: float(np.linalg.norm(point(node)[:2] - endpoint[:2])))
        for node in candidates:
            candidate = point(node)
            if blocked(candidate, candidate) or blocked(endpoint, candidate, margin=0.08):
                continue
            return node
        return None

    start_node = attach_node(start)
    goal_node = attach_node(goal)
    if start_node is None or goal_node is None:
        return None
    neighbors = [(dx, dy, np.hypot(dx, dy) * resolution)
                 for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    frontier = [(float(np.linalg.norm(point(start_node)[:2] - point(goal_node)[:2])), 0.0, start_node)]
    cost = {start_node: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    visited = set()
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current in visited:
            continue
        if current == goal_node:
            nodes = [current]
            while nodes[-1] != start_node:
                nodes.append(parent[nodes[-1]])
            raw_route = [point(node) for node in reversed(nodes)]
            smoothed = [raw_route[0]]
            index = 0
            while index < len(raw_route) - 1:
                candidate = len(raw_route) - 1
                while candidate > index + 1 and blocked(raw_route[index], raw_route[candidate]):
                    candidate -= 1
                smoothed.append(raw_route[candidate])
                index = candidate
            route = [start.copy(), *smoothed]
            if np.linalg.norm(route[-1] - goal) > 1e-6:
                route.append(goal.copy())
            return route
        visited.add(current)
        for dx, dy, step_cost in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < count and 0 <= nxt[1] < count) or nxt in visited:
                continue
            if blocked(point(current), point(nxt)):
                continue
            new_cost = current_cost + step_cost
            if new_cost >= cost.get(nxt, float("inf")):
                continue
            cost[nxt] = new_cost
            parent[nxt] = current
            heuristic = float(np.linalg.norm(point(nxt)[:2] - point(goal_node)[:2]))
            heapq.heappush(frontier, (new_cost + heuristic, new_cost, nxt))
    return None
