import numpy as np
from heapq import heappop, heappush
from functools import lru_cache

class Node:
    def __init__(self, pose):
        self.pose = tuple(pose)
        self.x, self.y = pose
        self.g_value = 0
        self.h_value = 0
        self.f_value = 0
        self.parent = None

    def __lt__(self, other):
        return self.f_value < other.f_value

    def __eq__(self, other):
        return self.pose == other.pose

class AStar:
    def __init__(self, map_path):
        self.map_path = map_path
        self.map = self.load_map(self.map_path).astype(int)
        self.resolution = 0.05
        self.y_dim, self.x_dim = self.map.shape
        print(f'Map size: ({self.x_dim}, {self.y_dim})')

    def load_map(self, path):
        return np.load(path)

    def reset_map(self):
        self.map = self.load_map(self.map_path)

    @lru_cache(maxsize=None)
    def heuristic(self, current_pose, goal_pose):
        x_gap, y_gap = goal_pose[0] - current_pose[0], goal_pose[1] - current_pose[1]
        return np.sqrt(x_gap**2 + y_gap**2)

    def get_successor(self, node):
        successors = []
        directions = [(1, 1, 1.414), (0, 1, 1), (-1, 1, 1.414), 
                      (-1, 0, 1), (-1, -1, 1.414), (0, -1, 1), 
                      (1, -1, 1.414), (1, 0, 1)]

        for dx, dy, cost in directions:
            x_, y_ = node.x + dx, node.y + dy
            if 0 <= x_ < self.y_dim and 0 <= y_ < self.x_dim and self.map[x_, y_] == 0:
                successor = Node((x_, y_))
                successor.g_value = node.g_value + cost
                successors.append(successor)
        return successors

    def calculate_path(self, node):
        path = []
        while node:
            path.append(list(node.pose))
            node = node.parent
        path.reverse()
        print(f'Path length: {len(path)}')
        return path

    def plan(self, start_ind, goal_ind):
        start_node = Node(start_ind)
        goal_node = Node(goal_ind)
        start_node.h_value = self.heuristic(start_node.pose, goal_node.pose)
        start_node.f_value = start_node.g_value + start_node.h_value

        self.reset_map()

        open_list = []
        open_set = {start_node.pose: start_node}
        closed_set = set()
        heappush(open_list, start_node)

        while open_list:
            current = heappop(open_list)
            open_set.pop(current.pose, None)
            closed_set.add(current.pose)

            if current == goal_node:
                print('Goal reached')
                return self.calculate_path(current)

            for successor in self.get_successor(current):
                if successor.pose in closed_set:
                    continue

                successor.parent = current
                successor.h_value = self.heuristic(successor.pose, goal_node.pose)
                successor.f_value = successor.g_value + successor.h_value

                if successor.pose in open_set:
                    if open_set[successor.pose].g_value > successor.g_value:
                        open_set[successor.pose] = successor
                        heappush(open_list, successor)
                else:
                    open_set[successor.pose] = successor
                    heappush(open_list, successor)

        print('Path not found')
        return None

    def run(self, cost_map, start_ind, goal_ind):
        if cost_map[start_ind[0], start_ind[1]] == 0 and cost_map[goal_ind[0], goal_ind[1]] == 0:
            return self.plan(start_ind, goal_ind)
        else:
            print('Start or goal position is occupied')
