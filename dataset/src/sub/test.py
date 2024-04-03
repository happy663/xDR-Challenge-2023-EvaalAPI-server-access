from collections import deque


def meets_condition(x, y):
    # 条件を満たすかどうかのチェック
    print(x, y)

    if x == 10 and y == 10:
        return True

    return False


def bfs_nearest_point():
    queue = deque([(0, 0)])
    visited = set([(0, 0)])

    while queue:
        x, y = queue.popleft()

        if meets_condition(x, y):
            return (x, y)

        for dx, dy in [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
        ]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited:
                queue.append((nx, ny))
                visited.add((nx, ny))

    return None


if __name__ == "__main__":
    print(bfs_nearest_point())
