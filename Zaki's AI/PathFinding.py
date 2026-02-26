import pygame
import random
import sys
import math
import itertools
import time
import heapq

pygame.init()

# ----- Constants -----
GRID_WIDTH, GRID_HEIGHT = 800, 800
SIDEBAR_WIDTH = 250
WIDTH = GRID_WIDTH + SIDEBAR_WIDTH
HEIGHT = GRID_HEIGHT
ROWS, COLS = 50, 50
SQUARE_SIZE = GRID_WIDTH // COLS  # 40px
NUMBER_OF_NODES = 30
SLEEPY_TIME = 0.02

# Colors
BG_COLOUR = (15, 15, 15)
SIDEBAR_COLOUR = (30, 30, 30)
WHITE = (230, 230, 230)
OBSTACLE_COLOUR = (80, 80, 80)
NODE_COLOUR = (255, 255, 255)
BUTTON_COLOUR = (50, 50, 50)
BUTTON_HOVER_COLOUR = (70, 70, 70)
ACTIVE_BUTTON_COLOUR = (200, 50, 50)
BEST_PATH_COLOUR = (128, 209, 255)
TEST_PATH_COLOUR = (100, 100, 100)
TEXT_COLOUR = (220, 220, 220)
ETA_COLOUR = (255, 200, 0)

# ----- Pygame Setup -----
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Path finding with Obstacles")
FONT = pygame.font.SysFont('Arial', 18, bold=True)
STAT_FONT = pygame.font.SysFont('Consolas', 14, bold=True)

# Buttons
BUTTON_HEIGHT = 50
BUTTON_WIDTH = 180
BUTTON_X = GRID_WIDTH + (SIDEBAR_WIDTH - BUTTON_WIDTH) // 2
SECTION_GAP = 130

random_button = pygame.Rect(BUTTON_X, 30, BUTTON_WIDTH, BUTTON_HEIGHT)
clear_walls_button = pygame.Rect(BUTTON_X, 30 + SECTION_GAP * 0.6, BUTTON_WIDTH, BUTTON_HEIGHT)
brute_button = pygame.Rect(BUTTON_X, 30 + SECTION_GAP * 1.5, BUTTON_WIDTH, BUTTON_HEIGHT)
greedy_button = pygame.Rect(BUTTON_X, 30 + SECTION_GAP * 2.5, BUTTON_WIDTH, BUTTON_HEIGHT)
two_opt_button = pygame.Rect(BUTTON_X, 30 + SECTION_GAP * 3.5, BUTTON_WIDTH, BUTTON_HEIGHT)


# ----- Pathfinding & Smoothing -----

def heuristic(a, b):
    # Manhattan distance for grid steps
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_dist(p1, p2):
    # Euclidean distance for the smoothed lines
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_line_clear(start, end, obstacles):
    """
    Checks if a straight line between start and end intersects any obstacle
    using Bresenham's Line Algorithm for efficiency.
    """
    r0, c0 = start
    r1, c1 = end
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc - dr

    while True:
        if (r0, c0) in obstacles:
            return False
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
    return True


def smooth_path(path, obstacles):
    """
    Takes a jagged A* path and removes unnecessary nodes
    if a straight line can be drawn between them.
    """
    if not path or len(path) < 3:
        return path

    smoothed = [path[0]]
    current_idx = 0

    while current_idx < len(path) - 1:
        # Check from the end backwards to find the furthest visible node
        for check_idx in range(len(path) - 1, current_idx, -1):
            if is_line_clear(path[current_idx], path[check_idx], obstacles):
                smoothed.append(path[check_idx])
                current_idx = check_idx
                break
        else:
            # If no visible node found, move to the next node in the original path
            current_idx += 1
            if current_idx < len(path):
                smoothed.append(path[current_idx])

    return smoothed


def get_astar_path(start, end, obstacles):
    """Standard A* returning grid path and its grid-based cost."""
    if start == end: return [start], 0

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    NEIGHBORS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == end: break

        row, col = current
        for dr, dc in NEIGHBORS:
            next_node = (row + dr, col + dc)
            r, c = next_node

            if not (0 <= r < ROWS and 0 <= c < COLS):
                continue
            if next_node in obstacles:
                continue

            new_cost = cost_so_far[current] + 1
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node, end)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current

    if end not in came_from: return None, float('inf')

    # Reconstruct path
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()

    if path[0] != start: return None, float('inf')

    return path, cost_so_far[end]


# ----- Distance Matrix Manager -----
class DistanceManager:
    def __init__(self):
        self.cache = {}

    def precalculate(self, nodes, obstacles):
        self.cache = {}
        node_list = list(nodes)
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                start = node_list[i]
                end = node_list[j]

                raw_path, _ = get_astar_path(start, end, obstacles)

                if raw_path is None:
                    self.cache[(start, end)] = ([], float('inf'))
                    self.cache[(end, start)] = ([], float('inf'))
                else:
                    smoothed = smooth_path(raw_path, obstacles)
                    total_dist = sum(get_dist(smoothed[k], smoothed[k + 1]) for k in range(len(smoothed) - 1))

                    self.cache[(start, end)] = (smoothed, total_dist)
                    self.cache[(end, start)] = (smoothed[::-1], total_dist)

    def get_path_cost(self, start, end):
        if (start, end) in self.cache:
            return self.cache[(start, end)]
        # This case should ideally not be hit if precalculation is always done
        if start == end: return ([start], 0)
        return None, float('inf')


# ----- Helpers -----
def get_grid_pos_from_mouse(pos):
    x, y = pos
    if x >= GRID_WIDTH: return None
    return (y // SQUARE_SIZE, x // SQUARE_SIZE)


def generate_random_nodes(number, rows, cols, obstacles):
    locations = set()
    while len(locations) < number:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if (r, c) not in obstacles:
            locations.add((r, c))
    return list(locations)


# ----- Generators -----
def brute_force_generator(nodes, dm):
    if not nodes: return
    shortest_dist = float('inf')
    best_order = []

    start_node = nodes[0]
    remaining_nodes = nodes[1:]
    all_orders = itertools.permutations(remaining_nodes)
    total_perms = math.factorial(len(remaining_nodes))

    counter = 0
    start_time_calc = time.time()

    for p in all_orders:
        current_order = [start_node] + list(p)
        total_dist = 0
        valid_path = True

        for i in range(len(current_order)):
            u = current_order[i]
            v = current_order[(i + 1) % len(current_order)]
            _, cost = dm.get_path_cost(u, v)
            if cost == float('inf'):
                valid_path = False
                break
            total_dist += cost

        if valid_path and total_dist < shortest_dist:
            shortest_dist = total_dist
            best_order = list(current_order)
            yield best_order, list(current_order), shortest_dist, "Calc..."

        counter += 1
        if counter % 500 == 0:
            elapsed = time.time() - start_time_calc
            rate = counter / elapsed if elapsed > 0 else 1
            remaining = total_perms - counter
            eta = f"{remaining / rate:.1f}s" if rate > 0 else "inf"
            yield best_order, list(current_order), shortest_dist, eta

    yield best_order, [], shortest_dist, "Done"


def greedy_generator(nodes, dm):
    if not nodes: return
    unvisited = set(nodes)
    current = nodes[0]
    path = [current]
    unvisited.remove(current)
    total_dist = 0
    yield list(path), [], 0, "..."

    while unvisited:
        time.sleep(SLEEPY_TIME * 5)
        
        # Find the nearest unvisited neighbor
        best_neighbor, min_cost = min(
            ((neighbor, dm.get_path_cost(current, neighbor)[1]) for neighbor in unvisited),
            key=lambda item: item[1],
            default=(None, float('inf'))
        )

        if best_neighbor is None: break

        path.append(best_neighbor)
        unvisited.remove(best_neighbor)
        total_dist += min_cost
        current = best_neighbor
        yield list(path), [], total_dist, "..."

    if len(path) == len(nodes):
        _, cost_to_start = dm.get_path_cost(path[-1], path[0])
        if cost_to_start != float('inf'):
            total_dist += cost_to_start
    
    yield list(path), [], total_dist, "Done"


def two_opt_generator(nodes, dm):
    if not nodes: return

    # Start with a greedy path
    unvisited = set(nodes)
    current = nodes[0]
    path = [current]
    unvisited.remove(current)
    
    while unvisited:
        best_neighbor, _ = min(
            ((neighbor, dm.get_path_cost(current, neighbor)[1]) for neighbor in unvisited),
            key=lambda item: item[1],
            default=(None, float('inf'))
        )
        if best_neighbor:
            path.append(best_neighbor)
            unvisited.remove(best_neighbor)
            current = best_neighbor
        else:
            break # No path to remaining nodes
    
    def calculate_path_dist(p):
        dist = 0
        for i in range(len(p)):
            _, c = dm.get_path_cost(p[i], p[(i + 1) % len(p)])
            if c == float('inf'): return float('inf')
            dist += c
        return dist

    curr_dist = calculate_path_dist(path)
    yield list(path), [], curr_dist, "Opt..."

    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) -1):
                # Cost of current edges
                cost_before = dm.get_path_cost(path[i-1], path[i])[1] + dm.get_path_cost(path[j], path[j+1])[1]
                # Cost of new edges
                cost_after = dm.get_path_cost(path[i-1], path[j])[1] + dm.get_path_cost(path[i], path[j+1])[1]
                
                if cost_after < cost_before:
                    path[i:j+1] = path[j:i-1:-1]
                    curr_dist = calculate_path_dist(path)
                    improved = True
                    yield list(path), [], curr_dist, "Opt..."


    yield list(path), [], curr_dist, "Done"


# ----- Drawing -----
def draw_button(win, rect, text, is_active=False):
    mouse_pos = pygame.mouse.get_pos()
    color = ACTIVE_BUTTON_COLOUR if is_active else (
        BUTTON_HOVER_COLOUR if rect.collidepoint(mouse_pos) else BUTTON_COLOUR)
    border = (255, 100, 100) if is_active else (100, 100, 100)
    pygame.draw.rect(win, color, rect, border_radius=8)
    pygame.draw.rect(win, border, rect, 2, border_radius=8)
    text_surf = FONT.render(text, True, TEXT_COLOUR)
    win.blit(text_surf, text_surf.get_rect(center=rect.center))


def draw_stats(win, x, y, time_val, dist_val, eta, is_running):
    if not is_running and time_val is None: return
    col = TEXT_COLOUR
    t_str = f"{time_val:.2f}s" if time_val else "..."
    win.blit(STAT_FONT.render(f"Time: {t_str}", True, col), (x, y))
    d_str = f"{dist_val:.1f}" if dist_val is not None and dist_val != float('inf') else "Inf"
    win.blit(STAT_FONT.render(f"Dist: {d_str}", True, col), (x, y + 20))
    if is_running and eta:
        win.blit(STAT_FONT.render(f"ETA: {eta}", True, ETA_COLOUR), (x, y + 40))


def draw_window(win, obstacles, nodes, best_path, test_path, dm,
                stats_brute, stats_greedy, stats_2_opt,
                algo_type, start_time, eta):
    win.fill(BG_COLOUR)
    gap = GRID_WIDTH // COLS

    # 1. Draw Obstacles
    for r, c in obstacles:
        pygame.draw.rect(win, OBSTACLE_COLOUR, (c * gap, r * gap, gap, gap))
        pygame.draw.rect(win, (40, 40, 40), (c * gap, r * gap, gap, gap), 1)

    # 2. Draw Paths
    def draw_complex_path(node_order, color, width):
        if len(node_order) < 2: return
        # Draw tour
        for i in range(len(node_order)):
            u, v = node_order[i], node_order[(i + 1) % len(node_order)]
            points_list, _ = dm.get_path_cost(u, v)
            if points_list:
                pixel_points = [(c * gap + gap // 2, r * gap + gap // 2) for r, c in points_list]
                if len(pixel_points) > 1:
                    pygame.draw.lines(win, color, False, pixel_points, width)
        # Draw test segments for brute force
        if algo_type == 'brute' and test_path and len(test_path) > 1:
             for i in range(len(test_path)-1):
                 u, v = test_path[i], test_path[i+1]
                 points_list, _ = dm.get_path_cost(u, v)
                 if points_list:
                     pixel_points = [(c * gap + gap // 2, r * gap + gap // 2) for r, c in points_list]
                     if len(pixel_points) > 1:
                         pygame.draw.lines(win, TEST_PATH_COLOUR, False, pixel_points, 1)


    if best_path: draw_complex_path(best_path, BEST_PATH_COLOUR, 3)

    # 3. Draw Nodes
    for r, c in nodes:
        padding = gap // 6
        rect = (c * gap + padding, r * gap + padding, gap - 2 * padding, gap - 2 * padding)
        pygame.draw.rect(win, NODE_COLOUR, rect)
        pygame.draw.rect(win, (0, 0, 0), rect, 1)

    # 4. Sidebar
    sidebar_rect = pygame.Rect(GRID_WIDTH, 0, SIDEBAR_WIDTH, GRID_HEIGHT)
    pygame.draw.rect(win, SIDEBAR_COLOUR, sidebar_rect)
    pygame.draw.line(win, (60, 60, 60), (GRID_WIDTH, 0), (GRID_WIDTH, HEIGHT), 2)

    draw_button(win, random_button, "Random Nodes")
    draw_button(win, clear_walls_button, "Clear Walls")

    draw_button(win, brute_button, "Brute Force", algo_type == "brute")
    draw_stats(win, BUTTON_X, brute_button.bottom + 5, stats_brute['time'], stats_brute['dist'],
               eta if algo_type == "brute" else None, algo_type == "brute")

    draw_button(win, greedy_button, "Greedy", algo_type == "greedy")
    draw_stats(win, BUTTON_X, greedy_button.bottom + 5, stats_greedy['time'], stats_greedy['dist'],
               eta if algo_type == "greedy" else None, algo_type == "greedy")

    draw_button(win, two_opt_button, "2-Opt", algo_type == "2-opt")
    draw_stats(win, BUTTON_X, two_opt_button.bottom + 5, stats_2_opt['time'], stats_2_opt['dist'], None, algo_type == "2-opt")

    pygame.display.update()


# ----- Main Loop -----
def main():
    run = True
    clock = pygame.time.Clock()

    obstacle_set = set()
    node_list = generate_random_nodes(NUMBER_OF_NODES, ROWS, COLS, obstacle_set)
    dm = DistanceManager()

    best_path = []
    test_path = []
    algo_gen = None
    algo_type = None
    start_time = None
    current_eta = ""

    stats_brute = {'time': None, 'dist': None}
    stats_greedy = {'time': None, 'dist': None}
    stats_2_opt = {'time': None, 'dist': None}

    mouse_drawing_mode = None

    def reset_algo():
        nonlocal algo_gen, algo_type, start_time, test_path, current_eta
        algo_gen = None
        algo_type = None
        start_time = None
        test_path = []
        current_eta = ""

    precalculation_needed = True

    while run:
        clock.tick(60)

        if precalculation_needed and node_list:
            dm.precalculate(node_list, obstacle_set)
            precalculation_needed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if random_button.collidepoint(pos):
                    reset_algo()
                    best_path = []
                    stats_brute = stats_greedy = stats_2_opt = {'time': None, 'dist': None}
                    node_list = generate_random_nodes(NUMBER_OF_NODES, ROWS, COLS, obstacle_set)
                    precalculation_needed = True
                elif clear_walls_button.collidepoint(pos):
                    reset_algo()
                    obstacle_set.clear()
                    precalculation_needed = True
                elif brute_button.collidepoint(pos) and not algo_gen:
                    reset_algo()
                    algo_type = "brute"
                    start_time = time.time()
                    algo_gen = brute_force_generator(node_list, dm)
                elif greedy_button.collidepoint(pos) and not algo_gen:
                    reset_algo()
                    algo_type = "greedy"
                    start_time = time.time()
                    algo_gen = greedy_generator(node_list, dm)
                elif two_opt_button.collidepoint(pos) and not algo_gen:
                    reset_algo()
                    algo_type = "2-opt"
                    start_time = time.time()
                    algo_gen = two_opt_generator(node_list, dm)
                else:
                    gp = get_grid_pos_from_mouse(pos)
                    if gp:
                        r, c = gp
                        if (r, c) not in node_list:
                            if (r, c) in obstacle_set:
                                mouse_drawing_mode = 'remove'
                                obstacle_set.remove((r, c))
                            else:
                                mouse_drawing_mode = 'add'
                                obstacle_set.add((r, c))
                            reset_algo()
                            best_path = []
                            precalculation_needed = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_drawing_mode = None
            elif event.type == pygame.MOUSEMOTION and mouse_drawing_mode:
                gp = get_grid_pos_from_mouse(pygame.mouse.get_pos())
                if gp:
                    r, c = gp
                    if (r, c) not in node_list:
                        if mouse_drawing_mode == 'add' and (r,c) not in obstacle_set:
                            obstacle_set.add((r, c))
                            precalculation_needed = True
                        elif mouse_drawing_mode == 'remove' and (r, c) in obstacle_set:
                            obstacle_set.remove((r, c))
                            precalculation_needed = True

        if algo_gen:
            try:
                best_path, test_path, dist, current_eta = next(algo_gen)
            except StopIteration:
                end_t = time.time() - start_time
                stats = {'time': end_t, 'dist': dist}
                if algo_type == "brute": stats_brute = stats
                elif algo_type == "greedy": stats_greedy = stats
                elif algo_type == "2-opt": stats_2_opt = stats
                reset_algo()

        draw_window(screen, obstacle_set, node_list, best_path, test_path, dm,
                    stats_brute, stats_greedy, stats_2_opt, algo_type, start_time, current_eta)

    pygame.quit()
    sys.exit()

main()