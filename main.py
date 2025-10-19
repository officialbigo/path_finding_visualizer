"""
Path Finding Visualizer
=======================

A Pygame-based interactive visualization tool for pathfinding algorithms including 
Depth-First Search (DFS), Breadth-First Search (BFS), Dijkstra's algorithm, and A* search.

Author: TRK Aashish
Date: December 2023

Features
--------
- Interactive wall drawing with mouse
- Custom start/end point selection
- Random maze generation using recursive backtracker algorithm
- Real-time algorithm visualization with color-coded exploration
- Performance statistics (path length, nodes visited, execution time)
- Adaptive fullscreen support with automatic scaling
- Algorithm interruption capability

Grid Values
-----------
    0: Walkable path (Black)
    -1: Wall/obstacle (White)
    2: Final path (Dark Red)
    3: Start marker (Green)
    4: End marker (Red)

Controls
--------
    Mouse Drag: Draw walls on the grid
    S: Set custom start point (click after pressing)
    E: Set custom end point (click after pressing)
    1: Run DFS (Depth-First Search)
    2: Run BFS (Breadth-First Search)
    3: Run Dijkstra's Algorithm
    4: Run A* Search
    C: Clear path only (keep walls)
    H: Hard clear (reset everything)
    G: Generate random maze
    F: Toggle fullscreen mode

Algorithms
----------
1. **DFS (Depth-First Search)**: Iterative implementation using explicit stack
2. **BFS (Breadth-First Search)**: Queue-based exploration guaranteeing shortest path
3. **Dijkstra's Algorithm**: Priority queue-based shortest path algorithm
4. **A* Search**: Heuristic-based pathfinding using Euclidean distance

Visualization Colors
--------------------
- Green: Start position
- Red: End position
- Black: Walkable path
- White: Walls/obstacles
- Dark Red: Final computed path
- Orange: Permanently visited nodes
- Light Orange: Nodes being explored (A*/Dijkstra only)

Requirements
------------
- Python 3.7+
- Pygame 2.0+

Installation
------------
    pip install pygame
    python pathfinder.py

Usage Example
-------------
    1. Run the program
    2. Draw walls by clicking and dragging mouse
    3. Press 'S' and click to set start point
    4. Press 'E' and click to set end point
    5. Press 1-4 to run different pathfinding algorithms
    6. Press 'C' to clear path and try another algorithm
    7. Press 'G' to generate a random maze
"""

from collections import deque
from collections import defaultdict
import pygame
import heapq
import random
import time


# ========================= EXCEPTION CLASSES =========================

class AlgorithmInterrupted(Exception):
    """Custom exception raised when user interrupts algorithm execution via C or H key."""
    pass


# ========================= GRAPH UTILITIES =========================

def convert_grid_to_graph(grid):
    """
    Convert a 2D grid into a graph representation for Dijkstra and A* algorithms.
    
    Args:
        grid (list[list[int]]): 2D grid where cells contain state values
    
    Returns:
        defaultdict: Graph as adjacency list where graph[node][neighbor] = edge_weight
        
    Notes:
        - Only considers 4-directional movement (up, down, left, right)
        - Treats cells with values 0, 3, 4 as walkable
        - Edge weight between adjacent cells is 1 (uniform cost)
    """
    moves = [[1,0], [-1,0], [0,-1], [0,1]]  # down, up, left, right
    graph = defaultdict(dict)
    rows = len(grid)
    cols = len(grid[0])
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 or grid[i][j] == 3 or grid[i][j] == 4:
                for move in moves:
                    new_i, new_j = i + move[0], j + move[1]
                    if valid(grid, new_i, new_j) and (grid[new_i][new_j] == 0 or 
                                                       grid[new_i][new_j] == 3 or 
                                                       grid[new_i][new_j] == 4):
                        graph[(i,j)][(new_i,new_j)] = 1
    return graph


def valid(grid, x, y):
    """
    Check if a grid position is valid and walkable.
    
    Args:
        grid (list[list[int]]): The 2D grid
        x (int): Row coordinate
        y (int): Column coordinate
    
    Returns:
        bool: True if position is within bounds and walkable, False otherwise
    """
    if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and (grid[x][y] == 0 or 
                                                           grid[x][y] == 3 or 
                                                           grid[x][y] == 4):
        return True
    return False


def check_interrupt():
    """
    Check if user wants to interrupt the currently running algorithm.
    
    Raises:
        AlgorithmInterrupted: If C or H key is pressed during execution
        
    Notes:
        Also handles QUIT event to allow clean program termination
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c or event.key == pygame.K_h:
                raise AlgorithmInterrupted(event.key)


# ========================= ALGORITHM IMPLEMENTATIONS =========================

def heuristic(node, end):
    """
    Calculate Euclidean distance heuristic for A* algorithm.
    
    Args:
        node (tuple): Current position as (row, col)
        end (tuple): Goal position as (row, col)
    
    Returns:
        float: Euclidean distance between node and end
    """
    return ((node[0] - end[0])**2 + (node[1] - end[1])**2) ** 0.5


def astar(graph, start, end, screen, grid, grid_size, display_state):
    """
    A* pathfinding algorithm using Euclidean distance heuristic.
    
    Args:
        graph (dict): Graph representation from convert_grid_to_graph()
        start (tuple): Starting position (row, col)
        end (tuple): Goal position (row, col)
        screen (pygame.Surface): Display surface for visualization
        grid (list[list[int]]): 2D grid for visualization
        grid_size (int): Size of each grid cell in pixels
        display_state (DisplayState): Display configuration object
    
    Returns:
        tuple: (path, nodes_visited) where path is list of (row,col) tuples and 
               nodes_visited is count of explored nodes
               
    Algorithm:
        Uses f(n) = g(n) + h(n) where:
        - g(n) is actual cost from start to n
        - h(n) is heuristic estimate from n to goal
    """
    open_set = [(0, start)]  # Priority queue: (f_score, node)
    previous = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {}
    
    # Initialize f_scores with heuristic
    for node in graph:
        f_score[node] = heuristic(node, end)
    
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    perma_visited = [[False for _ in range(cols)] for _ in range(rows)]
    nodes_visited = 0
    
    while open_set:
        check_interrupt()
        current = heapq.heappop(open_set)[1]
        
        if perma_visited[current[0]][current[1]]:
            continue
            
        draw_grid_with_visited_complex(screen, grid, visited, perma_visited, grid_size, display_state)
        
        # Goal reached - reconstruct path
        if current == end:
            path = [end]
            while current != start:
                path.append(current)
                current = previous[current]
            path.append(start)
            return path[::-1], nodes_visited
        
        perma_visited[current[0]][current[1]] = True
        nodes_visited += 1
        
        # Explore neighbors
        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + graph[current][neighbor]
            visited[neighbor[0]][neighbor[1]] = True
            
            if tentative_g_score < g_score[neighbor]:
                previous[neighbor] = current
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        draw_grid_with_visited_complex(screen, grid, visited, perma_visited, grid_size, display_state)
    
    return [], nodes_visited


def dijkstra(graph, grid, start, end, screen, grid_size, display_state):
    """
    Dijkstra's shortest path algorithm.
    
    Args:
        graph (dict): Graph representation from convert_grid_to_graph()
        grid (list[list[int]]): 2D grid for visualization
        start (tuple): Starting position (row, col)
        end (tuple): Goal position (row, col)
        screen (pygame.Surface): Display surface for visualization
        grid_size (int): Size of each grid cell in pixels
        display_state (DisplayState): Display configuration object
    
    Returns:
        tuple: (path, nodes_visited) where path is list of (row,col) tuples and 
               nodes_visited is count of explored nodes
               
    Algorithm:
        Guarantees shortest path by exploring nodes in order of increasing distance
        from start. Uses priority queue for efficient min-distance extraction.
    """
    shortest_dist = {node: float('inf') for node in graph}
    shortest_dist[start] = 0
    prev = {}
    pq = [(0, start)]  # Priority queue: (distance, node)
    
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    perma_visited = [[False for _ in range(cols)] for _ in range(rows)]
    nodes_visited = 0
    
    while pq:
        check_interrupt()
        current_distance, node = heapq.heappop(pq)
        
        visited[node[0]][node[1]] = True
        perma_visited[node[0]][node[1]] = True
        nodes_visited += 1
        draw_grid_with_visited_complex(screen, grid, visited, perma_visited, grid_size, display_state)
        
        # Goal reached - reconstruct path
        if node == end:
            path = [end]
            while node != start:
                path.append(prev[node])
                node = prev[node]
            return path[::-1], nodes_visited
        
        # Skip if we've found a better path already
        if current_distance > shortest_dist[node]:
            continue
        
        # Explore neighbors
        for neighbour in graph[node]:
            neighbour_distance, neighbour_node = graph[node][neighbour], neighbour
            visited[neighbour_node[0]][neighbour_node[1]] = True
            
            if shortest_dist[neighbour_node] > (current_distance + neighbour_distance):
                shortest_dist[neighbour_node] = current_distance + neighbour_distance
                prev[neighbour_node] = node
                heapq.heappush(pq, (shortest_dist[neighbour_node], neighbour_node))
        
        draw_grid_with_visited_complex(screen, grid, visited, perma_visited, grid_size, display_state)
    
    return [], nodes_visited


def bfs(grid, start, end, screen, grid_size, display_state):
    """
    Breadth-First Search pathfinding algorithm.
    
    Args:
        grid (list[list[int]]): 2D grid
        start (tuple): Starting position (row, col)
        end (tuple): Goal position (row, col)
        screen (pygame.Surface): Display surface for visualization
        grid_size (int): Size of each grid cell in pixels
        display_state (DisplayState): Display configuration object
    
    Returns:
        tuple: (path, nodes_visited) where path is list of (row,col) tuples and 
               nodes_visited is count of explored nodes
               
    Algorithm:
        Explores nodes level by level, guaranteeing shortest path in unweighted graphs.
        Uses queue (FIFO) data structure for exploration order.
    """
    moves = [[1,0], [-1,0], [0,-1], [0,1]]
    queue = deque([(start, [start])])
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    nodes_visited = 0
    
    while queue:
        check_interrupt()
        (x, y), path = queue.popleft()
        visited[x][y] = True
        nodes_visited += 1
        draw_grid_with_visited(screen, grid, visited, grid_size, display_state)
        
        # Goal reached
        if (x, y) == end:
            return path, nodes_visited
        
        # Explore neighbors
        for move in moves:
            new_x, new_y = x + move[0], y + move[1]
            if valid(grid, new_x, new_y) and not visited[new_x][new_y]:
                queue.append(((new_x, new_y), path + [(new_x, new_y)]))
                visited[new_x][new_y] = True
    
    return [], nodes_visited


def dfs(grid, start, end, screen, grid_size, display_state):
    """
    Depth-First Search pathfinding algorithm (iterative implementation).
    
    Args:
        grid (list[list[int]]): 2D grid
        start (tuple): Starting position (row, col)
        end (tuple): Goal position (row, col)
        screen (pygame.Surface): Display surface for visualization
        grid_size (int): Size of each grid cell in pixels
        display_state (DisplayState): Display configuration object
    
    Returns:
        tuple: (path, nodes_visited) where path is list of (row,col) tuples and 
               nodes_visited is count of explored nodes
               
    Algorithm:
        Explores as far as possible along each branch before backtracking.
        Uses stack (LIFO) data structure. Iterative implementation avoids 
        Python recursion limit issues.
        
    Notes:
        Does not guarantee shortest path, but useful for maze solving and 
        exploring all possibilities.
    """
    if not grid or not grid[0]:
        return [], 0
    
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    stack = [(start, [start])]
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    nodes_visited = 0
    
    while stack:
        check_interrupt()
        (x, y), path = stack.pop()
        
        if visited[x][y]:
            continue
            
        visited[x][y] = True
        nodes_visited += 1
        draw_grid_with_visited(screen, grid, visited, grid_size, display_state)
        
        # Goal reached
        if (x, y) == end:
            return path, nodes_visited
        
        # Explore neighbors (added in reverse to maintain consistent direction)
        for move in moves:
            new_x, new_y = x + move[0], y + move[1]
            if valid(grid, new_x, new_y) and not visited[new_x][new_y]:
                stack.append(((new_x, new_y), path + [(new_x, new_y)]))
    
    return [], nodes_visited


def backend_main(input_grid, path_algo, screen, grid_size, start, end, display_state):
    """
    Main backend function to execute selected pathfinding algorithm.
    
    Args:
        input_grid (list[list[int]]): The 2D grid
        path_algo (int): Algorithm selection (1=DFS, 2=BFS, 3=Dijkstra, 4=A*)
        screen (pygame.Surface): Display surface
        grid_size (int): Size of each grid cell in pixels
        start (tuple): Starting position (row, col)
        end (tuple): Goal position (row, col)
        display_state (DisplayState): Display configuration object
    
    Returns:
        tuple: (path, nodes_visited, elapsed_time, algo_name) or 
               ("interrupted", 0, 0, "") if user interrupts or
               (None, nodes, elapsed, algo_name) if no path found
               
    Side Effects:
        - Updates global currently_running variable
        - May display popup messages for errors
    """
    global currently_running
    algo_names = {1: "DFS", 2: "BFS", 3: "Dijkstra", 4: "A*"}
    currently_running = algo_names[path_algo]
    start_time = time.time()
    
    try:
        if path_algo == 1:
            return_val, nodes = dfs(input_grid, start, end, screen, grid_size, display_state)
        elif path_algo == 2:
            return_val, nodes = bfs(input_grid, start, end, screen, grid_size, display_state)
        elif path_algo == 3:
            input_graph = convert_grid_to_graph(input_grid)
            return_val, nodes = dijkstra(input_graph, input_grid, start, end, screen, grid_size, display_state)
        elif path_algo == 4:
            input_graph = convert_grid_to_graph(input_grid)
            return_val, nodes = astar(input_graph, start, end, screen, input_grid, grid_size, display_state)
        
        currently_running = ""
        end_time = time.time()
        elapsed = end_time - start_time
        
        if return_val:
            return return_val, nodes, elapsed, algo_names[path_algo]
        else:
            show_popup(screen, "No Path Found", 
                      f"{algo_names[path_algo]} could not find a path!", display_state)
            return None, nodes, elapsed, algo_names[path_algo]
            
    except AlgorithmInterrupted as e:
        currently_running = ""
        return "interrupted", 0, 0, ""
    except RecursionError:
        currently_running = ""
        show_popup(screen, "Stack Overflow Error", 
                  "DFS recursion limit exceeded! Try a smaller maze.", display_state)
        return None, 0, 0, ""


# ========================= PYGAME INITIALIZATION =========================

pygame.init()
pygame.mixer.init()

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
ORANGE = (204, 85, 0) 
LIGHT_ORANGE = (255, 200, 100)
GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)
DARK_RED = (139, 0, 0)

# Get screen dimensions for adaptive sizing
display_info = pygame.display.Info()
SCREEN_WIDTH = display_info.current_w
SCREEN_HEIGHT = display_info.current_h

# Grid dimensions
ROWS, COLS = 50, 50


# ========================= DISPLAY STATE CLASS =========================

class DisplayState:
    """
    Encapsulates all display-related state that changes with window resizing.
    
    Attributes:
        window_width (int): Current window width in pixels
        window_height (int): Current window height in pixels
        info_box_height (int): Height of info box (25% of window height)
        grid_height (int): Height available for grid display
        grid_width (int): Width available for grid display
        grid_size (int): Size of each grid cell in pixels
        actual_grid_width (int): Actual width occupied by grid
        actual_grid_height (int): Actual height occupied by grid
        title_font_size (int): Calculated font size for titles
        text_font_size (int): Calculated font size for regular text
        small_font_size (int): Calculated font size for small text
        title_font (pygame.font.Font): Font object for titles
        text_font (pygame.font.Font): Font object for regular text
        small_font (pygame.font.Font): Font object for small text
    """
    
    def __init__(self, width, height):
        """
        Initialize display state for given window dimensions.
        
        Args:
            width (int): Window width in pixels
            height (int): Window height in pixels
        """
        self.window_width = width
        self.window_height = height
        self.info_box_height = int(height * 0.25)  # 25% of window height
        self.grid_height = height - self.info_box_height
        self.grid_width = width
        self.grid_size = min(self.grid_width // COLS, self.grid_height // ROWS)
        
        # Recalculate actual grid display area
        self.actual_grid_width = self.grid_size * COLS
        self.actual_grid_height = self.grid_size * ROWS
        
        # Font sizes based on window size
        self.title_font_size = max(14, int(width * 0.018))
        self.text_font_size = max(12, int(width * 0.015))
        self.small_font_size = max(10, int(width * 0.013))
        
        # Update fonts
        self.title_font = pygame.font.SysFont("Arial", self.title_font_size, bold=True)
        self.text_font = pygame.font.SysFont("Arial", self.text_font_size)
        self.small_font = pygame.font.SysFont("Arial", self.small_font_size)


# Initialize display state with default windowed size
DEFAULT_WIDTH = 700
DEFAULT_HEIGHT = 900
display_state = DisplayState(DEFAULT_WIDTH, DEFAULT_HEIGHT)

# Setup screen
screen = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Path Finding Visualizer")

# Global state variables
start_pos = [0, 0]
end_pos = [ROWS-1, COLS-1]
setting_start = False
setting_end = False
last_stats = {"algo": "", "path_length": 0, "nodes_visited": 0, "time": 0}
currently_running = ""


# ========================= UI FUNCTIONS =========================

def draw_text(text, font, color, x, y):
    """
    Render and draw text on screen.
    
    Args:
        text (str): Text to display
        font (pygame.font.Font): Font to use
        color (tuple): RGB color tuple
        x (int): X coordinate
        y (int): Y coordinate
    """
    img = font.render(text, True, color)
    screen.blit(img, (x, y))


def show_popup(screen, title, message, display_state):
    """
    Display a modal popup message that waits for user input.
    
    Args:
        screen (pygame.Surface): Display surface
        title (str): Popup title text
        message (str): Popup message text
        display_state (DisplayState): Display configuration object
        
    Notes:
        Blocks until user presses any key or mouse button
    """
    popup_width = int(display_state.window_width * 0.4)
    popup_height = int(display_state.window_height * 0.15)
    popup_x = (display_state.window_width - popup_width) // 2
    popup_y = (display_state.grid_height - popup_height) // 2
    
    # Draw semi-transparent background
    s = pygame.Surface((display_state.window_width, display_state.grid_height))
    s.set_alpha(128)
    s.fill(BLACK)
    screen.blit(s, (0, 0))
    
    # Draw popup box
    pygame.draw.rect(screen, WHITE, (popup_x, popup_y, popup_width, popup_height))
    pygame.draw.rect(screen, RED, (popup_x, popup_y, popup_width, popup_height), 3)
    
    # Draw text
    draw_text(title, display_state.title_font, RED, popup_x + 20, popup_y + 20)
    draw_text(message, display_state.text_font, BLACK, popup_x + 20, popup_y + 60)
    draw_text("Press any key to continue...", display_state.small_font, GRAY, 
             popup_x + 20, popup_y + popup_height - 30)
    
    pygame.display.flip()
    
    # Wait for key press
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                waiting = False


def draw_info_box(display_state):
    """
    Draw the information panel at bottom of screen with controls, legend, and stats.
    
    Args:
        display_state (DisplayState): Display configuration object
        
    Displays:
        - Title with currently running algorithm
        - Color legend
        - Controls reference
        - Algorithm selection keys
        - Performance statistics from last run
        - Author credit
    """
    info_y_start = display_state.grid_height
    pygame.draw.rect(screen, GRAY, (0, info_y_start, display_state.window_width, 
                                    display_state.info_box_height))
    pygame.draw.rect(screen, WHITE, (0, info_y_start, display_state.window_width, 
                                    display_state.info_box_height), 2)
    
    margin = int(display_state.window_width * 0.015)
    
    # Title with running algorithm indicator
    title_text = "Path Finding Visualizer"
    if currently_running:
        title_text += f" - Running: {currently_running}..."
    draw_text(title_text, display_state.title_font, WHITE, margin, info_y_start + 5)
    
    # Color Legend
    legend_y = info_y_start + int(display_state.info_box_height * 0.15)
    draw_text("Color Legend:", display_state.text_font, WHITE, margin, legend_y)
    
    # Draw color samples
    color_box_size = max(12, int(display_state.window_width * 0.015))
    col1_x = margin
    col2_x = int(display_state.window_width * 0.2)
    
    colors = [
        (GREEN, "Start", col1_x, legend_y + 25),
        (RED, "End", col1_x, legend_y + 50),
        (BLACK, "Path", col1_x, legend_y + 75),
        (WHITE, "Wall", col2_x, legend_y + 25),
        (ORANGE, "Visited", col2_x, legend_y + 50),
        (LIGHT_ORANGE, "Exploring", col2_x, legend_y + 75),
    ]
    
    for color, label, x, y in colors:
        pygame.draw.rect(screen, color, (x, y, color_box_size, color_box_size))
        pygame.draw.rect(screen, WHITE, (x, y, color_box_size, color_box_size), 1)
        draw_text(label, display_state.small_font, LIGHT_GRAY, x + color_box_size + 5, y)
    
    # Controls
    controls_x = int(display_state.window_width * 0.38)
    draw_text("Controls:", display_state.text_font, WHITE, controls_x, legend_y)
    controls = [
        "• Drag: Draw walls",
        "• S: Set start point",
        "• E: Set end point",
        "• C: Clear path only",
        "• H: Hard clear (all)",
        "• G: Generate maze",
        "• F: Fullscreen",
    ]
    line_spacing = int(display_state.info_box_height * 0.1)
    for i, control in enumerate(controls):
        draw_text(control, display_state.small_font, LIGHT_GRAY, controls_x, 
                 legend_y + 25 + i * line_spacing)
    
    # Algorithms
    algo_x = int(display_state.window_width * 0.68)
    draw_text("Algorithms:", display_state.text_font, WHITE, algo_x, legend_y)
    algos = [
        "• 1: DFS",
        "• 2: BFS",
        "• 3: Dijkstra",
        "• 4: A* Search",
    ]
    for i, algo in enumerate(algos):
        draw_text(algo, display_state.small_font, LIGHT_GRAY, algo_x, 
                 legend_y + 25 + i * line_spacing)
    
    # Statistics (split into two lines to avoid overlap)
    if last_stats["algo"]:
        stats_y = info_y_start + display_state.info_box_height - 45
        draw_text(f"Last Run: {last_stats['algo']} | Path Length: {last_stats['path_length']}",
                 display_state.small_font, WHITE, margin, stats_y)
        draw_text(f"Nodes Visited: {last_stats['nodes_visited']} | Time: {last_stats['time']:.3f}s",
                 display_state.small_font, WHITE, margin, stats_y + 15)
    
    # Footer
    draw_text("Created by TRK Aashish", display_state.small_font, LIGHT_GRAY, margin,
             info_y_start + display_state.info_box_height - 15)


# ========================= MAZE GENERATION =========================

def is_valid(grid, x, y):
    """
    Check if grid position is within bounds.
    
    Args:
        grid (list[list[int]]): The 2D grid
        x (int): Row coordinate
        y (int): Column coordinate
    
    Returns:
        bool: True if position is within grid bounds
    """
    return 0 <= x < len(grid) and 0 <= y < len(grid[0])


def recursive_backtracker(grid, x, y):
    """
    Recursive backtracking algorithm for maze generation.
    
    Args:
        grid (list[list[int]]): The 2D grid to modify
        x (int): Current row position
        y (int): Current column position
        
    Algorithm:
        Carves paths through the grid by randomly selecting unvisited neighbors
        and recursively visiting them, creating a perfect maze (no loops).
    """
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    random.shuffle(directions)
    
    for dx, dy in directions:
        next_x, next_y = x + 2 * dx, y + 2 * dy
        if is_valid(grid, next_x, next_y) and grid[next_x][next_y] == -1:
            grid[x + dx][y + dy] = 0
            grid[next_x][next_y] = 0
            recursive_backtracker(grid, next_x, next_y)


def generate_maze(grid, start_pos, end_pos):
    """
    Generate a random maze using recursive backtracker algorithm.
    
    Args:
        grid (list[list[int]]): The 2D grid to fill with maze
        start_pos (list): [row, col] of start position
        end_pos (list): [row, col] of end position
    
    Returns:
        list[list[int]]: Modified grid with generated maze
        
    Notes:
        Ensures start and end positions have clear paths by clearing
        a 3x3 area around each position to prevent them being walled in.
    """
    start_x, start_y = 1, 1
    grid[start_x][start_y] = 0
    recursive_backtracker(grid, start_x, start_y)
    
    # Ensure start and end positions are clear
    grid[start_pos[0]][start_pos[1]] = 3
    grid[end_pos[0]][end_pos[1]] = 4
    
    # Clear a path around start to prevent it being walled in
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = start_pos[0] + dx, start_pos[1] + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                if grid[nx][ny] == -1:
                    grid[nx][ny] = 0
    
    # Clear a path around end to prevent it being walled in
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = end_pos[0] + dx, end_pos[1] + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                if grid[nx][ny] == -1:
                    grid[nx][ny] = 0
    
    # Restore start and end markers
    grid[start_pos[0]][start_pos[1]] = 3
    grid[end_pos[0]][end_pos[1]] = 4
    
    return grid


# ========================= DRAWING FUNCTIONS =========================

def draw_grid(grid, start_pos, end_pos, display_state):
    """
    Draw the main pathfinding grid.
    
    Args:
        grid (list[list[int]]): The 2D grid to draw
        start_pos (list): [row, col] of start position
        end_pos (list): [row, col] of end position
        display_state (DisplayState): Display configuration object
        
    Color Mapping:
        Green: Start position
        Red: End position
        Black: Walkable paths
        White: Walls
        Dark Red: Computed path
    """
    for x in range(ROWS):
        for y in range(COLS):
            color = WHITE  # Default wall
            
            if (x, y) == (start_pos[0], start_pos[1]):
                color = GREEN
            elif (x, y) == (end_pos[0], end_pos[1]):
                color = RED
            elif grid[x][y] == 0 or grid[x][y] == 3 or grid[x][y] == 4: 
                color = BLACK 
            elif grid[x][y] == -1: 
                color = WHITE
            elif grid[x][y] == 2:
                color = DARK_RED
                
            pygame.draw.rect(screen, color, 
                           (y * display_state.grid_size, x * display_state.grid_size, 
                            display_state.grid_size, display_state.grid_size))
            pygame.draw.rect(screen, LIGHT_GRAY, 
                           (y * display_state.grid_size, x * display_state.grid_size, 
                            display_state.grid_size, display_state.grid_size), 1)


def update_grid_with_path(grid, sets_list):
    """
    Mark the final computed path on the grid.
    
    Args:
        grid (list[list[int]]): The 2D grid to modify
        sets_list (list): List of (row, col) tuples forming the path
        
    Notes:
        Does not overwrite start (3) or end (4) markers
    """
    if sets_list:
        for s in sets_list:
            row, col = s
            if grid[row][col] != 3 and grid[row][col] != 4:
                grid[row][col] = 2


def draw_grid_with_visited(screen, grid, visited, grid_size, display_state):
    """
    Draw grid with visited cells highlighted (for BFS/DFS visualization).
    
    Args:
        screen (pygame.Surface): Display surface
        grid (list[list[int]]): The 2D grid
        visited (list[list[bool]]): 2D array tracking visited cells
        grid_size (int): Size of each grid cell in pixels
        display_state (DisplayState): Display configuration object
        
    Color Mapping:
        Orange: Visited cells
        (Plus all colors from draw_grid)
    """
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            color = WHITE
            
            if (x, y) == (start_pos[0], start_pos[1]):
                color = GREEN
            elif (x, y) == (end_pos[0], end_pos[1]):
                color = RED
            elif visited[x][y]: 
                color = ORANGE
            elif grid[x][y] == 0 or grid[x][y] == 3 or grid[x][y] == 4: 
                color = BLACK 
            elif grid[x][y] == -1: 
                color = WHITE
                
            pygame.draw.rect(screen, color, 
                           (y * grid_size, x * grid_size, grid_size, grid_size))
            pygame.draw.rect(screen, LIGHT_GRAY, 
                           (y * grid_size, x * grid_size, grid_size, grid_size), 1)
    draw_info_box(display_state)
    pygame.display.flip()


def draw_grid_with_visited_complex(screen, grid, visited, perma_visited, grid_size, display_state):
    """
    Draw grid with two levels of visited cells (for Dijkstra/A* visualization).
    
    Args:
        screen (pygame.Surface): Display surface
        grid (list[list[int]]): The 2D grid
        visited (list[list[bool]]): 2D array tracking cells being considered
        perma_visited (list[list[bool]]): 2D array tracking fully processed cells
        grid_size (int): Size of each grid cell in pixels
        display_state (DisplayState): Display configuration object
        
    Color Mapping:
        Orange: Permanently visited/processed cells
        Light Orange: Cells being explored/in open set
        (Plus all colors from draw_grid)
    """
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            color = WHITE
            
            if (x, y) == (start_pos[0], start_pos[1]):
                color = GREEN
            elif (x, y) == (end_pos[0], end_pos[1]):
                color = RED
            elif perma_visited[x][y]: 
                color = ORANGE
            elif visited[x][y]:
                color = LIGHT_ORANGE
            elif grid[x][y] == 0 or grid[x][y] == 3 or grid[x][y] == 4: 
                color = BLACK 
            elif grid[x][y] == -1: 
                color = WHITE
                
            pygame.draw.rect(screen, color, 
                           (y * grid_size, x * grid_size, grid_size, grid_size))
            pygame.draw.rect(screen, LIGHT_GRAY, 
                           (y * grid_size, x * grid_size, grid_size, grid_size), 1)
    draw_info_box(display_state)
    pygame.display.flip()


# ========================= GRID MANIPULATION =========================

def toggle_obstacle(grid, row, col, start_pos, end_pos):
    """
    Toggle wall at given position (only if not start/end).
    
    Args:
        grid (list[list[int]]): The 2D grid
        row (int): Row coordinate
        col (int): Column coordinate
        start_pos (list): [row, col] of start position
        end_pos (list): [row, col] of end position
        
    Notes:
        Does not allow walls to be placed on start or end positions
    """
    if (row, col) == (start_pos[0], start_pos[1]) or (row, col) == (end_pos[0], end_pos[1]):
        return
    if grid[row][col] == 0:
        grid[row][col] = -1


def clean_grid(grid):
    """
    Clear only path cells (2), preserving walls and start/end markers.
    
    Args:
        grid (list[list[int]]): The 2D grid to clean
    """
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 2:
                grid[i][j] = 0


def hard_clean_grid(grid, start_pos, end_pos):
    """
    Clear everything and reset to blank grid with only start/end markers.
    
    Args:
        grid (list[list[int]]): The 2D grid to reset
        start_pos (list): [row, col] of start position
        end_pos (list): [row, col] of end position
    """
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            grid[i][j] = 0
    grid[start_pos[0]][start_pos[1]] = 3
    grid[end_pos[0]][end_pos[1]] = 4


def clear_visited_colors(grid):
    """
    Clear visited visualization colors before running new algorithm.
    
    Args:
        grid (list[list[int]]): The 2D grid to clean
        
    Notes:
        Calls clean_grid() to remove path markers (2)
    """
    clean_grid(grid)


# ========================= MAIN PROGRAM =========================

# Initialize grid with start and end markers
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
grid[start_pos[0]][start_pos[1]] = 3
grid[end_pos[0]][end_pos[1]] = 4

# Main loop state
mouse_pressed = False 
running = True
fullscreen = False

# Main game loop
while running:
    screen.fill(BLACK)
    draw_grid(grid, start_pos, end_pos, display_state)
    draw_info_box(display_state)
    
    # Show instruction for setting start/end
    if setting_start:
        draw_text("Click to set START position", display_state.title_font, GREEN, 
                 display_state.window_width // 4, 10)
    elif setting_end:
        draw_text("Click to set END position", display_state.title_font, RED, 
                 display_state.window_width // 4, 10)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.VIDEORESIZE:
            # Handle window resize - recalculate display state
            display_state = DisplayState(event.w, event.h)
            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if setting_start or setting_end:
                mouse_pos = pygame.mouse.get_pos()
                col = mouse_pos[0] // display_state.grid_size
                row = mouse_pos[1] // display_state.grid_size
                if row < ROWS and col < COLS:
                    if setting_start:
                        grid[start_pos[0]][start_pos[1]] = 0
                        start_pos[0], start_pos[1] = row, col
                        grid[row][col] = 3
                        setting_start = False
                    elif setting_end:
                        grid[end_pos[0]][end_pos[1]] = 0
                        end_pos[0], end_pos[1] = row, col
                        grid[row][col] = 4
                        setting_end = False
            else:
                mouse_pressed = True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_pressed = False
            
        elif event.type == pygame.MOUSEMOTION and mouse_pressed:
            if not setting_start and not setting_end:
                mouse_pos = pygame.mouse.get_pos()
                col = mouse_pos[0] // display_state.grid_size
                row = mouse_pos[1] // display_state.grid_size
                if row < ROWS and col < COLS and mouse_pos[1] < display_state.grid_height:
                    toggle_obstacle(grid, row, col, start_pos, end_pos)
                    
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                setting_start = True
                setting_end = False
                
            elif event.key == pygame.K_e:
                setting_end = True
                setting_start = False
                
            elif event.key == pygame.K_1:
                clear_visited_colors(grid)
                result = backend_main(grid, 1, screen, display_state.grid_size, 
                                    tuple(start_pos), tuple(end_pos), display_state)
                if result and result[0] != "interrupted":
                    path_list, nodes, elapsed, algo = result
                    if path_list:
                        update_grid_with_path(grid, path_list)
                        last_stats = {"algo": algo, "path_length": len(path_list), 
                                    "nodes_visited": nodes, "time": elapsed}
                elif result and result[0] == "interrupted":
                    clean_grid(grid)
                    
            elif event.key == pygame.K_2:
                clear_visited_colors(grid)
                result = backend_main(grid, 2, screen, display_state.grid_size, 
                                    tuple(start_pos), tuple(end_pos), display_state)
                if result and result[0] != "interrupted":
                    path_list, nodes, elapsed, algo = result
                    if path_list:
                        update_grid_with_path(grid, path_list)
                        last_stats = {"algo": algo, "path_length": len(path_list), 
                                    "nodes_visited": nodes, "time": elapsed}
                elif result and result[0] == "interrupted":
                    clean_grid(grid)
                    
            elif event.key == pygame.K_3:
                clear_visited_colors(grid)
                result = backend_main(grid, 3, screen, display_state.grid_size, 
                                    tuple(start_pos), tuple(end_pos), display_state)
                if result and result[0] != "interrupted":
                    path_list, nodes, elapsed, algo = result
                    if path_list:
                        update_grid_with_path(grid, path_list)
                        last_stats = {"algo": algo, "path_length": len(path_list), 
                                    "nodes_visited": nodes, "time": elapsed}
                elif result and result[0] == "interrupted":
                    clean_grid(grid)
                    
            elif event.key == pygame.K_4:
                clear_visited_colors(grid)
                result = backend_main(grid, 4, screen, display_state.grid_size, 
                                    tuple(start_pos), tuple(end_pos), display_state)
                if result and result[0] != "interrupted":
                    path_list, nodes, elapsed, algo = result
                    if path_list:
                        update_grid_with_path(grid, path_list)
                        last_stats = {"algo": algo, "path_length": len(path_list), 
                                    "nodes_visited": nodes, "time": elapsed}
                elif result and result[0] == "interrupted":
                    clean_grid(grid)
                    
            elif event.key == pygame.K_c: 
                clean_grid(grid)
                setting_start = False
                setting_end = False
                
            elif event.key == pygame.K_h:
                hard_clean_grid(grid, start_pos, end_pos)
                last_stats = {"algo": "", "path_length": 0, "nodes_visited": 0, "time": 0}
                setting_start = False
                setting_end = False
                
            elif event.key == pygame.K_f:
                fullscreen = not fullscreen
                if fullscreen:
                    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
                    display_state = DisplayState(SCREEN_WIDTH, SCREEN_HEIGHT)
                else:
                    screen = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE)
                    display_state = DisplayState(DEFAULT_WIDTH, DEFAULT_HEIGHT)
                    
            elif event.key == pygame.K_g:
                grid = [[-1 for _ in range(COLS)] for _ in range(ROWS)]
                generate_maze(grid, start_pos, end_pos)
    
    pygame.display.flip()

pygame.quit()