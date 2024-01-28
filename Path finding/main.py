from collections import deque
from collections import defaultdict
import pygame
import heapq
import random

def convert_grid_to_graph(grid):
    moves=[[1,0],[-1,0],[0,-1],[0,1]]
    graph=defaultdict(dict)
    rows=len(grid)
    cols=len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if grid[i][j]==0:
                for move in moves:
                    new_i,new_j=i+move[0],j+move[1]
                    if valid (grid,new_i,new_j) and grid[new_i][new_j]==0 :
                        graph[(i,j)][(new_i,new_j)]=1
    return graph



def valid(grid,x,y):
    if 0<=x<len(grid) and 0<=y<len(grid[0]) and grid[x][y]==0:
        return True
    else:
        return False
    
def heuristic(node, end):
    return ((node[0] - end[0])**2 + (node[1] - end[1])**2) ** 0.5

def astar(graph, start, end):
    open_set = [(0, start)] 
    previous = {}
    g_score = {node: float('inf') for node in graph} 
    g_score[start] = 0
    f_score = {} 
    for node in graph:
        f_score[node]=heuristic(node,end)
    rows, cols = len(grid), len(grid[0])
    visited=[[False for _ in range(cols)] for _ in range(rows)]
    perma_visited=[[False for _ in range(cols)] for _ in range(rows)]
    while open_set:
        current = heapq.heappop(open_set)[1]
        if perma_visited[current[0]][current[1]]==True:
            continue
        draw_grid_with_visited_complex(screen, grid, visited,perma_visited, GRID_SIZE)
        if current == end:
            path = [end]
            while current!=start:
                path.append(current)
                current = previous[current]
            path.append(start)
            return path[::-1]  
        perma_visited[current[0]][current[1]]=True  
        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + graph[current][neighbor]
            visited[neighbor[0]][neighbor[1]]=True
            if tentative_g_score < g_score[neighbor]:
                previous[neighbor] = current
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
        draw_grid_with_visited_complex(screen, grid, visited,perma_visited, GRID_SIZE)
    return []



def dijkstra(graph, grid,start, end):
    shortest_dist = {node: float('inf') for node in graph}
    shortest_dist[start] = 0
    prev = {}
    pq = [(0, start)]
    rows, cols = len(grid), len(grid[0])
    visited=[[False for _ in range(cols)] for _ in range(rows)]
    perma_visited=[[False for _ in range(cols)] for _ in range(rows)]
    while pq:
        current_distance, node = heapq.heappop(pq)
        visited[node[0]][node[1]]=True
        perma_visited[node[0]][node[1]]=True
        draw_grid_with_visited_complex(screen, grid,visited, perma_visited, GRID_SIZE)
        if node == end:
            path = [end]
            while node != start:
                path.append(prev[node])
                node = prev[node]
            return path[::-1]
        if current_distance > shortest_dist[node]:
            continue
        for neighbour in graph[node]:
            neighbour_distance, neighbour_node = graph[node][neighbour], neighbour
            visited[neighbour_node[0]][neighbour_node[1]]=True
            if shortest_dist[neighbour_node] > (current_distance+neighbour_distance):
                shortest_dist[neighbour_node] = current_distance + \
                    neighbour_distance
                prev[neighbour_node] = node
                heapq.heappush(
                    pq, (shortest_dist[neighbour_node], neighbour_node))
        draw_grid_with_visited_complex(screen, grid, visited ,perma_visited, GRID_SIZE)
    return []


def bfs(grid,start,end,screen,GRID_SIZE):
    moves=[[1,0],[-1,0],[0,-1],[0,1]]
    queue=deque([(start,[start])])
    rows, cols = len(grid), len(grid[0])
    visited=[[False for _ in range(cols)] for _ in range(rows)]
    while queue:
        (x,y),path=queue.popleft()
        visited[x][y]=True
        draw_grid_with_visited(screen, grid, visited, GRID_SIZE)
        if (x,y)==end:
            return path
        for move in moves:
            new_x,new_y=x+move[0],y+move[1]
            if valid(grid,new_x,new_y) and not visited[new_x][new_y]:
                queue.append(((new_x, new_y), path + [(new_x, new_y)]))
                visited[new_x][new_y]=True
    return []




def dfs_rec_fn(grid,start,end,visited,path,screen,GRID_SIZE):
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if start == end:
        path.append(start)
        return True
    x, y = start
    visited[x][y] = True
    draw_grid_with_visited(screen, grid, visited, GRID_SIZE)
    path.append((x, y))
    for move in moves:
        new_x, new_y = x + move[0], y + move[1]
        if valid(grid, new_x, new_y) and not visited[new_x][new_y]:
            if dfs_rec_fn(grid, (new_x, new_y), end, visited, path,screen,GRID_SIZE):
                return True
    path.pop()
    return False

def dfs(grid,start,end,screen,GRID_SIZE):
    if not grid or not grid[0]:
        return []
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    path = []
    if dfs_rec_fn(grid, start, end, visited, path,screen,GRID_SIZE):
        return path
    else:
        return []

def backend_main(input_grid,path_algo,screen,GRID_SIZE,start,end):
    if path_algo==1:
        return_val = dfs(input_grid,start,end,screen,GRID_SIZE)
        if return_val:
            return return_val
        else:
            print("not possible")
            return 
    elif path_algo==2:
        return_val=bfs(input_grid,start,end,screen,GRID_SIZE)
        if return_val:
            return return_val
        else:
            print("not possible")
            return 
    elif path_algo==3:
        input_graph=convert_grid_to_graph(input_grid)
        return_val=dijkstra(input_graph,grid,start,end)
        if return_val:
            return return_val
        else:
            print("not possible")
            return 
    elif path_algo==4:
        input_graph=convert_grid_to_graph(input_grid)
        return_val=astar(input_graph,start,end)
        if return_val:
            return return_val
        else:
            print("not possible")
            return 











































pygame.init()
pygame.mixer.init()
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (204, 85, 0) 
LIGHT_ORANGE = (255, 200, 100)

pygame.display.set_caption("Welcome")
screen = pygame.display.set_mode((1000, 700))
running =  True
text_font=pygame.font.SysFont("Arial",30)

def draw_text(text,font,color,x,y):
        img=font.render(text,True,color)
        screen.blit(img,(x,y))

# popup_text = '''
line1="Welcome to Path Finder!!"
line2="Instructions :-"

line3="    1) Use (click and drag) mouse to draw walls."

line4="    2) Press h (hard clear) to clear the screen completely."

line5="    3) Press c (clear) to clear only the path and not the walls."

line6="    4) Press g to generate a random maze."

line7="    5) After drawing the path press from 1-4 for :-"

line8="        1 - dfs search"
line9="        2 - bfs search"
line10="       3 - Dijkstra search"
line11="       4 - A* search"

line12="Created by - TRK Aashish "
linex="press s to begin"
while running:
    screen.fill((255,255,255))
    draw_text(line1,text_font,(0,0,0),0,0)
    draw_text(line2,text_font,(0,0,0),0,50)
    draw_text(line3,text_font,(0,0,0),0,100)
    draw_text(line4,text_font,(0,0,0),0,150)
    draw_text(line5,text_font,(0,0,0),0,200)
    draw_text(line6,text_font,(0,0,0),0,250)
    draw_text(line7,text_font,(0,0,0),0,300)
    draw_text(line8,text_font,(0,0,0),0,350)
    draw_text(line9,text_font,(0,0,0),0,400)
    draw_text(line10,text_font,(0,0,0),0,450)
    draw_text(line11,text_font,(0,0,0),0,500)
    draw_text(line12,text_font,(0,0,0),0,550)
    draw_text(linex,text_font,(0,0,0),0,600)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                running = False

    
    pygame.display.flip()

pygame.quit()

WIDTH, HEIGHT = 700, 700 
ROWS, COLS = 50, 50  
GRID_SIZE = WIDTH // COLS




screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Path Finding Visualizer")

def is_valid(grid, x, y):
    return 0 <= x < len(grid) and 0 <= y < len(grid[0])

def recursive_backtracker(grid, x, y):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    random.shuffle(directions)

    for dx, dy in directions:
        next_x, next_y = x + 2 * dx, y + 2 * dy

        if is_valid(grid, next_x, next_y) and grid[next_x][next_y] == -1:
            grid[x + dx][y + dy] = 0
            grid[next_x][next_y] = 0
            recursive_backtracker(grid, next_x, next_y)

def generate_maze(grid):
    start_x, start_y = 1, 1

    grid[start_x][start_y] = 0
    recursive_backtracker(grid, start_x, start_y)

    return grid


def draw_grid():
    for x in range(ROWS):
        for y in range(COLS):
            if grid[x][y]==0: 
                color = BLACK 
            elif grid[x][y]==-1: 
                color=WHITE
            elif grid[x][y]==2:
                color=RED
            pygame.draw.rect(screen, color, (y * GRID_SIZE, x * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(screen, WHITE, (y * GRID_SIZE, x * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)
    

def update_gird_with_path(grid, sets_list):
    if sets_list:
        for s in sets_list:
            row, col =s
            grid[row][col] = 2
    else:
        pass

def draw_grid_with_visited(screen, grid, visited, GRID_SIZE):
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if visited[x][y]: 
                color = ORANGE
            elif grid[x][y]==0: 
                color = BLACK 
            elif grid[x][y]==-1: 
                color=WHITE
            pygame.draw.rect(screen, color, (y * GRID_SIZE, x * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(screen, WHITE, (y * GRID_SIZE, x * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)
    pygame.display.flip()

def draw_grid_with_visited_complex(screen, grid, visited,perma_visited, GRID_SIZE):
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if perma_visited[x][y]: 
                color = ORANGE
            elif visited[x][y]:
                color= LIGHT_ORANGE
            elif grid[x][y]==0: 
                color = BLACK 
            elif grid[x][y]==-1: 
                color=WHITE
            pygame.draw.rect(screen, color, (y * GRID_SIZE, x * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(screen, WHITE, (y * GRID_SIZE, x * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)
    pygame.display.flip()




def toggle_obstacle(row, col):
    if grid[row][col]==0:
        grid[row][col] = -1


def clean_grid(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j]==2:
                grid[i][j]=0



def hard_clean_grid(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            grid[i][j]=0


grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]

mouse_pressed = False 
running = True
fullscreen = True
while running:
    screen.fill(BLACK)
    draw_grid()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_pressed = False
        elif event.type == pygame.MOUSEMOTION and mouse_pressed:
            mouse_pos=pygame.mouse.get_pos()
            col = mouse_pos[0] // GRID_SIZE
            row = mouse_pos[1] // GRID_SIZE
            toggle_obstacle(row, col)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1: 
                path_list = backend_main(grid,1,screen,GRID_SIZE,(0,0),(ROWS-1,COLS-1))
                update_gird_with_path(grid,path_list)
            elif event.key == pygame.K_2: 
                path_list = backend_main(grid,2,screen,GRID_SIZE,(0,0),(ROWS-1,COLS-1))
                update_gird_with_path(grid,path_list)
            elif event.key == pygame.K_3:  
                path_list = backend_main(grid,3,screen,GRID_SIZE,(0,0),(ROWS-1,COLS-1))
                update_gird_with_path(grid,path_list)
            elif event.key == pygame.K_4:  
                path_list = backend_main(grid,4,screen,GRID_SIZE,(0,0),(ROWS-1,COLS-1))
                update_gird_with_path(grid,path_list)
            elif event.key == pygame.K_c: 
                clean_grid(grid) 
            elif event.key == pygame.K_h:
                hard_clean_grid(grid)
            elif event.key == pygame.K_f:
                fullscreen = not fullscreen
                pygame.display.toggle_fullscreen()
            elif event.key == pygame.K_g:
                grid = [[-1 for _ in range(COLS)] for _ in range(ROWS)]
                generate_maze(grid)
                grid[0][0]=0
                grid[0][1]=0
    pygame.display.flip()
    if not running:
        pygame.quit()

