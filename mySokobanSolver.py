
'''

    Sokoban assignment


The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

You are NOT allowed to change the defined interfaces.
In other words, you must fully adhere to the specifications of the 
functions, their arguments and returned values.
Changing the interfacce of a function will likely result in a fail 
for the test of your code. This is not negotiable! 

You have to make sure that your code works with the files provided 
(search.py and sokoban.py) as your code will be tested 
with the original copies of these files. 

Last modified by 2021-08-17  by f.maire@qut.edu.au
- clarifiy some comments, rename some functions
  (and hopefully didn't introduce any bug!)

'''

# You have to make sure that your code works with 
# the files provided (search.py and sokoban.py) as your code will be tested 
# with these files
import search 
import sokoban

import functools
import operator

import sys
import time

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(10362380, "Hsiang-Ling", "Fan"), 
            (9854258, "Tan Chieu Duong", "Nguyen"), 
            (10178414, "Pei-Fang", "Shen")]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def taboo_cells(warehouse):
    '''
    Identify the taboo cells of a warehouse. A "taboo cell" is by definition
    a cell inside a warehouse such that whenever a box get pushed on such
    a cell then the puzzle becomes unsolvable.

    Cells outside the warehouse are not taboo. It is a fail to tag one as taboo.

    When determining the taboo cells, you must ignore all the existing boxes,
    only consider the walls and the target  cells.
    Use only the following rules to determine the taboo cells;
     Rule 1: if a cell is a corner and not a target, then it is a taboo cell.
     Rule 2: all the cells between two corners along a wall are taboo if none of
             these cells is a target.

    @param warehouse:
        a Warehouse object with a worker inside the warehouse

    @return
       A string representing the warehouse with only the wall cells marked with
       a '#' and the taboo cells marked with a 'X'.
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.
    '''

    X, Y = zip(*warehouse.walls)  # get walls from the warehouse text file 
    x_size, y_size = 1+max(X), 1+max(Y)
    taboo_cells = set()
    vis = [[" "] * x_size for y in range(y_size)]
    for (x, y) in warehouse.walls:
        vis[y][x] = "#"

    inside_cells = check_inside(warehouse.walls, warehouse.worker)

    # check corners
    for (i, j) in inside_cells:
        if (i, j) not in warehouse.targets:
            if (i+1, j) in warehouse.walls and (i, j+1) in warehouse.walls:
                taboo_cells.add((j, i))
            elif (i+1, j) in warehouse.walls and (i, j-1) in warehouse.walls:
                taboo_cells.add((j, i))
            elif (i-1, j) in warehouse.walls and (i, j+1) in warehouse.walls:
                taboo_cells.add((j, i))
            elif (i-1, j) in warehouse.walls and (i, j-1) in warehouse.walls:
                taboo_cells.add((j, i))

    # check rows
    for row in range(1, y_size-1):
        line = vis[row]
        left_wall, right_wall = 0, 0
        left_wall = "".join(line).find("# ", left_wall)
        while left_wall != -1:
            p = max(left_wall, right_wall+1)
            right_wall = "".join(line).find(" #", p) + 1

            temp_taboo_cells = set()
            j = left_wall + 1
            while (j, row) in inside_cells and j != right_wall:
                # up and down, both two side is not wall or row is in the targets
                # not taboo cell
                if (j, row+1) not in warehouse.walls and \
                   (j, row-1) not in warehouse.walls or \
                   (j, row) in warehouse.targets:
                    temp_taboo_cells = set()
                    break
                temp_taboo_cells.add((row, j))
                j += 1
            taboo_cells = taboo_cells.union(temp_taboo_cells)

            left_wall = "".join(line).find("# ", left_wall+1)

    # check columns
    for col in range(1, x_size-1):
        line = [vis[i][col] for i in range(y_size)]
        upper_wall, bottom_wall = 0, 0
        upper_wall = "".join(line).find("# ", upper_wall)
        while upper_wall != -1:
            p = max(upper_wall, bottom_wall+1)
            bottom_wall = "".join(line).find(" #", p) + 1

            i = upper_wall + 1
            temp_taboo_cells = set()
            while (col, i) in inside_cells and i != bottom_wall:
                # if left/right is not wall or column is in the targets, clear the temp_taboo_cells and stop check this line
                if (col+1, i) not in warehouse.walls and \
                   (col-1, i) not in warehouse.walls or \
                   (col, i) in warehouse.targets:
                    temp_taboo_cells = set()
                    break
                temp_taboo_cells.add((i, col))
                i += 1
            taboo_cells = taboo_cells.union(temp_taboo_cells)

            upper_wall = "".join(line).find("# ", upper_wall+1)

    for x, y in taboo_cells:
        vis[x][y] = "X"
    return "\n".join(["".join(line) for line in vis])

def check_inside(walls, start):
    '''
    Identify if the cells is inside a warehouse or not. The position of the worker
    must be inside the warehouse. Choose the position of the worker as start point
    to check the cells.

    @param warehouse:
        walls: the positions of the warehouse
        start: the position of the worker, also the start point

    @return
       A set of positions of cells inside the warehouse.
    '''
    frontier = [start]
    explored = []
    inside_cells = set()
    while len(frontier) > 0:
        now = frontier.pop()
        up = (now[0]-1, now[1])
        if up not in frontier and up not in explored and up not in walls:
            frontier.append(up)
        down = (now[0]+1, now[1])
        if down not in frontier and down not in explored and down not in walls:
            frontier.append(down)
        left = (now[0], now[1]-1)
        if left not in frontier and left not in explored and left not in walls:
            frontier.append(left)
        right = (now[0], now[1]+1)
        if right not in frontier and right not in explored and right not in walls:
            frontier.append(right)
        explored.append(now)
        inside_cells.add(now)
    return inside_cells

def print_taboo_cells(warehouse):
    '''
    Get taboo cells by calling taboo_cells()

    @param warehouse:
        a Warehouse object with a worker inside the warehouse

    @return
       A list of positions of taboo cells.
    '''
    taboo = taboo_cells(warehouse).split("\n")
    taboo_cells_in_warehouse = list(sokoban.find_2D_iterator(taboo, "X"))
    return taboo_cells_in_warehouse

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class Sokoban_State:
    '''
    Define the state of a node.
    An instance contains information about the boxes and the worker.
    '''
    def __init__(self, worker, boxes):
        self.worker = worker
        self.boxes = boxes

    def __len__(self):
        return len(self.boxes)

    def __lt__(self, other):
        return self.boxes < other.boxes

    def __eq__(self, other):
        if self.worker == other.worker:
            for box in self.boxes:
                # when one of the box is in different position, return False.
                if box not in other.boxes:
                    return False
        # when worker in different position, return False
        else:
            return False
        
        return True

    def __contains__(self, key):
        return key in self.boxes

    def __getitem__(self, key):
        for value, item in enumerate(self.boxes):
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the State")

    def __hash__(self):
        return hash(self.worker) ^ functools.reduce(operator.xor, [hash(box) for box in self.boxes])


class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. 
    
    '''
    
    #
    #         "INSERT YOUR CODE HERE"
    #
    #     Revisit the sliding puzzle and the pancake puzzle for inspiration!
    #
    #     Note that you will need to add several functions to 
    #     complete this class. For example, a 'result' method is needed
    #     to satisfy the interface of 'search.Problem'.
    #
    #     You are allowed (and encouraged) to use auxiliary functions and classes

    
    def __init__(self, warehouse, initial=None, goal=None):
        self.warehouse = warehouse.copy()
        if goal is None:
            self.goal = self.warehouse.targets
        else:
            self.goal = goal

        if initial is None:
            self.initial = Sokoban_State(warehouse.worker, warehouse.boxes)
        else:
            self.initial = initial

        self.taboo_cells_in_warehouse = print_taboo_cells(warehouse)

        assert len(self.initial.boxes) == len(self.goal)

        self.push_costs = warehouse.weights

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        """
        self.warehouse.worker = state.worker
        self.warehouse.boxes = state.boxes
        L = []
        # Up
        if check_elem_action(self.warehouse, "Up", self.taboo_cells_in_warehouse):
            L.append("Up")
        # Down
        if check_elem_action(self.warehouse, "Down", self.taboo_cells_in_warehouse):
            L.append("Down")
        # Left
        if check_elem_action(self.warehouse, "Left", self.taboo_cells_in_warehouse):
            L.append("Left")
        # Right
        if check_elem_action(self.warehouse, "Right", self.taboo_cells_in_warehouse):
            L.append("Right")

        return L

    def result(self, state, action):
        '''
        return the state results from executing the action in the given state.
        The action must be one of self.actions(state).
        The number of boxes and targets must be the same.
        '''
        # the value for each actions
        move = {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}
        # The action must be one of self.actions(state).
        assert action in self.actions(state)
        self.warehouse.worker = state.worker
        self.warehouse.boxes = state.boxes.copy()
        new_pos = (self.warehouse.worker[0]+move[action][0], self.warehouse.worker[1]+move[action][1])
        # when new position is on the box, the box need to be moved
        if new_pos in self.warehouse.boxes:
            box_index = self.warehouse.boxes.index(new_pos)
            self.warehouse.boxes[box_index] = (
                self.warehouse.boxes[box_index][0]+move[action][0],
                self.warehouse.boxes[box_index][1]+move[action][1]
            )
        # assign new position for worker
        self.warehouse.worker = new_pos
        assert len(state.boxes) == len(self.goal)
        return Sokoban_State(self.warehouse.worker, self.warehouse.boxes)

    def goal_test(self, state):
        '''
        If current state arrives goal state, return TRUE. Otherwise, return FALSE.
        The method of checking: compare the boxes and goal, when these two variables
        have same values, that means the current state arrives to the goal state.
        The length of boxes and goals must be the same.
        '''
        # check the number of boxes is same as the number of targets
        assert len(state.boxes) == len(self.goal)
        for box in state.boxes:
            if box not in self.goal:
                return False
        return True

    def path_cost(self, c, state1, action, state2):
        '''
        Return the path cost (within weights of boxes - weight cost/push cost) of the arrived state2 from the prior state1 via actions. 
        When the boxes are moved, the push cost (weight cost) will be added into cost.
        The number of boxes must remain unchanged through states.
        At most one box can be moved.
        '''
        # Elementary cost
        cost = c + 1
        # The number of boxes in state1 and state2 must be the same. 
        assert len(state1.boxes) == len(state2.boxes)
        # get moved_box index
        moved_box_index = [index for index, box in enumerate(state2.boxes) if box not in state1.boxes]
        # at most one box can be moved
        assert len(moved_box_index) <= 1
        # exist moved box
        if len(moved_box_index) == 1:
            cost += self.push_costs[moved_box_index[0]]
        return cost

    def h(self, node):
        '''
        Heuristic function for Sokoban to calculate the minimum distance from each boxes to their targets by using Manhattan distance.
        The push cost (weight) of the box is involved in the distance calculation.
        Different boxes can go to different targets.
        The target's variables represent the targets that do not have box on them. Similarly for the box's variables.
        Thus, the number of targets (without boxes) equals the number of boxes (not stand on any targets).    
        '''
        assert len(node.state.boxes) == len(self.goal)

        # boxes that not on targets
        boxes = [i for i in node.state.boxes if i not in self.warehouse.targets]

        # targets that not have box on them
        targets = [i for i in self.warehouse.targets if i not in node.state.boxes]

        # The number of boxes not on targets and the number of targets do not have box must be the same.
        assert len(boxes) == len(targets)
        min_distance = 0
        for box in boxes:
            min_value = sys.maxsize
            box_index = -1
            for target in targets:
                d = abs(box[0]-target[0]) + abs(box[1]-target[1])
                if d < min_value:
                    min_value = d
                    box_index = node.state.boxes.index(box)
            # multiply by push cost
            min_distance += (self.push_costs[box_index]*min_value)

        return min_distance

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_elem_action(warehouse, action, taboos):
    '''
    Determine if the given action is legal of not. 
    The illegal action include that pushing box onto taboo cells.

    @param warehouse: a valid Warehouse object

    @param action: an action. There are four possible actions which are "Right", "Left", "Up", "Down"

    @param taboos: a list of the positions of the taboo cells

    @return
        True: the action is legal
        False: the action is illegal

    '''
    # the value of each action
    move = {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}
    new_pos = (warehouse.worker[0]+move[action][0], warehouse.worker[1]+move[action][1])
    # new position of worker is not in the wall
    if new_pos in warehouse.walls:
        return False
    # when new position is on the box, the box need to be moved
    elif new_pos in warehouse.boxes:
        new_pos_box = (new_pos[0]+move[action][0], new_pos[1]+move[action][1])
        # new position of box not in wall or other boxes or taboo cells
        if new_pos_box in warehouse.walls or new_pos_box in warehouse.boxes or new_pos_box in taboos:
            return False
    return True


def check_elem_action_seq(warehouse, action_seq):
    '''
    
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Impossible', if one of the action was not valid.
           For example, if the agent tries to push two boxes at the same time,
                        or push a box into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''
    
    ##         "INSERT YOUR CODE HERE"
    # the value of each action 
    move = {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}
    worker = warehouse.worker
    boxes = warehouse.boxes.copy()
    for action in action_seq:
        new_position = (worker[0]+move[action][0], worker[1]+move[action][1])
        # new position of worker is not in the walls
        if new_position in warehouse.walls:
            return "Impossible"
        elif new_position in boxes:
            new_position_box = (new_position[0]+move[action][0], new_position[1]+move[action][1])
            # new position of box not in wall or other boxes
            if new_position_box in warehouse.walls or new_position_box in boxes:
                return "Impossible"
            # if new position of box is legal, it will move the box
            else:  
                index = boxes.index(new_position)
                boxes[index] = new_position_box
        # new position for worker
        worker = new_position

    X, Y = zip(*warehouse.walls)  # get walls from the warehouse text file
    x_size, y_size = 1+max(X), 1+max(Y)

    vis = [[" "] * x_size for y in range(y_size)]
    for (x, y) in warehouse.walls:
        vis[y][x] = "#"
    for (x, y) in warehouse.targets:
        vis[y][x] = "."
    # if worker is on a target, display "!", otherwise "@"
    # exploit the fact that Targets has been already processed
    if vis[worker[1]][worker[0]] == ".":  # y is worker[1], x is worker[0]
        vis[worker[1]][worker[0]] = "!"
    else:
        vis[worker[1]][worker[0]] = "@"
    # if a box is on a target, display "*"
    # exploit the fact that Targets has been already processed
    for (x, y) in boxes:
        if vis[y][x] == ".":  # if on target
            vis[y][x] = "*"
        else:
            vis[y][x] = "$"
    return "\n".join(["".join(line) for line in vis])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_weighted_sokoban(warehouse):
    '''
    This function analyses the given warehouse.
    It returns the two items. The first item is an action sequence solution. 
    The second item is the total cost of this action sequence.
    
    @param 
     warehouse: a valid Warehouse object

    @return
    
        If puzzle cannot be solved 
            return 'Impossible', None
        
        If a solution was found, 
            return S, C 
            where S is a list of actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
            C is the total cost of the action sequence C

    '''
    
    sokopan_puzzle = SokobanPuzzle(warehouse)
    t0 = time.time()
    sol_ts = search.astar_graph_search(sokopan_puzzle)
    t1 = time.time()
    print("Time: {}".format(t1-t0))
    actions = []
    c = 0
    if sol_ts is not None:
        print("Cost: {}".format(sol_ts.path_cost))
        c = sol_ts.path_cost
        for node in sol_ts.path():
            if node.action is not None:
                actions.append(node.action)           
        return actions, c
    else:
        return "Impossible"


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == "__main__":
    game = sokoban.Warehouse()
    game.load_warehouse("./warehouses/warehouse_8a.txt")
    print("The taboo cells found in the warehouse:")
    print(taboo_cells(game))
    
    # Test action sequences
    # print(check_elem_action_seq(game, ["Right", "Up", "Up", "Left"]))
    print(check_elem_action_seq(game, ["Left", "Left", "Left", "Up"]))
    print(check_elem_action_seq(game, ["Left", "Left", "Left", "Up", "Up"]))

    



