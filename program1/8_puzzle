import heapq
import random
from math import sqrt, pow

class Node:
    def __init__(self, parent=None, state=[], left = None, right = None, down=None, up=None):
        self.parent= parent
        self.state = state
        if not parent:
            self.height = 0
        else:
            self.height = parent.height + 1
    def __eq__(self, other):
        return (self.height == other.height)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return (self.height < other.height)

    def __gt__(self, other):
        return (self.height > other.height)

    def __le__(self, other):
        return (self < other) or (self == other)

    def __ge__(self, other):
        return (self > other) or (self == other) 
    
class Game:
    def __init__(self,goal_state=None, initial_state=None):
        '''
        Sets up several class variables, including initial state
        and goal state, a priority queue for keeping track of nodes
        to visit based off its calculated cost, a visited set to keep
        track of what states/configurations have been visited, a 
        hashmap keeping track of all the positions the goal state are in, 
        and all the moves that are available from each square
        '''
        self.goal_state = goal_state
        self.initial_state = initial_state
        self.visited = set()
        self.priority_queue = []
        heapq.heapify(self.priority_queue)
        self.root = Node(None,self.initial_state)
        heapq.heappush(self.priority_queue, (0, self.root.height, 0, self.root))
        self.goal_positions = self.get_positions(self.goal_state)
        self.moves = self.construct_moves()
        
    def start_game(self, algorithm, heuristic):
        '''
        Goes through the priority queue made up of nodes that will pop
        the node with lowest cost based off the algorithm/heuristic 
        combination. Child nodes will be added to the queue if the state
        of those nodes have not been visited. Will return once the correct
        state has been found or the number of nodes visited > ten million
        :param algorithm: String 
        :param heuristic: String
        : return: number of nodes visited (int), height of goal state node (int)
        '''
        count = 0
        moves = 1
        num_nodes = 1
        while count < 10000000:
            count += 1
            priority, height, idx, current_node = heapq.heappop(self.priority_queue)
            current_state = current_node.state
            self.visited.add(tuple(current_state))
            state_positions = self.get_positions(current_state)
            if state_positions == self.goal_positions:
                self.path = self.get_path(current_node)
                return count, current_node.height 
            b_position = self.get_index(current_state,'b')
            moves = self.moves[b_position]
            for i in range(len(moves)):
                idx, direction = moves[i]
                new_state = self.get_new_state(current_state,b_position,idx)
                child_node = Node(current_node, new_state)
                num_nodes += 1
                if direction == 'left':
                    current_node.left = child_node
                    placement = 0
                elif direction == 'up':
                    current_node.up = child_node
                    placement = 1
                elif direction == 'right':
                    current_node.right = child_node
                    placement = 2
                elif direction == 'down':
                    current_node.down = child_node
                    placement = 3
                
                # Check if node has already been visited
                if tuple(new_state) in self.visited:
                    continue
                
                # Add state to visited set
                self.visited.add(tuple(new_state))
                positions = self.get_positions(new_state)
                
                # Different heuristic options
                if heuristic == 'tiles_mismatched':
                    cost = self.misplaced_tiles(self.goal_positions,positions)
                elif heuristic == 'manhattan':
                    cost = self.manhattan(positions)
                elif heuristic == 'euclidean':
                    cost = self.euclidean(positions)
                '''
                else:
                    print('This program needs to be given a heuristic')
                    exit()
                '''
                # To change algorithm to A* 
                if algorithm == 'a_star':
                    cost += child_node.height
                
                # Add child node to the priority queue     
                heapq.heappush(self.priority_queue, (cost,child_node.height,placement,child_node))
        return count
                
    def convert_to_matrix(self,state):
        matrix = []
        z = 0
        for i in range(3):
            matrix.append([])
            for j in range(3):
                matrix[i].append(state[z])
                z += 1
        return matrix
    
    def euclidean(self, positions):
        total = 0
        for i in range(1,9):
            coord1 = self.goal_positions[i]
            coord2 = positions[i]
            diff = sqrt(pow(coord1[0]-coord2[0],2) + pow(coord1[1]-coord2[1],2))
            total += diff
        return total
    
    def manhattan(self, positions):
        total = 0
        for i in range(1,9):
            coord1 = self.goal_positions[i]
            coord2 = positions[i]
            vertical_diff = abs(coord1[0] - coord2[0])
            horizontal_diff = abs(coord1[1] - coord2[1])
            total += vertical_diff + horizontal_diff
        return total
           
    # Returns path from initial state to goal state
    def get_path(self,parent):
        current = parent
        path = []
        while current != self.root:
            path.append(current.state)
            current = current.parent
        path.append(current.state)
        path.reverse()
        #self.display_path(path)
        return path
    
    def display_path(self,path):
        for i in range(len(path)):
            if (i+1) % 3 == 0 or i == len(path) - 1:
                print(f'{path[i]}\n') 
            else:
                print(f'{path[i]} -> ',end='')
        '''
        for i in range(len(path)):
            print(path[i])
            matrix = self.convert_to_matrix(path[i])
            for row in range(len(matrix)):
                print(matrix[row])
            print('\n')
        '''
            
    def get_new_state(self,state,index,new_index):
        new_state = state.copy()
        new_state[index], new_state[new_index] = new_state[new_index], new_state[index] 
        return new_state
        
    def construct_moves(self):
        moves = {}
        '''
        moves[0] = [1,3]
        moves[1] = [0,2,4]
        moves[2] = [1,5]
        moves[3] = [0,4,6]
        moves[4] = [3,1,5,7]
        moves[5] = [4,2,8]
        moves[6] = [3,7]
        moves[7] = [6,4,8]
        moves[8] = [7,5]
        '''
        moves[0] = [(1,'right'),(3,'down')]
        moves[1] = [(0,'left'),(2,'right'),(4,'down')]
        moves[2] = [(1,'left'),(5,'down')]
        moves[3] = [(0,'up'),(4,'right'),(6,'down')]
        moves[4] = [(3,'left'),(1,'up'),(5,'right'),(7,'down')]
        moves[5] = [(4,'left'),(2,'up'),(8,'down')]
        moves[6] = [(3,'up'),(7,'right')]
        moves[7] = [(6,'left'),(4,'up'),(8,'right')]
        moves[8] = [(7,'left'),(5,'up')]
        return moves
        
    
    def get_index(self,state,val):
        for i in range(len(state)):
            if state[i] == val:
                return i
        return -1
        
    def get_positions(self,state):
        z = 0
        positions = {}
        for i in range(3):
            for j in range(3):
                positions[state[z]] = [i,j]
                z += 1
        return positions
                
    def misplaced_tiles(self,goal,state):
        total = 0
        for i in range(1,9):
            if goal[i] != state[i]:
                total += 1
        return total             

def count_inversions(state):
    count = 0
    for i in range(len(state)):
        for j in range(i+1, len(state)):
            if state[j] == 'b' or state[i] == 'b':
                continue
            if state[j] < state[i]:
                count += 1
    return count 
    
def check_inversions(goal_state,initial_state):
    goal_inversions = count_inversions(goal_state)
    initial_inversions = count_inversions(initial_state)
    if goal_inversions % 2 != initial_inversions % 2:
        return False
    return True

def randomize_state():
    state = [1,2,3,4,5,6,7,8,'b']
    random.shuffle(state)
    return state

def randomize_initial_state(goal_state):
    solvable = False
    while solvable == False:
        initial_state = randomize_state()
        if check_inversions(goal_state, initial_state):
            return initial_state
        
if __name__ == '__main__':
    algorithms = ['best_first','a_star']
    heuristics = ['tiles_mismatched', 'manhattan', 'euclidean']
    
    #  Randomize States
    #goal_state = randomize_state()
    #initial_state = randomize_initial_state(goal_state)
    
    #  Hard-code States
    goal_state = [1,2,3,4,5,6,7,8,'b']
    initial_state = [2,1,4,3,5,6,7,8,'b']
    if check_inversions(goal_state,initial_state) == False:
        print('These states are not solvable')
  
    # Five random initial states for testing
    initial_states = []
    for _ in range(5):
        initial_states.append(randomize_initial_state(goal_state)) 
       
    # Test each heuristic for each algorithm and display results for 
    # initial_states
    for algorithm in algorithms:
        game = Game(goal_state,initial_state)
        for heuristic in heuristics:
            print(f'{algorithm} search:\n{heuristic}')
            num_steps = 0
            num_visited = 0
            for i in range(len(initial_states)):
                if check_inversions(goal_state,initial_states[i]) == False:
                    print('These states are not solvable')
                    exit
                game = Game(goal_state,initial_states[i])
                path_num = i + 1
                print(f'\nSolution path #{path_num}:')
                nodes_visited, height = game.start_game(algorithm, heuristic)
                game.display_path(game.path)
                num_steps += height
                print(f'Height for this solution: {height}')
                print(f'Number of nodes visited: {nodes_visited}\n')
                num_visited += nodes_visited
            average_num_steps = num_steps // len(initial_states)
            average_num_visited = num_visited // len(initial_states)
            print('Averages:')
            print(f'Average number of steps: {average_num_steps}')
            print(f'Average number of nodes visited is: {average_num_visited}\n\n')
            
