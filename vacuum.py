import random

class Cleaner:
    def __init__(self,num_dirty):
        self.robot_location = random.randint(1,9)
        self.dirty_squares = {}
        self.robot_moves = {}
        for i in range(1,10):
            self.robot_moves[i] = i + 1
        self.robot_moves[9] = 1
        
        while len(self.dirty_squares) < num_dirty:
            dirty_square = random.randint(1,9)
            if dirty_square in self.dirty_squares:
                continue
            self.dirty_squares[dirty_square] = True
    
    def simple(self,murphy):
        count = 0
        moves = 0
        while count < 1000:
            if len(self.dirty_squares) == 0:
                break
            count +=1 
            moves += 1
            if murphy:
                luck = random.randint(1,4)
                sensor_luck = random.randint(1,10)
            else:
                luck = 0
                sensor_luck = 0
            if self.robot_location in self.dirty_squares and sensor_luck != 1:
                if luck == 1:
                    continue
                del self.dirty_squares[self.robot_location]
            else:
                if luck == 1:
                    self.dirty_squares[self.robot_location] = True
                self.robot_location = self.robot_moves[self.robot_location]
        #print(f'Number of moves was: {moves}')
        return moves
            
    def randomized(self,murphy):
        count = 0
        moves = 0
        if murphy:
            luck = random.randint(1,4)
            sensor_luck = random.randint(1,10)
        else:
            luck = 0
            sensor_luck = 0
        while count < 10000000:
            if len(self.dirty_squares) == 0:
                break
            count +=1 
            moves += 1
            if self.robot_location in self.dirty_squares and sensor_luck != 1:
                if luck == 1:
                    continue
                del self.dirty_squares[self.robot_location]
            else:
                if luck == 1:
                    self.dirty_squares[self.robot_location] = True
                self.robot_location = random.randint(1,9)
        #print(f'Number of moves was: {moves}')
        return moves
    
    def run_test(self, num_dirty):
        num_tests = 100
        total_moves = 0
        for _ in range(num_tests):
            robot = Cleaner(num_dirty)
            total_moves += robot.simple(False)
        average_moves = total_moves // num_tests
        print(f'Average number of moves for simple test with {num_dirty} dirty rooms: {average_moves}')
        for _ in range(num_tests):
            robot = Cleaner(num_dirty)
            total_moves += robot.randomized(False)
        average_moves = total_moves // num_tests
        print(f'Average number of moves for randomized test with {num_dirty} dirty rooms: {average_moves}')
        for _ in range(num_tests):
            robot = Cleaner(num_dirty)
            total_moves += robot.simple(True)
        average_moves = total_moves // num_tests
        print(f'Average number of moves for simple murphy test with {num_dirty} dirty rooms: {average_moves}')
        for _ in range(num_tests):
            robot = Cleaner(num_dirty)
            total_moves += robot.randomized(True)
        average_moves = total_moves // num_tests
        print(f'Average number of moves for randomized murphy test with {num_dirty} dirty rooms: {average_moves}')
    
        
robot = Cleaner(1)
robot.run_test(1)
robot = Cleaner(3)
robot.run_test(3)
robot = Cleaner(5)
robot.run_test(5)
