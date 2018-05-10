import Queue
from itertools import count
from numpy import asarray
from copy import deepcopy

class SixPuzzle(object):

	def __init__(self, start):
		self.board = [tile for sublist in start for tile in sublist]
		self.height = len(start)
		self.width = len(start[0])
		self.empty_tile = None

		if self.height != 2 or self.width != 3:
			raise Exception("Start state is not a proper six-puzzle board.")

	@property
	def unflattened(self):
		return [self.board[:3], self.board[3:]]

	@property
	def all_transitions(self):
		''' 
			Since we flattened our board, moving left or right corresponds to moving the index by -1 or +1, respectively.
			Moving down or up corresponds to moving the index by -width of board or +width, respectively.
		'''
		possible = []

		for tile in (self.board.index(self.empty_tile) - 1, self.board.index(self.empty_tile) + 1):
			if tile / self.width == self.board.index(self.empty_tile) / self.width:
				possible.append(tile)

		for tile in (self.board.index(self.empty_tile) - self.width, self.board.index(self.empty_tile) + self.width):
			if (tile >= 0) and (tile < len(self.board)):
				possible.append(tile)

		# sort the possible moves such that a tile with a smaller number is moved first.
		return sorted(possible)


	@property
	def is_goal_state(self):
		'''
			Our goal state is [[None, 1, 2], 
								  [5, 4, 3]]
		'''
		return self.board == [self.empty_tile, 1, 2, 5, 4, 3]

	

	def transition(self, tile, board):

		new = list(board)
		empty_index = self.board.index(self.empty_tile)
		new_tile_val = board[tile]
		new[tile] = board[empty_index]
		new[empty_index] = new_tile_val
		return SixPuzzle([new[:3],new[3:]])


class State(object):

	def __init__(self, puzzle, transition=None, parent_state=None):
		'''
			If the parents and transition are None, we have a start state.
		'''
		self.puzzle = puzzle
		self.transition = transition
		self.parent_state = parent_state

	@property
	def solution_path(self):
		'''
			Returns the path from the start state to the goal state.
		'''
		path = []
		while self:
			path.append(self)
			self = self.parent_state
		path = reversed(path)
		return path

	
def solveBFS(start):
	"""
	Breadth-First Search
	"""
	queue, visited = [State(start)], set()
	if queue == []: raise Exception("Failure, empty queue.")
	else:
		while queue:
			state = queue.pop(0)

			#if current state equals goal state, we are done
			if state.puzzle.is_goal_state:
				return state.solution_path 

			
			children = []

			# otherwise, we need to explore all children states
			for tile in state.puzzle.all_transitions:
				copy = deepcopy(state)
				tile_value = copy.puzzle.board[tile]
				child_board = copy.puzzle.transition(tile, copy.puzzle.board)
				child = State(child_board, tile , copy)

				if child.puzzle not in visited:
					queue.append(child)
					visited.add(child.puzzle)

def solveUCS(start):
	"""
	Breadth-First Search
	"""
	queue, visited = Queue.PriorityQueue(), []
	queue.put((0,State(start))) #start node has cost of 0

	while not queue.empty():
		deqeued = queue.get()
		state_cost = deqeued[0] #priority is the cost
		state = deqeued[1] # the actual state

		if state.puzzle.is_goal_state:
			return state.solution_path

		children = []

		for tile in state.puzzle.all_transitions:
			copy = deepcopy(state)
			tile_value = copy.puzzle.board[tile]
			child_board = state.puzzle.transition(tile, copy.puzzle.board)
			child = State(child_board, tile, copy)

			if child.puzzle not in visited:
				queue.put((state_cost + 1, child))
				visited.append(child.puzzle)

def solveDFS(start):
	"""
	Solve using Depth First Search. This is exactly the same as the solveBFS() method
	but uses a stack instead of a queue.
	"""
	stack, visited = [State(start)], set()
	if stack == []: raise Exception("Failure, empty queue.")
	else:
		while stack:
			state = stack.pop()
			#if current state equals goal state, we are done
			if state.puzzle.is_goal_state:
				return state.solution_path 

	
			# otherwise, we need to explore all children states
			for tile in reversed(state.puzzle.all_transitions):
				copy = deepcopy(state)
				tile_value = copy.puzzle.board[tile]
				child_board = copy.puzzle.transition(tile, copy.puzzle.board)
				child = State(child_board, tile , copy)

				if child.puzzle not in visited:
					stack.append(child)
					visited.add(child.puzzle)

'''
Based off of Wikipedia's IDA* pseudocode.
'''
def solveIDS(start):

	''' using Manhattan distance as heuristic'''
	def h(board):
		cost = 0
		goal = [None,1,2,3,4,5]
		for i in range(1,6):
			cost += abs(goal.index(i) - board.index(i))
		return cost

	''' in this case, edges don't hold different weights ''' 
	def g(cost):
		return cost + 1

	def search(path, cost, bound):

		state = path[-1]
		f = g(cost) + h(state.puzzle.board)
		if f > bound:
			return float(f)
		if state.puzzle.is_goal_state:
			return 'FOUND'
		minimum = float('inf')

		# explore all children states
		for tile in state.puzzle.all_transitions:
			copy = deepcopy(state)
			child_board = copy.puzzle.transition(tile, copy.puzzle.board)
			child = State(child_board, tile , copy)
			if child.puzzle not in path:
				path.append(child)
				t = search(path, cost + 1, bound)
				if t == 'FOUND':
					return 'FOUND'
				if isinstance(t, float) and t < minimum:
					minimum = t 
					path.pop()
		return minimum

	path = [State(start)]
	bound = h(start.board) # we start our bound at the heuristic value for our initial start board
	
	while(True):
		t = search(path, 0, bound)
		if t == 'FOUND':
			return path
		if t == float('inf'):
			raise Exception('no solution found.')
		else:
			bound = t


if __name__ == "__main__":

	board = [[1,4,2],[5,3,None]]
	puzzle = SixPuzzle(board)
	path = solveBFS(puzzle)

	#using numpy to help visualize board while printing
	for state in path:
		print asarray(state.puzzle.unflattened)





