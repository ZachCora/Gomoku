"""Authors:
Zach Cora
Austin Moore
Vivek Bigelow
"""
################################################################################
from abc import ABCMeta, abstractmethod
import random
from math import *
import time
import copy
################################################################################
"""Tokens"""
White = u'\u25CB'
Black = u'\u25cf'

def opposite(token):
	if token == Black:
		return White
	else:
		return Black
################################################################################
"""Returns current time in milliseconds"""		
current_milli_time = lambda: int(round(time.time() * 1000))
################################################################################
"""Monte Carlo player allowed search time(in milliseconds)"""
Search_Duration = 3500
################################################################################
"""Set this to print board and player turns"""
Printing = True
"""Set this to print board value matrix (testing)"""
pValues = True
################################################################################
"""Runs n games with computer vs computer, returns win stats (set Printing = False)"""
def enumerateGames(n):
	i = 0
	B = 0
	W = 0
	D = 0
	while i < n:
		G = Game(Gomoku(), 15, True)
		x = G.play()
		if x == Black:
			B += 1
		if x == White:
			W += 1
		if x == "draw":
			D += 1
		i += 1
	print "Black: " + str(B)
	print "White: " + str(W)
	print "Draw: " + str(D)
	
################################################################################
"""Call this to begin a game of gomoku, with optional size argument"""
def playGomoku(size = 15):
	game = Game(Gomoku(), size)
	game.play()
	
################################################################################
"""The Game class, that runs the show"""
class Game:

	def __init__(self, rules, size, all_cpu = False):
		self.rules = rules
		self.board = Board(size)
		self.players = []
		if all_cpu:
			self.players.append(Computer("black"))
			self.players.append(Computer("white"))
		else:
			self.players.append(getPlayerType("black"))
			self.players.append(getPlayerType("white"))
		self.turn = 0
	
	"""Plays the game (return values are for win stat testing purposes)"""	
	def play(self):
		p = self.players[self.turn]
		self.board.show()
		m = (self.board.size / 2)
		self.rules.makeMove(self.board, [m,m], p.token)
		self.turn = 1 - self.turn
		while True:
			if self.board.isFull():
				break
			p = self.players[self.turn]
			self.board.show()
			m = p.getMove(self.board)
			self.rules.makeMove(self.board, m, p.token)
			if self.rules.checkForWin(self.board, m, p.token):
				self.board.show()
				if self.turn == 0:
					if Printing:
						print "Black wins!"
					return Black
					break
				else:
					if Printing:
						print "White wins!"
					return White
					break
			self.turn = 1 - self.turn
		self.board.show()
		if Printing:
			print "Draw!"
		return "draw"

################################################################################
"""The game board!!!"""
class Board:

	def __init__(self, size):
		self.size = size
		self.data = [['+'] * (size) for i in range(size)]
		"""Sets a negative bias for the edges of the board, positive in center"""
		self.values = [[0] * (size) for i in range(size)]
		for row in range(0, self.size):
			for col in range(0, self.size):
				if abs(row - (self.size / 2)) < 2 and abs(col - (self.size / 2)) < 2:
					self.values[row][col] = 1
				if row < 4 or row > (self.size - 5) or col < 4 or col > (self.size - 5):
					if row < 3 or row > (self.size - 4) or col < 3 or col > (self.size - 4):
						if row < 2 or row > (self.size - 3) or col < 2 or col > (self.size - 3):
							if row < 1 or row > (self.size - 2) or col < 1 or col > (self.size - 2):
								self.values[row][col] = -584
							else:
								self.values[row][col] = -72
						else:
							self.values[row][col] = -8
					else:
						self.values[row][col] = -1
						
	def __str__(self):
		board = ''
		board += '   '
		for c in range(self.size):
			cn = str(c + 1)
			board += cn + (' ' * (2-len(cn)))
		board += '\n'
		for r in range(self.size):
			rn = str(r + 1)
			board += rn + (' ' * (3 - len(rn)))
			for c in range(self.size):
				board += self.data[r][c] + ' '
			board += '\n'
		return board
		
	"""Shows the board (just printing causes problems with unicode)"""
	def show(self):
		if Printing:
			print self.__str__()
		self.showValues()
			
	"""Shows the current value matrix"""
	def showValues(self):
		if pValues:
			for x in self.values:
				list = []
				for y in x:
					s = str(y)
					list.append(s + (" " * (4-len(s))))
				print " ".join(list)

	"""Determines if a move is legal"""
	def isLegal(self, move):
		return self.isWithinBounds(move) and self.data[x][y] == '+'
		
	"""Determines if a move is within the bounds of the board"""
	def isWithinBounds(self,move):
		x = move[0]
		y = move[1]
		return x >= 0 and y >= 0 and x < self.size and y < self.size
		
	"""Creates a board clone, for use in the AI"""	
	def clone(self):
		clone = Board(self.size)
		clone.data = copy.deepcopy(self.data)
		clone.values = copy.deepcopy(self.values)
		return clone
		
	"""returns a list of all currently legal moves"""
	def getLegalMoves(self):
		moves = []
		for i in range(0, self.size):
			for j in range(0,self.size):
				if self.data[i][j] == "+":
					moves.append([i,j])
		return moves

	"""Determines if the board is full"""
	def isFull(self):
		for x in self.data:
			for y in x:
				if y == '+':
					return False
		return True	
		
	"""Finds nearest tile in 4/8 directions from one point, not including opposites"""
	def findNeighbors (self, row, column):
		list = []
		list.append((row-1, column))
		list.append((row-1, column-1))
		list.append((row, column-1))
		list.append((row-1, column+1))
		return list

	"""Value Matrix Update Functions"""
	def update(self, move, token):
		self.update1(move, token)
		self.update2(move, token)
		self.update3(move, token)
		self.update4(move, token)

	def update1(self, move, token): #Updates values based on offense
		legal = self.findNeighbors(move[0], move[1])
		self.values[move[0]][move[1]] = -float("inf")
		for x in legal:
			self.updateDirection1(move, x, token)

	def update2(self, move, token): #updates values based on defense, minus threats
		O = opposite(token)
		legal = self.findNeighbors(move[0], move[1])
		self.values[move[0]][move[1]] = -float("inf")
		for x in legal:
			self.updateDirection2(move, x, token)

	def update3(self, move, token): #updates values based on tiles surrounded on two sides
		legal = self.findNeighbors(move[0], move[1])
		self.values[move[0]][move[1]] = -float("inf")
		for x in legal:
			self.updateDirection3(move, x, token)
			
	def update4(self, move, token): #Updates values based on blocking threats
		legal = self.findNeighbors(move[0], move[1])
		self.values[move[0]][move[1]] = -float("inf")
		for x in legal:
			self.updateDirection4(move, x, token)
			
	"""Each of these updates on one of the 4 directions given by findNeighbors()"""		
	def updateDirection1(self, m1, m2, token):
		change1 = [m2[0] - m1[0], m2[1] - m1[1]]
		change2 = [m1[0] - m2[0], m1[1] - m2[1]]
		
		r1 = self.range(m1, change1, token) - 1
		r2 = self.range(m1, change2, token) - 1
	
		line = []
		
		current = [m1[0] + min(r1,4) * change1[0], m1[1] + min(r1,4) * change1[1]]
		for i in range(max(-r1,-4), min(r2 + 1,5)):
			line.append([current, self.data[current[0]][current[1]]])
			current = [current[0]+change2[0], current[1]+change2[1]]
		for i in range(0,len(line)):
			place = line[i]
			if place[1] != '+':
				continue
			points = 0
			upper = i
			lower = i
			j = i
			while j >= max(0,i-4):
				this = line[j]
				if this[1] == opposite(token):
					break
				if this[1] == token:
					points = incrementPoints(points)
				j -= 1
			j = i
			while j <= min(len(line)-1,i+4):
				this = line[j]
				if this[1] == opposite(token):
					break
				if this[1] == token:
					points = incrementPoints(points)
				j += 1
			self.values[place[0][0]][place[0][1]] += points
			self.values[place[0][0]][place[0][1]] -= decrementPoints(points)
			
			"""This part accounts for threats of size 3 or 4"""
			f1 = False
			f2 = False
			right = 0
			left = 0
			j = i + 1
			while j < len(line) and line[j][1] == token:
				right += 1
				j += 1
			if j < len(line) and line[j][1] == '+':
				f2 = True
			j = i - 1
			while j >= 0 and line[j][1] == token:
				left += 1
				j -= 1
			if j >= 0 and line[j][1] == '+':
				f1 = True
			number = right + left
			if number == 3:
				if right < 3 and right > 0:
					self.values[place[0][0]][place[0][1]] = 2000
					if f1 and f2:
						self.values[place[0][0]][place[0][1]] = 8000
				else:
					self.values[place[0][0]][place[0][1]] = 5000
					if (f1 and (left == 3)) or (f2 and (right == 3)):
						self.values[place[0][0]][place[0][1]] = 10000
				
			if number >= 4:
				self.values[place[0][0]][place[0][1]] = float("inf")
			
	def updateDirection2(self, m1, m2, token):
		change1 = [m2[0] - m1[0], m2[1] - m1[1]]
		change2 = [m1[0] - m2[0], m1[1] - m2[1]]
		
		enemy = opposite(token)
		
		r1 = self.range(m2, change1, enemy)
		r2 = self.range([m1[0] + change2[0],m1[1] + change2[1]] , change2, enemy)
		
		line = []
		current = [m1[0] + 3 * change1[0], m1[1] + 3 * change1[1]]
		for i in range(-3,4):
			if self.isWithinBounds(current):
				line.append([current, self.data[current[0]][current[1]]])
			else:
				line.append([current, "OutOfBounds"])
			current = [current[0]+change2[0], current[1]+change2[1]]
		for i in range(-3,4):
			if i == 0:
				continue
			place = line[i+3]
			if place[1] != '+':
				continue
			points = 0
			left = 0
			right = 0
			total = 0
			
			for j in range(max(-3, i-4),min(4, i+5)):
				if (i < 0 and j > r2) or (i > 0 and j < -r1):
					continue
				if line[j+3][1] == enemy:
					total += 1
					points = incrementPoints(points)
					if j < 0:
						left += 1
					if j > 0:
						right += 1
			if i < 0:
				self.values[place[0][0]][place[0][1]] -= points
				while right > 0:
					points = decrementPoints(points)
					right -= 1
				self.values[place[0][0]][place[0][1]] += points
			elif i > 0:
				self.values[place[0][0]][place[0][1]] -= points
				while left > 0:
					points = decrementPoints(points)
					left -= 1
				self.values[place[0][0]][place[0][1]] += points

	def updateDirection3(self, m1, m2, token):
		change = [m2[0] - m1[0], m2[1] - m1[1]]
		counter = 0
		points = 0
		list = []
		while counter < 5 and self.isWithinBounds(m2):
			if self.data[m2[0]][m2[1]] == opposite(token):
				points = incrementPoints(points)
			elif self.data[m2[0]][m2[1]] == "+":
				list.append(m2)
			elif self.data[m2[0]][m2[1]] == token:
				for x in list:
					self.values[x[0]][x[1]] -= points
				break
			m2 = [m2[0] + change[0], m2[1] + change[1]]
			counter += 1
		m2 = [m1[0] - change[0], m1[1] - change[1]]
		counter = 0
		points = 0
		list = []
		while counter < 5 and self.isWithinBounds(m2):
			if self.data[m2[0]][m2[1]] == opposite(token):
				points = incrementPoints(points)
			elif self.data[m2[0]][m2[1]] == "+":
				list.append(m2)
			elif self.data[m2[0]][m2[1]] == token:
				for x in list:
					self.values[x[0]][x[1]] -= points
				break
			m2 = [m2[0] - change[0], m2[1] - change[1]]
			counter += 1
	
	"""This should be contained in updateDirection2, but when coding updateDirection2,
	certain things were done which made this incompatible with it, which would take 
	almost entirely rewriting updateDirection2 in order to fix."""
	def updateDirection4(self, m1, m2, token):
		change1 = [m2[0] - m1[0], m2[1] - m1[1]]
		change2 = [m1[0] - m2[0], m1[1] - m2[1]]
		
		r1 = self.range(m1, change1, token) - 1
		r2 = self.range(m1, change2, token) - 1
	
		line = []
		
		current = [m1[0] + 4 * change1[0], m1[1] + 4 * change1[1]]
		for i in range(-4,5):
			if self.isWithinBounds(current):
				line.append([current, self.data[current[0]][current[1]]])
			else:
				line.append([current, "OutOfBounds"])
			current = [current[0]+change2[0], current[1]+change2[1]]
		for i in range(0,9):
			place = line[i]
			if place[1] != '+':
				continue
			
			enemy = opposite(token)
				
			f1 = False
			f2 = False
			right = 0
			left = 0
			j = i + 1
			while j < 9 and line[j][1] == enemy:
				right += 1
				j += 1
			if j < 9 and line[j][1] == '+':
				f2 = True
			j = i - 1
			while j >= 0 and line[j][1] == enemy:
				left += 1
				j -= 1
			if j >= 0 and line[j][1] == '+':
				f1 = True
			number = right + left
			if number == 3:
				if right < 3 and right > 0:
					self.values[place[0][0]][place[0][1]] = 2000
					if f1 and f2:
						self.values[place[0][0]][place[0][1]] = 8000
				else:
					self.values[place[0][0]][place[0][1]] = 5000
					if (f1 and (left == 3)) or (f2 and (right == 3)):
						self.values[place[0][0]][place[0][1]] = 10000
				

	#how far a token has influence in a certain direction
	def range(self, move, change, token): 
		if not self.isWithinBounds(move) or self.data[move[0]][move[1]] == opposite(token):
			return 0
		else:
			return 1 + self.range([move[0]+change[0],move[1]+change[1]], change, token)	
		
################################################################################
"""Player classes"""
class Player:
	__metaclass__ = ABCMeta

	@abstractmethod
	def getMove(self): pass
	
"""Accepts a raw input to play the game"""
class Human (Player):

	def __init__(self, color):
		self.name = raw_input("Name for " + color.lower() + " player?\n")
		if color.lower() == "black":
			self.token = u'\u25cf'
		if color.lower() == "white":
			self.token = u'\u25CB'

	def getMove(self, board):#Gets a move from the human player
		if Printing:
			print "It is now " + self.name + "'s turn."
		row = raw_input("In which row would you like to place your token?\n")
		col = raw_input("In which column would you like to place your token?\n")
		move = []
		if not (isNumber(row) and isNumber(col)):
			print "One or more of the values you entered is not an integer!\n"
			return self.getMove(board)
		move.append(int(row) - 1)
		move.append(int(col) - 1)
		if not board.isWithinBounds(move):
			print "Values out of bounds! Please try again.\n"
			return self.getMove(board)
		if board.data[move[0]][move[1]] != '+':
			print "That space is taken! Please try again.\n"
			return self.getMove(board)
		return move
		
"""Uses value matrix of board to determine best move"""
class Computer (Player):

        def __init__(self, color):
                if color.lower() == "black":
                        self.token = u'\u25cf'
                        self.name = "Black"
                if color.lower() == "white":
                        self.token = u'\u25CB'
                        self.name = "White"

        def getMove(self, board):
                if Printing:
                	print "It is now " + self.name + "'s turn."
                maxima = []
                for i in range(0, board.size):
                        maxima.append(max(board.values[i]))
                maxVal = max(maxima)
                positions = []
                for i in range(0, board.size):
                        for j in range(0, board.size):
                                if board.values[i][j] == maxVal:
                                        positions.append([i, j])
                return random.choice(positions)
                
"""Is given a globally-defined search time, and when time expires, 
it returns the best move it has found so far"""
class MonteCarlo (Player):
	def __init__(self,color):
		self.name = "Monte Carlo Player"
		if color.lower() == "black":
			self.token = Black
		if color.lower() == "white":
			self.token = White
		self.rules = Gomoku()
		self.rootNode = None

	def getMove(self,board):
		if Printing:
			print "It is now " + self.name + "'s turn."
		boardClone = board.clone()
		legalMoves = boardClone.getLegalMoves()
		self.rootNode = Node(board = boardClone, token = opposite(self.token))
		self.rootNode.getChildren(legalMoves)
		# startTime = current_milli_time()
		# endTime = startTime + Search_Duration
		theRightMove = self.findOptimumMove(boardClone, self.rootNode.childNodes)
		return theRightMove
		#return the move that has the most wins
		
		#return sorted(self.rootNode.childNodes, key = lambda c: c.wins) [-1].move 
		
	def findOptimumMove(self,board,nodes):
		startTime = current_milli_time()
		endTime = startTime + Search_Duration
		while current_milli_time() < endTime:
			self.monteCarloSearch(endTime, board, nodes)

		bestNode = sorted(self.rootNode.childNodes, key = lambda c: c.wins + c.value) [-1] 
		#return the move that has the most wins
		print bestNode
		return bestNode.move

	def monteCarloSearch(self,time,board,nodes):
		simulationCount = 0
		while current_milli_time() < time :
			nodes = sorted(nodes, key = lambda c: c.value, reverse = True)
			nodesToSimulate = []
			for x in range(0,5):
				nodesToSimulate.append(nodes[x])
			self.simulateInOrderOfArray(time, board.clone(), nodesToSimulate, nodes)
			simulationCount += 1
		print "simulations: " + str(simulationCount)

	def simulateInOrderOfArray(self, time, board, nodesToSimulate, nodes):
		for x in nodesToSimulate:
			even = True
			i = 0
			self.rules.makeMove(board,x.move, x.token)
			while current_milli_time() < time and i < len(nodes):
				if even == True:
					token = self.token
				else:
					token = opposite(self.token)
				node = random.choice(nodes)

				self.rules.makeMove(board, node.move, token)
				even = not even
				i += 1
			if self.rules.checkForWin(board, x.move, self.token):
				result = 1
			elif self.rules.checkForWin(board, x.move, opposite(self.token)):
				result = -1
			else:
				result = 0
			self.backpropagate(x, result, board)

	def backpropagate(self, node, result, board):
		node.updateNode(result,board)

	def orderNodes(self, nodes):
		newOrder = sorted(nodes, key = lambda c: (c.value + c.wins), reverse =True)
		return newOrder

"""Returns a random legal move"""
class Dummy (Player):
	def __init__(self,color):
		self.name = "Dummy A.I."
		if color.lower() == "black":
			self.token = Black
		if color.lower() == "white":
			self.token = White

	def getMove(self,board): #Makes a random move
		if Printing:
			print "It is now " + self.name + "'s turn."
		moves = board.getLegalMoves()
		move = random.choice(moves)
		
		return move

################################################################################
"""Node for the Monte Carlo Tree"""
class Node:
	def __init__(self, move = None, board = None, parent = None, token = None):
		self.move = move
		self.board = board
		self.parent = parent
		self.token = token
		self.childNodes = []
		self.wins = 0
		self.visits = 0
		self.value = 0
		# self.untriedMoves = self.board.getLegalMoves()

	def getChildren(self,moves):
		for x in moves :
			n = Node(move = x, parent = self, board = self.board, token = opposite(self.token))
			n.value = n.board.values[n.move[0]][n.move[1]]
			self.childNodes.append(n)

	def addChild(self,node):
		self.childNodes.append(node)

	def updateNode(self, result, board):
		self.visits += 1
		self.wins += result
		# self.value = board.values[self.move[0]][self.move[1]] + ((self.wins * 10) /self.visits)

	def __repr__(self):
		return "[Move: " + str(self.move) + " Wins/Visits: " + str(self.wins) + "/" + str(self.visits) + " Value: " + str(self.value) + "]"

	def __eq__(self,other):
		if self.move == other.move:
			if self.parentNode == other.parentNode:
				return True
			else: return False
		else: return False

################################################################################
"""Rule sets (Computer-based players may lose compatibility if used with new sets)"""
class Rules: 
	__metaclass__ = ABCMeta

	@abstractmethod
	def makeMove(self, board, move, token): pass

	@abstractmethod
	def checkForWin(self, board, move, token): pass

class Gomoku(Rules):

	def __init__(self): 
		pass

	def makeMove(self, board, move, token):
		board.data[move[0]][move[1]] = token
		board.update(move, token)

	#checks to see if a move is a winning move.
	def checkForWin(self, board, move, token):
		x = [[-1,-1],[-1,0],[-1,1],[0,1]]
		for item in x:
			n1 = extent(board, token, move, [move[0]+item[0],move[1]+item[1]])
			n2 = extent(board, token, move, [move[0]-item[0],move[1]-item[1]])
			if (n1 + n2) >= 4:
				return True
			else:
				continue
		return False
		
################################################################################
"""Extra Functions"""
#not currently used, but may be in the future
def getRules():
	s = raw_input("What game will you play? (Only Gomoku right now)\n")
	if s.lower() == "gomoku":
		return Gomoku()
	else:
		print "That is not a valid game!\n"
		return getRules()

#manages player type decisions
def getPlayerType(color):
	x = raw_input("What type of player will " + color + " be? (Human, Computer, MonteCarlo, or Dummy)\n")
	if x.lower() == "human":
		return Human(color)
	elif x.lower() == "computer":
		return Computer(color)
	elif x.lower() == "montecarlo":
		return MonteCarlo(color)
	elif x.lower() == "dummy":
		return Dummy(color)
	else:
		print "That is not a valid player type\n"
		return getPlayerType(color)

#determines how many stones of type token are in a direction of a move location.
def extent(board, token, m1, m2): 
	if board.isWithinBounds(m2) and board.data[m2[0]][m2[1]] == token:
		return 1 + extent(board, token, m2, [2 * m2[0]-m1[0],2 * m2[1]-m1[1]])
	else:
		return 0

#determines if typed object is a number
def isNumber(s): 
    try:
        int(s)
        return True
    except ValueError:
        return False
                
def incrementPoints(p):
	return (p * 8) + 1

def decrementPoints(p):
	return (p - 1) / 8
	
################################################################################
"""Previous versions of code (flawed/bugged code)

updateDirection1
		
	V 1.0				
	def updateDirection1(self, m1, m2, token):
		change1 = [m2[0] - m1[0], m2[1] - m1[1]]
		change2 = [m1[0] - m2[0], m1[1] - m2[1]]
		next = [m1[0] + change2[0], m1[1] + change2[1]]
		r1 = self.range(m2, change1, token)
		r2 = self.range(next, change2, token)
		print m2, r1, r2
		counter = 1
		points = 1
		done = False
		while counter < 5 and self.isWithinBounds(m2):
			far = [m2[0] - (5 * change1[0]), m2[1] - (5 * change1[1])]
			if self.isWithinBounds(far) and (4 - counter) < r2:
				self.values[far[0]][far[1]] += points
			if done:
				m2 = [m2[0] + change1[0], m2[1] + change1[1]]
				counter += 1
				continue
			elif self.data[m2[0]][m2[1]] == token:
				points = incrementPoints(points)
			elif self.data[m2[0]][m2[1]] == opposite(token):
				done = True
			
			m2 = [m2[0] + change1[0], m2[1] + change1[1]]
			counter += 1
		m2 = next
		done = False
		counter = 1
		points = 1
		while counter < 5 and self.isWithinBounds(m2):
			far = [m2[0] - (5 * change2[0]), m2[1] - (5 * change2[1])]
			if self.isWithinBounds(far) and (4 - counter) < r1:
				self.values[far[0]][far[1]] += points
			if done:
				m2 = [m2[0] + change2[0], m2[1] + change2[1]]
				counter += 1
				continue
			elif self.data[m2[0]][m2[1]] == token:
				points = incrementPoints(points)
			elif self.data[m2[0]][m2[1]] == opposite(token):
				done = True
			
			m2 = [m2[0] + change2[0], m2[1] + change2[1]]
			counter += 1
			
	V 2.0
	def updateDirection1(self, m1, m2, token):
		change = [m2[0] - m1[0], m2[1] - m1[1]]
		counter = 0
		points = 1
		list = []
		while counter < 4 and self.isWithinBounds(m2) and self.data[m2[0]][m2[1]] != opposite(token):
			if self.data[m2[0]][m2[1]] == '+':
				list.append(m2)
			if self.data[m2[0]][m2[1]] == token:
				points = (points * 2) + 1
			m2 = [m2[0] + change[0], m2[1] + change[1]]
			counter += 1
		m2 = [m1[0] - change[0], m1[1] - change[1]]
		counter = 0
		while counter < 4 and self.isWithinBounds(m2) and self.data[m2[0]][m2[1]] != opposite(token):
			if self.data[m2[0]][m2[1]] == '+':
				list.append(m2)
			if self.data[m2[0]][m2[1]] == token:
				points += (points * 2) + 1
			m2 = [m2[0] - change[0], m2[1] - change[1]]
			counter += 1
		for move in list:
			self.values[move[0]][move[1]] += points
			
			
updateDirection2

	V 1.0
	def updateDirection2(self, m1, m2, token):
		change = [m2[0] - m1[0], m2[1] - m1[1]]
		counter = 0
		points = 0
		while counter < 4 and self.isWithinBounds(m2) and self.data[m2[0]][m2[1]] != opposite(token):
			if self.data[m2[0]][m2[1]] == token:
				points = incrementPoints(points)
			far = [m2[0] - (4 * change[0]), m2[1] - (4 * change[1])]
			if self.isWithinBounds(far):
				self.values[far[0]][far[1]] -= points
			m2 = [m2[0] + change[0], m2[1] + change[1]]
			counter += 1
		m2 = [m1[0] - change[0], m1[1] - change[1]]
		counter = 0
		points = 0
		while counter < 4 and self.isWithinBounds(m2) and self.data[m2[0]][m2[1]] != opposite(token):
			if self.data[m2[0]][m2[1]] == token:
				points = incrementPoints(points)
			far = [m2[0] + (4 * change[0]), m2[1] + (4 * change[1])]
			if self.isWithinBounds(far):
				self.values[far[0]][far[1]] -= points
			m2 = [m2[0] - change[0], m2[1] - change[1]]
			counter += 1
"""
        

