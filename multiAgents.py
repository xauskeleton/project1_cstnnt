# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Winning is the best thing that can happen, take it immediately.
        if successorGameState.isWin():
            return float('inf')

        # Never step onto, or next to, a ghost that can still kill us:
        # the ghost moves after Pacman, so distance 1 is already deadly.
        for ghostState in newGhostStates:
            ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
            if ghostState.scaredTimer == 0 and ghostDistance <= 1:
                return float('-inf')

        # Start from the game score: +10 per food eaten, -1 per time step.
        value = successorGameState.getScore()

        # Standing still only burns time, so make it the last resort.
        if action == Directions.STOP:
            value -= 10

        # Reciprocal of the distance to the closest food: the closer the
        # food, the larger the bonus.  It is at most 1, so it never
        # outweighs actually eating a dot (+10).
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        value += 1.0 / min(foodDistances)

        return value

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def minimaxValue(state, agentIndex, depth):
            """
            Value of `state` when it is `agentIndex`'s turn to move and
            `depth` full plies (Pacman + every ghost) have been played.
            """
            # Leaves: the game is over, or the depth limit has been reached.
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Who moves next.  When the last ghost has moved, one ply is
            # finished: it is Pacman's turn again and the depth goes up.
            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth = depth + 1

            values = [minimaxValue(state.generateSuccessor(agentIndex, action),
                                   nextAgent, nextDepth)
                      for action in state.getLegalActions(agentIndex)]

            # Pacman (agent 0) maximizes, every ghost minimizes.
            if agentIndex == 0:
                return max(values)
            return min(values)

        # Root: same as a max node, but we remember the action, not the value.
        nextAgent = 1 % numAgents
        nextDepth = 1 if nextAgent == 0 else 0
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            value = minimaxValue(gameState.generateSuccessor(0, action),
                                 nextAgent, nextDepth)
            if bestAction is None or value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def alphaBetaValue(state, agentIndex, depth, alpha, beta):
            """
            Minimax value of `state`, skipping branches that cannot matter.
            alpha: best value Pacman (max) can already guarantee on this path.
            beta:  best value the ghosts (min) can already guarantee.
            """
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth = depth + 1

            if agentIndex == 0:
                # Max node.
                value = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphaBetaValue(successor, nextAgent,
                                                      nextDepth, alpha, beta))
                    # A ghost above already has something better (smaller)
                    # than this, so it will never let the game come here.
                    # Strictly greater: we do not prune on equality.
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value

            # Min node.  Consecutive ghosts are all min nodes, so they share
            # the same alpha and each one keeps tightening beta.
            value = float('inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                value = min(value, alphaBetaValue(successor, nextAgent,
                                                  nextDepth, alpha, beta))
                # Pacman above already has something better (larger).
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

        # Root max node: keep the best action and keep raising alpha so the
        # later children can be pruned too.
        nextAgent = 1 % numAgents
        nextDepth = 1 if nextAgent == 0 else 0
        alpha = float('-inf')
        beta = float('inf')
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            value = alphaBetaValue(gameState.generateSuccessor(0, action),
                                   nextAgent, nextDepth, alpha, beta)
            if bestAction is None or value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def expectimaxValue(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth = depth + 1

            values = [expectimaxValue(state.generateSuccessor(agentIndex, action),
                                      nextAgent, nextDepth)
                      for action in state.getLegalActions(agentIndex)]

            # Pacman still picks his best move ...
            if agentIndex == 0:
                return max(values)
            # ... but a ghost picks uniformly at random, so its node is worth
            # the average (expected value) of its children, not the minimum.
            return sum(values) / len(values)

        nextAgent = 1 % numAgents
        nextDepth = 1 if nextAgent == 0 else 0
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            value = expectimaxValue(gameState.generateSuccessor(0, action),
                                    nextAgent, nextDepth)
            if bestAction is None or value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: The game score plus a few hand-weighted features of the
    state itself:
      - the score so far (food eaten, time spent, ghosts eaten, win/lose);
      - reciprocal of the distance to the closest food, so Pacman always
        has a direction to walk in even when no dot is within the search
        depth;
      - a penalty per remaining food and per remaining capsule, so that
        eating them is always better than standing next to them;
      - for a scared ghost that can still be reached in time, a bonus that
        grows as Pacman gets closer (eating it is worth 200 points);
      - for a dangerous ghost, a large penalty for being right next to it.
    """
    "*** YOUR CODE HERE ***"
    # Finished games: the score already contains +500 / -500.
    if currentGameState.isWin() or currentGameState.isLose():
        return currentGameState.getScore()

    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()

    value = currentGameState.getScore()

    # Food: pull towards the closest dot, and prefer states with fewer dots.
    closestFood = min(manhattanDistance(pacmanPos, food) for food in foodList)
    value += FOOD_DISTANCE_WEIGHT / closestFood
    value -= FOOD_COUNT_WEIGHT * len(foodList)

    # Capsules: each one left on the board costs points, so eating one pays.
    value -= CAPSULE_COUNT_WEIGHT * len(capsules)

    for ghostState in currentGameState.getGhostStates():
        ghostDistance = manhattanDistance(pacmanPos, ghostState.getPosition())
        if ghostState.scaredTimer > 0:
            # Only chase a scared ghost we can reach before it recovers.
            if ghostDistance < ghostState.scaredTimer:
                value += SCARED_GHOST_WEIGHT / (ghostDistance + 1)
        elif ghostDistance <= 1:
            # One step away from a live ghost: it can kill us on its move.
            value -= GHOST_DANGER_PENALTY

    return value

# Weights of the features used by betterEvaluationFunction.
FOOD_DISTANCE_WEIGHT = 5.0
FOOD_COUNT_WEIGHT = 4.0
CAPSULE_COUNT_WEIGHT = 20.0
SCARED_GHOST_WEIGHT = 100.0
GHOST_DANGER_PENALTY = 200.0

# Abbreviation
better = betterEvaluationFunction
