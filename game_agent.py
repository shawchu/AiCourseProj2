"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    #  copied from the sample_players.py
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)
    
    raise NotImplementedError


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            pass

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        
        gscore, gmove = self.minimax(game, 3, maximizing_player=True)
        return gmove
        #return self.minimax(game, 3, maximizing_player=True)
        
        raise NotImplementedError

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        
        #  To start with, just use the same GreedPlayer method for now.
        
        #print("blah")
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            return (-1, -1)
        
        bscore = 0.0
        bscorelist = []
        bdict = {}
        move = legal_moves[0]
        cdepth = depth
        
        
        #  going to assume that depth will be passed recursively as the minimax function
        #   is assumed to be called recursively as many times as (depth-1)
        #  When reached 2nd to last layer, can run calculations on last layer
        while depth > 2:
            depth -= 1
            for m in legal_moves:
                tboard = game.copy()
                tboard.apply_move(m)
                #  After applying the move, the active player will change to other player for
                #   next player down
                #print(tboard.__active_player__)
                #break
                
                #  Start recursing the minimax from here
                tscore, move = self.minimax(tboard, depth, not maximizing_player)
                print("tscore ", tscore)
                bdict.update({m : tscore})
                
                #if tscore > bscore:
                #    bscorelist.append(tscore)
            
            print("bdict ", bdict)    
            move = max(bdict, key=bdict.get)
            tscore = bdict[move]
            return tscore, move
        
        #  should only get to below when have recursed down to 2nd to last layer        
        print(game.__active_player__)
        print("maxi  ", maximizing_player)
        print(game.to_string())
        
        print(legal_moves)
#        for m in legal_moves:
#            
#            #  The score function is still from point of view of CustomPlayer()
#            tscore, move = max([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
#            print("lowest level tscore ", tscore, "move  ", move)
        
        #  The score function is still from point of view of CustomPlayer()
        if maximizing_player:
            tscore, move = max([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
        else:
            tscore, move = min([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
#        tscore, move = max([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
#        print("lowest level tscore max ", tscore, "move  ", move)
#        tscore, move = min([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
#        print("lowest level tscore min ", tscore, "move  ", move)
        return tscore, move
    
#==============================================================================
#         for m in legal_moves:
#             print("legal_move ", m)
#             tboard = game.copy()
#             tboard.apply_move(m)
#             print("tboard")
#             print(tboard.to_string())
#             tscore, tmove = max([(self.score(tboard.forecast_move(m1), self), m1) for m1 in tboard.get_legal_moves()] or [(0,0)])
#             #tscore = min([(self.score(tboard.forecast_move(m1), self)) for m1 in tboard.get_legal_moves()] or [0])
#             print("tscore ", tscore)
#             if tscore > bscore:
#                 bscore = tscore
#                 move = m
#                 print("tmove ", tmove)
#                 
#         print("final move ", move) 
#         
#         #_, move = max([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
#         return move
#         
#==============================================================================
        raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        raise NotImplementedError
