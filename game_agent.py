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

#    own_moves = len(game.get_legal_moves(player))
#    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
#    return float(own_moves - opp_moves)
    
    #  heuristic number 1
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    centrearea = [(2,2), (2,3), (2,4), (3,2), (3,3), (3,4), (4,2), (4,3), (4,4)]
    own_posit = game.get_player_location(player)
    if own_posit in centrearea:
        return float(own_moves - opp_moves) + 2
    else:
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
        
        gscore = float("-inf")
        gmove = (-1,-1)
        
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            
            if self.iterative:
                
                
                cdepth = 0
                while cdepth < 1e3:
                    gscore, gmove = self.minimax(game, int(cdepth), True)
                    #print("   cdepth=", cdepth, " gscore=", gscore, " gmove=", gmove)
                    cdepth += 1
                    if self.time_left() < self.TIMER_THRESHOLD:
                        return gmove
            
            else:
                gscore = float("-inf")
                gmove = (-1,-1)
                #print("game=", game)
                #print("search_depth=", self.search_depth)
                gscore, gmove = self.minimax(game, self.search_depth, True)
                #print(" gscore=", gscore, " gmove=", gmove)
                return gmove
            
            #gscore, gmove = self.minimax(game, 1, maximizing_player=True)


        except Timeout:
            # Handle any actions required at timeout, if necessary
            #  going to rely on timeout function to arrive here
            if gmove == (-1, -1):
                return legal_moves[0]
            else:
                return gmove
            #pass

        # Return the best move from the last completed search iteration
        
        
        #return gmove
        
    
        #raise NotImplementedError

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
        legal_moves = game.get_legal_moves()
        
        #def depth_search(self, game, depth, maximizing_player=True):
#            tscore = float("-inf")
#            move = (-1,-1)
            
            #print("legal_moves=", legal_moves, " depth=", depth)
                    
        if not legal_moves or depth <= 0:
            #print("depth= ", depth, " ", self, " score=", self.score(game, self), " posit=", game.get_player_location(self))
#            if maximizing_player:
#                tscore = self.score(game, game.__active_player__)
#            else:
#                tscore = self.score(game, game.__inactive_player__)
            
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()            

            tscore = self.score(game, self)
            #print(game.to_string())
            #print("depth 0 score=", tscore)
            return tscore, (-1, -1)
            #return self.score(game, game.__inactive_player__), (-1,-1)
            #tscore, move = max([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
            
            #tscore = self.score(game, game.__active_player__)
            #move = legal_moves[0]
            #return tscore, move
        #print("depth=", depth)
        while depth > 0:
            #tscore, move = max([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
            
            #print("max_player=", maximizing_player)
            #print("depth=", depth, " legal_moves=", legal_moves)
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
                
            if maximizing_player:
                tscore = float("-inf")
                move = (-1,-1)
                for m in legal_moves:
                    #tboard = game.copy()
                    #tboard.apply_move(m)
                    #print("m=", m)
                    tboard = game.forecast_move(m)
                    v, move_m = self.minimax(tboard, depth-1, not maximizing_player)
                    #tdict.update({m : v})
                    if v > tscore:
                        tscore = v
                        move = m
                #move = max(tdict, key=tdict.get)
                #tscore = tdict[move]
                #print("depth =", depth, " tscore = ", tscore, " move=", move)
                return tscore, move

            else:
                tdict={}
                tscore = float("inf")
                move = (-1,-1)
                for m in legal_moves:
                    #tboard = game.copy()
                    #tboard.apply_move(m)
                    tboard = game.forecast_move(m)
                    v, move_m = self.minimax(tboard, depth-1, not maximizing_player)
                    #tdict.update({m : v})
                    if v < tscore:
                        tscore = v
                        move = m
                #move = min(tdict, key=tdict.get)
                #tscore = tdict[move]
                #print("depth =", depth, " tscore = ", tscore, " move=", move)
                return tscore, move
            
        #depth_search(self, game, depth, maximizing_player=True)
        


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
        


        def max_val(self, game, depth, alpha=float("-inf"), beta=float("inf")):               
            legal_moves = game.get_legal_moves()
            #print("maxdepth=", depth, " max_val moves=", legal_moves)
            if not legal_moves or depth <= 0:
                tscore = self.score(game, game.__active_player__)
                return tscore, (-1, -1)
            #else:
            while depth > 0:
                tscore = float("-inf")
                v = float("-inf")
                #m = (-1,-1)
                move = (-1,-1)
                #print("legal_moves=", legal_moves)
                for m in legal_moves:
                    tboard = game.forecast_move(m)
                    v, move_m = min_val(self, tboard, depth-1, alpha, beta)
                    #tdict.update({m : v})
                    if v > tscore:
                        tscore = v
                        move = m
                    if tscore >= beta:
                        return tscore, move
                        #break
                    alpha = max(alpha, v)
                        
                        
                    #print("alpha=", alpha)
          
                    #move = max(tdict, key=tdict.get)
                    #tscore = tdict[move]
                    #tscore = v
                return tscore, move

        def min_val(self, game, depth, alpha=float("-inf"), beta=float("inf")):               
            legal_moves = game.get_legal_moves()
            #print("mindepth=", depth, " min_val moves=", legal_moves)
            if not legal_moves or depth <= 0:
                tscore = self.score(game, game.__inactive_player__)
                return tscore, (-1, -1)
            #else:
            while depth > 0:
                tscore = float("inf")
                v = float("inf")
                #m = (-1,-1)
                move = (-1,-1)
                for m in legal_moves:
                    #tboard = game.copy()
                    #tboard.apply_move(m)
                    tboard = game.forecast_move(m)
                    v, move_m = max_val(self, tboard, depth-1, alpha, beta)
                    #tdict.update({m : v})
                    if v < tscore:
                        tscore = v
                        move = m
                    if tscore <= alpha:
                        return tscore, move
                    beta = min(beta, v)
                    #if v < beta:
                    #    move = m
                
                #move = min(tdict, key=tdict.get)
                #tscore = tdict[move]
                return tscore, move
        
        #print("abdepth=", depth)    
        abval, abmove = max_val(self, game, depth, float("-inf"), float("inf"))
        return abval, abmove
        

        
        #raise NotImplementedError
