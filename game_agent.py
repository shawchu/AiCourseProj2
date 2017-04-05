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

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    #  the calculation of number of moves of self vs opponent is
    #   copied from the sample_players.py
    #own_moves = len(game.get_legal_moves(player))
    #opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    #return float(own_moves - opp_moves)
    
    #  heuristic number 1
    #   consider that if end on the centre grid of squares 2 squares
    #   away from edge as desirable. Adds 2 to score if in centre area/grid
    def use_h1(game, player):
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
        
        centre_area = []
        #  have to check if game board is fact big enough for a centre area
        if game.width > 4 and game.height > 4:
            cwidth = list(range(2, (game.width - 2)))
            cheight = list(range(2, (game.height - 2)))
            for i in cwidth:
                for j in cheight:
                    centre_area.append((i, j))
            
            #  will only run below check if there is a centre area
            own_posit = game.get_player_location(player)
            if own_posit in centre_area:
                return float(own_moves - opp_moves) + 2
            else:
                return float(own_moves - opp_moves)
    
        else:
            return float(own_moves - opp_moves)


    #  heuristic number 2
    #   consider that if end up on outer band of squares  within 2 squares
    #   away from edge as being undesirable. Subtract 2 from score if in 
    #   this outer band close to board edge
    def use_h2(game, player):
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
        
        centre_area = []
        if game.width > 4 and game.width > 4:
            cwidth = list(range(2, (game.width - 2)))
            cheight = list(range(2, (game.height - 2)))
            for i in cwidth:
                for j in cheight:
                    centre_area.append((i, j))
        
        whole_grid = []
        for i in range(0, game.width):
            for j in range(0, game.height):
                whole_grid.append((i,j))
                
        out_band = [x for x in whole_grid if x not in centre_area]
        
        own_posit = game.get_player_location(player)
        if own_posit in out_band:
            return float(own_moves - opp_moves) - 2
        else:
            return float(own_moves - opp_moves)
    
        
    #  heuristic number 3
    #   consider that if one or more of own moves can block opponent's moves
    #   The number of own moves which could be possible blocking moves 
    #   is added to score 
    def use_h3(game, player):
        own_pmoves = game.get_legal_moves(player)
        opp_pmoves = game.get_legal_moves(game.get_opponent(player))
        blockmoves = len([x for x in own_pmoves if x in opp_pmoves])
        return float(len(own_pmoves) - len(opp_pmoves)) + blockmoves
    
    #  choose the heuristic to use here. Comment out the unused.
    #return use_h1(game, player)             
    #return use_h2(game, player)             
    return use_h3(game, player)             
                 
    #raise NotImplementedError


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

        # TODO: finish this function for the assignment

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        
        gscore = float("-inf")
        if len(legal_moves) > 0:
            gmove = legal_moves[0]
        else:
            gmove = (-1,-1)
        
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            
            #  following is opening book moves for use with heuristic 3
            #  if move is available in centre grid area, truncate the 
            #  legal moves to those only in the centre grid.
            centre_area = []
            if game.width > 4 and game.width > 4:
                cwidth = list(range(2, (game.width - 2)))
                cheight = list(range(2, (game.height - 2)))
                for i in cwidth:
                    for j in cheight:
                        centre_area.append((i, j))
            
                cent_moves = []
                for m in legal_moves:
                    if m in centre_area:
                        cent_moves.append(m)
                if len(cent_moves) > 0:
                    legal_moves = cent_moves
            
            if self.method == 'minimax':
                if self.iterative:
                    cdepth = 0
                    while cdepth < float("inf"):
                        gscore, gmove = self.minimax(game, int(cdepth), True)
                        #print("   cdepth=", cdepth, " gscore=", gscore, " gmove=", gmove)
                        cdepth += 1
                        if self.time_left() < self.TIMER_THRESHOLD:
                            return gmove
                else:
                    gscore, gmove = self.minimax(game, self.search_depth, True)
                    return gmove
                
            else:
                alpha_init = float("-inf")
                beta_init = float("inf")
                if self.iterative:
                    cdepth = 0
                    while cdepth < float("inf"):
                        gscore, gmove = self.alphabeta(game, int(cdepth), alpha_init, beta_init, True)
                        cdepth += 1
                        if self.time_left() < self.TIMER_THRESHOLD:
                            return gmove
                else:
                    gscore, gmove = self.alphabeta(game, self.search_depth, alpha_init, beta_init, True)
                    #print(" gscore=", gscore, " gmove=", gmove)
                    return gmove
            

        except Timeout:
            # Handle any actions required at timeout, if necessary
            #  going to rely on timeout function to arrive here
            #  going to rely on some move being available at this stage
            #   going to assume that enough time for move to be calculated
                
            return gmove

        # Return the best move from the last completed search iteration
        #  going to rely on above functions to return gmove as the best move
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

        # TODO: the below is function of minimax for assignment
        
        legal_moves = game.get_legal_moves()
        
        #  if no legal moves or when depth decrement to zero, will have
        #   reached last layer in depth
                   
        if not legal_moves or depth <= 0:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()            

            #  self is the game_agent.py, which is the CustomPlayer()
            #   so score always from point of view of CustomPlayer
            tscore = self.score(game, self)
            #print("depth= ", depth, " ", self, " score=", self.score(game, self), " posit=", game.get_player_location(self))
            return tscore, (-1, -1)

        #  While haven't reach last layer, going to keep on recursing itself
        #   and use the minimax function
        while depth > 0:
            #print("max_player=", maximizing_player)
            #print("depth=", depth, " legal_moves=", legal_moves)
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            if maximizing_player:
                tscore = float("-inf")
            else:
                tscore = float("inf")
            move = (-1,-1)
            for m in legal_moves:
                #  forecase_move will apply the move m
                tboard = game.forecast_move(m)
                #  as it applies the minimax for next layer, will flip the maximizing_player bool
                #   v, move_m is temporary holders as check if better than tscore, move
                v, move_m = self.minimax(tboard, depth-1, not maximizing_player)
                if maximizing_player:
                    if v > tscore:
                        tscore = v
                        move = m
                else:
                    if v < tscore:
                        tscore = v
                        move = m
            return tscore, move

                
#            if maximizing_player:
#                    if v > tscore:
#                        tscore = v
#                        move = m
#                return tscore, move
#
#            else:
#                tscore = float("inf")
#                move = (-1,-1)
#                for m in legal_moves:
#                    tboard = game.forecast_move(m)
#                    v, move_m = self.minimax(tboard, depth-1, not maximizing_player)
#                    if v < tscore:
#                        tscore = v
#                        move = m
#                return tscore, move
            


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
        #  based on psudeo code in AIMA and course github
        #   create a max_val & min_val function
        
        legal_moves = game.get_legal_moves()

        if not legal_moves or depth <= 0:
            tscore = self.score(game, self)
            return tscore, (-1, -1)
        
        if maximizing_player:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            while depth > 0:
                tscore = float("-inf")
                v = float("-inf")
                move = (-1,-1)
                for m in legal_moves:
                    tboard = game.forecast_move(m)
                    #  at this point the alphabeta func could be several layers down
                    #   going to rely on updated beta passed from prev layer
                    v, move_m = self.alphabeta(tboard, depth-1, alpha, beta, not maximizing_player)
                    #print("depth=", depth, " v=", v, " move_m=", move_m, " alpha=", alpha, " beta=", beta)
                    if v > tscore:
                        tscore = v
                        move = m
                    if tscore >= beta:
                        return tscore, move
                    # this updated alpha will be passed into the next minimising alphabeta
                    #  for next m in legal_moves
                    alpha = max(alpha, v)
                return tscore, move
        
        else:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            while depth > 0:
                tscore = float("inf")
                v = float("inf")
                move = (-1,-1)
                for m in legal_moves:
                    tboard = game.forecast_move(m)
                    v, move_m = self.alphabeta(tboard, depth-1, alpha, beta, not maximizing_player)
                    #print("depth=", depth, " v=", v, " move_m=", move_m, " alpha=", alpha, " beta=", beta)
                    #  as this is min layer, find lowest score for this layer
                    if v < tscore:
                        tscore = v
                        move = m
                    #  in effect, tscore is going to be passed up to next layer above, presumed to be max layer
                    #   so if less than alpha, will prune it out
                    if tscore <= alpha:
                        return tscore, move
                    #  similar to the max val above
                    #  this updated beta will be passed into the next maximising alphabeta layer
                    #  for next m in legal_moves
                    beta = min(beta, v)
                return tscore, move
         
        #   starting node is presumed to be maximizing node as it should be
        #   CustomPlayer's turn to move and make a decision
#        abval, abmove = self.alphabeta(game, depth, float("-inf"), float("inf"))
#        return abval, abmove
            
            
            
#        #  max_val for maximizing node
#        #   starting node is presumed to be maximizing node as it should be
#        #   CustomPlayer's turn to move and make a decision
#        def max_val(self, game, depth, alpha=float("-inf"), beta=float("inf")):               
#            if self.time_left() < self.TIMER_THRESHOLD:
#                raise Timeout()            
#            legal_moves = game.get_legal_moves()
#            #print("maxdepth=", depth, " max_val moves=", legal_moves)
#            if not legal_moves or depth <= 0:
#                tscore = self.score(game, game.__active_player__)
#                return tscore, (-1, -1)
#
#            while depth > 0:
#                tscore = float("-inf")
#                v = float("-inf")
#                move = (-1,-1)
#                #print("legal_moves=", legal_moves)
#                for m in legal_moves:
#                    tboard = game.forecast_move(m)
#                    #  at this point the alphabeta func could be several layers down
#                    #   going to rely on updated beta passed from prev layer
#                    v, move_m = min_val(self, tboard, depth-1, alpha, beta)
#                    if v > tscore:
#                        tscore = v
#                        move = m
#                    if tscore >= beta:
#                        return tscore, move
#                    # this updated alpha will be passed into the next min_val
#                    #  for next m in legal_moves
#                    alpha = max(alpha, v)
#                        
#                        
#                return tscore, move
#
#        def min_val(self, game, depth, alpha=float("-inf"), beta=float("inf")):               
#            if self.time_left() < self.TIMER_THRESHOLD:
#                raise Timeout()            
#            legal_moves = game.get_legal_moves()
#            if not legal_moves or depth <= 0:
#                tscore = self.score(game, game.__inactive_player__)
#                return tscore, (-1, -1)
#            while depth > 0:
#                tscore = float("inf")
#                v = float("inf")
#                move = (-1,-1)
#                for m in legal_moves:
#                    tboard = game.forecast_move(m)
#                    v, move_m = max_val(self, tboard, depth-1, alpha, beta)
#                    #  as this is min layer, find lowest score for this layer
#                    if v < tscore:
#                        tscore = v
#                        move = m
#                    #  in effect, tscore is going to be passed up to next layer above, presumed to be max layer
#                    #   so if less than alpha, will prune it out
#                    if tscore <= alpha:
#                        return tscore, move
#                    #  similar to the max val above
#                    #  this updated beta will be passed into the next max_val
#                    #  for next m in legal_moves
#                    beta = min(beta, v)
#                return tscore, move
#        
#        #print("abdepth=", depth)    
#        #   starting node is presumed to be maximizing node as it should be
#        #   CustomPlayer's turn to move and make a decision
#        abval, abmove = max_val(self, game, depth, float("-inf"), float("inf"))
#        return abval, abmove
        

        
        #raise NotImplementedError
