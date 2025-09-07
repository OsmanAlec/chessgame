from pieces import *
from board import *

class GameManager(object):
    """Game logic, checking for checkmates is centralised here.
    Game settings, timer, menu, everything will be added here."""
    def __init__(self):
        self.board = Board()
    
    def restartBoard(self):
        """Restarts the board"""

        self.board = Board ()

        
    
    def isCheckMate(self):
        colour = self.board.turn
        if not self.board.isKingInCheck(colour):
            return False
        
        # If king is in check, see if ANY legal move exists
        for row in self.board.spaces:
            for piece in row:
                if piece and piece.colour == colour:
                    if self.board.getLegalMoves(piece):
                        return False
        return True
    
    def isStaleMate(self):
        colour = self.board.turn
        if self.board.isKingInCheck(colour):
            return False
        
        # If king is in not check, see if ANY legal move exists
        for row in self.board.spaces:
            for piece in row:
                if piece and piece.colour == colour:
                    if self.board.getLegalMoves(piece):
                        return False
        return True


    def getInput(self):
        usr_input = input()
        try:
            start, end = usr_input.split(" ")
        
            start_col = ord(start[0]) - ord('a')
            end_col = ord(end[0]) - ord('a')

            start_row = 8 - int(start[1])
            end_row = 8 - int(end[1])

        except ValueError as e:
            print("Check your input. Correct format is start and end squares.\nFor example to move white's pawn write: d2 d4.")
            return self.getInput()

        start_pos = (start_row, start_col)
        end_pos = (end_row, end_col)

        if not all(0 <= x <= 7 for x in (*start_pos, *end_pos)):
            print("Invalid square. Use coordinates a1-h8.")
            return self.getInput()

        return start_pos, end_pos

