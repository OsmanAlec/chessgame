from pieces import *
from board import *

class GameManager(object):
    """Game logic, checking for checkmates is centralised here.
    Game settings, timer, menu, everything will be added here."""
    def __init__(self):
        self.board = Board()
    
    def startGame(self):
        """Contains the main game loop."""

        print("Welcome to the game of chess")
        while not self.isCheckMate():
            self.board.printWithNotation()
            print(f"{self.board.turn}'s turn.")
            start_pos, end_pos = self.getInput()
            self.board.movePiece(start_pos, end_pos)

    def isCheckMate(self):
        return False

    def getInput(self):
        usr_input = input()
        try:
            start, end = usr_input.split(" ")
        except ValueError as e:
            print("Check your input. Correct format is start and end squares. Use usual chess notation as you see it.")
            return self.getInput()

        start_col = ord(start[0]) - ord('a')
        end_col = ord(end[0]) - ord('a')

        start_row = 8 - int(start[1])
        end_row = 8 - int(end[1])

        start_pos = (start_row, start_col)
        end_pos = (end_row, end_col)

        if not all(0 <= x <= 7 for x in (*start_pos, *end_pos)):
            print("Invalid square. Use coordinates a1-h8.")
            return self.getInput()

        return start_pos, end_pos



game_manager = GameManager()
game_manager.startGame()