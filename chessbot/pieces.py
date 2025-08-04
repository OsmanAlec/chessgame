from enum import Enum

class Colour(Enum):
    WHITE = 1
    BLACK = 2

class Piece(object):
    """A base class for all chess pieces."""
    def __init__(self, colour, position):
        self.colour: Colour = colour
        self.position: tuple = position # (row, col)
        self.symbol = ' '

    def isValidMove(self, new_position, board):
        """
        This method will be implemented by each specific piece class.
        It should return True if the move is valid, False otherwise.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def __str__(self):
        return self.symbol

class Pawn(Piece):
    """Represents a Pawn piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'P' if colour == Colour.WHITE else 'p'

    def isValidMove(self, new_position, board):
        start_row, start_col = self.position
        end_row, end_col = new_position
        
        # Pawns cannot move backward or in place
        direction = -1 if self.colour == Colour.WHITE else 1
        if (end_row - start_row) * direction >= 0:
            return False
            
        # Forward Movement
        if start_col == end_col:
            # Single move forward
            if end_row == start_row + direction and board.isSquareEmpty(new_position):
                return True
            
            # Double move forward
            start_rank = 6 if self.colour == Colour.WHITE else 1
            if start_row == start_rank and end_row == start_row + 2 * direction and board.isPathClear(self.position, new_position):
                return True
                
        # Captures
        if abs(end_col - start_col) == 1 and end_row == start_row + direction:
            if board.getPieceAt(new_position) and board.getPieceAt(new_position).colour != self.colour:
                return True
                
        return False


class Rook(Piece):
    """Represents a Rook piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'R' if colour == Colour.WHITE else 'r'

    def isValidMove(self, new_position, board):
        pass


class Knight(Piece):
    """Represents a Knight piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'N' if colour == Colour.WHITE else 'n'

    def isValidMove(self, new_position, board):
        pass

class Bishop(Piece):
    """Represents a Bishop piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'B' if colour == Colour.WHITE else 'b'

    def isValidMove(self, new_position, board):
        pass

class Queen(Piece):
    """Represents a Queen piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'Q' if colour == Colour.WHITE else 'q'

    def isValidMove(self, new_position, board):
        pass

class King(Piece):
    """Represents the King piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'K' if colour == Colour.WHITE else 'k'

    def isValidMove(self, new_position, board):
        pass