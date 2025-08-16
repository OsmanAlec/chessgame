from enum import Enum

class Colour(Enum):
    WHITE = 1
    BLACK = 2

    def __str__(self):
        return "White" if self == Colour.WHITE else "Black"

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
        # universal checks
        if self.position == new_position:
            print("Cannot move in place.")
            return False
        
        target_piece = board.getPieceAt(new_position)
        if target_piece and target_piece.colour == self.colour:
            print("Cannot land on your own piece.")
            return False
        
        return self._isValidSpecific(new_position, board)

    def _isValidSpecific(self, new_position, board):
        """To be implemented by each subclass."""
        raise NotImplementedError
    
    def __str__(self):
        return self.symbol

class Pawn(Piece):
    """Represents a Pawn piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'P' if colour == Colour.WHITE else 'p'

    def _isValidSpecific(self, new_position, board):
        
        start_row, start_col = self.position
        end_row, end_col = new_position
        
        # Pawns cannot move backward or in place
        direction = -1 if self.colour == Colour.WHITE else 1
        if (end_row - start_row) * direction < 0:
            print("Wrong Direction")
            return False
            
        # Forward Movement
        if start_col == end_col:
            # Single move forward
            if end_row == start_row + direction and not board.getPieceAt(new_position):
                return True
            
            # Double move forward
            start_rank = 6 if self.colour == Colour.WHITE else 1
            if start_row == start_rank and end_row == start_row + 2 * direction and board.isPathClear(self.position, new_position):
                return True
                
        # Captures
        if abs(end_col - start_col) == 1 and end_row == start_row + direction:
            target_piece = board.getPieceAt(new_position)
            if target_piece and target_piece.colour != self.colour:
                return True
                
        return False


class Rook(Piece):
    """Represents a Rook piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'R' if colour == Colour.WHITE else 'r'

    def _isValidSpecific(self, new_position, board):
        start_row, start_col = self.position
        end_row, end_col = new_position

        if board.isPathClear(self.position, new_position) and (start_row == end_row or start_col == end_col):
            return True
        
        return False


class Knight(Piece):
    """Represents a Knight piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'N' if colour == Colour.WHITE else 'n'

    def _isValidSpecific(self, new_position, board):
      
        start_row, start_col = self.position
        end_row, end_col = new_position

        row_diff = abs(end_row - start_row)
        col_diff = abs(end_col - start_col)

        # Must move in L-shape
        if (row_diff, col_diff) in [(2, 1), (1, 2)]:
            return True

        return False

        

class Bishop(Piece):
    """Represents a Bishop piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'B' if colour == Colour.WHITE else 'b'

    def _isValidSpecific(self, new_position, board):
        start_row, start_col = self.position
        end_row, end_col = new_position

        row_diff = abs(start_row - end_row)
        col_diff = abs(start_col - end_col)

        if board.isPathClear(self.position, new_position) and row_diff == col_diff:
            return True
        
        return False

class Queen(Piece):
    """Represents a Queen piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'Q' if colour == Colour.WHITE else 'q'

    def _isValidSpecific(self, new_position, board):  
        start_row, start_col = self.position
        end_row, end_col = new_position

        row_diff = abs(start_row - end_row)
        col_diff = abs(start_col - end_col)

        if board.isPathClear(self.position, new_position) and (
            row_diff == col_diff or start_row == end_row or start_col == end_col
            ):
            return True
        
        return False

class King(Piece):
    """Represents the King piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'K' if colour == Colour.WHITE else 'k'

    def _isValidSpecific(self, new_position, board):
        start_row, start_col = self.position
        end_row, end_col = new_position

        row_diff = abs(start_row - end_row)
        col_diff = abs(start_col - end_col)

        return max(row_diff, col_diff) == 1