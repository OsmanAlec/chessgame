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

    def getPossibleMoves(self, board):
        """
        This method will be implemented by each specific piece class.
        It should return True if the move a list of moves.
        """
        raise NotImplementedError
            
    def __str__(self):
        return self.symbol

class Pawn(Piece):
    """Represents a Pawn piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'P' if colour == Colour.WHITE else 'p'


    def getAttackSquares(self):
        attacks = []
        x, y = self.position
        direction = -1 if self.colour == "white" else 1

        for dy in [-1, 1]:
            new_x, new_y = x + direction, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                attacks.append((new_x, new_y))
        return attacks

    def getPossibleMoves(self, board):
        moves = []
        
        x,y = self.position
        
        direction = -1 if self.colour == Colour.WHITE else 1
        start_row = 6 if self.colour == Colour.WHITE else 1

        if board.getPieceAt((x + direction, y)) is None:
            moves.append((x + direction, y))
            
            # 2-step forward
            if x == start_row and board.getPieceAt((x + 2*direction, y)) is None:
                moves.append((x + 2*direction, y))
            
        # Diagonal captures
        for dy in [-1, 1]:
            new_x, new_y = x + direction, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                target = board.getPieceAt((new_x, new_y))
                if target and target.colour != self.colour:
                    moves.append((new_x, new_y))
                
        for dy in [-1, 1]:
            new_x, new_y = x + direction, y + dy
            if (new_x, new_y) == board.enpassant_target:
                moves.append((new_x, new_y))

        return moves


class Rook(Piece):
    """Represents a Rook piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'R' if colour == Colour.WHITE else 'r'
        self.has_moved = False

    def getPossibleMoves(self, board):
        moves = []

        # Check in 4 directions until hit a piece
        for direction in [(1,0), (-1,0), (0,1), (0,-1)]:
            x, y = self.position

            while True:
                x += direction[0]
                y += direction[1]
                if not (0 <= x < 8 and 0 <= y < 8):
                    break
                if board.getPieceAt((x,y)):
                    if board.getPieceAt((x,y)).colour != self.colour:
                        moves.append((x,y)) # capture
                    break
                moves.append((x,y))
        return moves


class Knight(Piece):
    """Represents a Knight piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'N' if colour == Colour.WHITE else 'n'

    def getPossibleMoves(self, board):
      
        moves = []
        x, y = self.position

        directions = [
            (2, 1), (2, -1),
            (-2, 1), (-2, -1),
            (1, 2), (1, -2),
            (-1, 2), (-1, -2)
        ]

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                target = board.getPieceAt((new_x, new_y))
                if target is None or target.colour != self.colour:
                    moves.append((new_x, new_y))

        return moves
        

class Bishop(Piece):
    """Represents a Bishop piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'B' if colour == Colour.WHITE else 'b'

    def getPossibleMoves(self, board):
        moves = []

        for direction in [(-1, -1), (1, 1), (-1, 1), (1, -1)]:
            x, y = self.position
            while True:
                x += direction[0]
                y += direction[1]
                if not (0 <= x < 8 and 0 <= y < 8):
                    break
                if board.getPieceAt((x,y)):
                    if board.getPieceAt((x,y)).colour != self.colour:
                        moves.append((x,y)) # capture
                    break
                moves.append((x,y))

        return moves


class Queen(Piece):
    """Represents a Queen piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'Q' if colour == Colour.WHITE else 'q'

    def getPossibleMoves(self, board):  
        moves = []
        
        # Check in 4 directions until hit a piece
        for direction in [(1,0), (-1,0), (0,1), (0,-1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            x, y = self.position
            while True:
                x += direction[0]
                y += direction[1]
                if not (0 <= x < 8 and 0 <= y < 8):
                    break
                if board.getPieceAt((x,y)):
                    if board.getPieceAt((x,y)).colour != self.colour:
                        moves.append((x,y)) # capture
                    break
                moves.append((x,y))
        return moves

class King(Piece):
    """Represents the King piece."""
    def __init__(self, colour, position):
        super().__init__(colour, position)
        self.symbol = 'K' if colour == Colour.WHITE else 'k'
        self.has_moved = False


    def getPossibleMoves(self, board):
        moves = []
        x, y = self.position

        for dx, dy in [
            (1,0), (1,1), (0,1), (-1,0),
            (-1,-1), (0,-1), (1,-1), (-1,1)
        ]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                target = board.getPieceAt((nx, ny))
                if target is None or target.colour != self.colour:
                    moves.append((nx, ny))

        # Assume (row, col) where row = rank, col = file
        if not self.has_moved:
            row, col = self.position
            
            # Kingside castling
            rook = board.getPieceAt((row, 7))
            if isinstance(rook, Rook) and not rook.has_moved:
                if all(board.getPieceAt((row, c)) is None for c in [5, 6]):
                    if all(not board.squareAttackedByOpponent((row, c), self.colour) for c in [4, 5, 6]):
                        moves.append((row, 6))

            # Queenside castling
            rook = board.getPieceAt((row, 0))
            if isinstance(rook, Rook) and not rook.has_moved:
                if all(board.getPieceAt((row, c)) is None for c in [1, 2, 3]):
                    if all(not board.squareAttackedByOpponent((row, c), self.colour) for c in [2, 3, 4]):
                        moves.append((row, 2))



        return moves