from pieces import Colour, Piece, Pawn, Rook, Knight, Bishop, Queen, King 

class Board:
    def __init__(self):
        """Initializes an 8x8 chessboard with all pieces in their starting positions."""
        self.spaces = self._create_initial_board()
        self.captured = []
        self.turn = Colour.WHITE

    def _create_initial_board(self):
        """
        Helper method to create and populate the board with piece objects.
        """
        # Create empty board
        board = [[None for _ in range(8)] for _ in range(8)]

        # Place black back rank
        board[0][0] = Rook(Colour.BLACK, (0, 0))
        board[0][1] = Knight(Colour.BLACK, (0, 1))
        board[0][2] = Bishop(Colour.BLACK, (0, 2))
        board[0][3] = Queen(Colour.BLACK, (0, 3))
        board[0][4] = King(Colour.BLACK, (0, 4))
        board[0][5] = Bishop(Colour.BLACK, (0, 5))
        board[0][6] = Knight(Colour.BLACK, (0, 6))
        board[0][7] = Rook(Colour.BLACK, (0, 7))

        # Place black pawns
        for i in range(8):
            board[1][i] = Pawn(Colour.BLACK, (1, i))

        # Place white pawns
        for i in range(8):
            board[6][i] = Pawn(Colour.WHITE, (6, i))

        # Place white back rank
        board[7][0] = Rook(Colour.WHITE, (7, 0))
        board[7][1] = Knight(Colour.WHITE, (7, 1))
        board[7][2] = Bishop(Colour.WHITE, (7, 2))
        board[7][3] = Queen(Colour.WHITE, (7, 3))
        board[7][4] = King(Colour.WHITE, (7, 4))
        board[7][5] = Bishop(Colour.WHITE, (7, 5))
        board[7][6] = Knight(Colour.WHITE, (7, 6))
        board[7][7] = Rook(Colour.WHITE, (7, 7))

        return board

    def __str__(self):
        """
        Returns a string representation of the board.
        """
        board_string = ''
        for i, row in enumerate(self.spaces):
            row_symbols = []
            for piece in row:
                if piece:
                    row_symbols.append(str(piece))
                else:
                    row_symbols.append('.')
            board_string += " ".join(row_symbols) + "\n"
        return board_string

    def printWithNotation(self):
        """
        Prints the board with algebraic notation.
        """
        print('  a b c d e f g h')
        print(' +-----------------+')
        for i, row in enumerate(self.spaces):
            row_symbols = []
            for piece in row:
                if piece:
                    row_symbols.append(str(piece))
                else:
                    row_symbols.append('.')
            print(f'{8 - i}| {" ".join(row_symbols)} |{8 - i}')
        print(' +-----------------+')
        print('  a b c d e f g h')
    
    def isPathClear(self, start_pos: tuple, end_pos: tuple) -> bool:
        """
        Takes in two parameters: start and end position as tuples.
        Returns whether or not the straight path is clear.
        """
        # Get the direction
        x_dir = end_pos[0] - start_pos[0]
        y_dir = end_pos[1] - start_pos[1]
        

        if x_dir == 0 and y_dir == 0:
            raise ValueError("isPathClear called with identical start and end positions. This indicates a bug.")
        
        # Normalise directions into 1 or 0
        x_step = int(x_dir / abs(x_dir)) if x_dir != 0 else 0
        y_step = int(y_dir / abs(y_dir)) if y_dir != 0 else 0
        
        # Start one step ahead
        current_pos_x = start_pos[0] + x_step
        current_pos_y = start_pos[1] + y_step
        
        # Iterate over the path until end position
        while (current_pos_x, current_pos_y) != end_pos:
            if self.spaces[current_pos_x][current_pos_y] is not None:
                return False # A piece exists on current square
            current_pos_x += x_step
            current_pos_y += y_step
            
        return True
    
    def getPieceAt(self, position: tuple):
        x, y = position
        return self.spaces[x][y]
    
    def movePiece(self, start_pos, end_pos):
        """
        Attempts to move a piece from start_pos to end_pos.
        Returns True on a successful move, False otherwise.
        """
        piece_to_move = self.getPieceAt(start_pos)

        # Check if there's a piece to move
        if not piece_to_move:
            print("No piece at the starting position.\n")
            return False
            
        # Check if it's the correct turn
        if piece_to_move.colour != self.turn:
            print("Selected opponent's piece.\n")
            return False

        # Check if the move is valid
        if piece_to_move.isValidMove(end_pos, self):
            # Handle captures
            captured_piece = self.getPieceAt(end_pos)
            if captured_piece:
                self.captured.append(captured_piece)
                print(f"Captured {str(captured_piece)}!")

            # Perform the move
            self.spaces[start_pos[0]][start_pos[1]] = None
            self.spaces[end_pos[0]][end_pos[1]] = piece_to_move
            piece_to_move.position = end_pos
            
            # Switch turns
            self.turn = Colour.BLACK if self.turn == Colour.WHITE else Colour.WHITE
            
            return True
        else:
            print("Invalid move.\n")
            return False
