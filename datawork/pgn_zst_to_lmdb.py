#!python
"""
Stream .pgn.zst -> LMDB of (binary, eval). Each record stores:
 key: 8-byte big-endian integer (uint64)
 value: eval(float32, 4 bytes) + 768 bytes board binary = 772 bytes
 Optionally write an augmented flipped copy (same key+suffix).
 Acknowledgment of LLM use: coded with the help of AI.
"""
import io, struct, argparse
import zstandard as zstd
import chess.pgn
import lmdb
import numpy as np
from tqdm import tqdm

# pack float32 eval then bytes
def board_to_binary_bytes(board):
    planes = np.zeros((12, 8, 8), dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        plane_idx = (piece.piece_type - 1) + (0 if piece.color else 6)
        rank = 7 - (sq // 8)
        file = sq % 8
        planes[plane_idx, rank, file] = 1
    return planes.tobytes()

def parse_eval_from_comment(comment):
    if not comment:
        return None
    # very small regex-free parse: find "[%eval " and parse until ]
    i = comment.find("[%eval")
    if i == -1:
        return None
    j = comment.find("]", i)
    token = comment[i+6:j].strip()
    # token could be "#3" mate or float
    if token.startswith("#"):
        sign = -1.0 if token.startswith("#-") else 1.0
        return sign * 10000.0
    try:
        return float(token)
    except:
        return None

def write_lmdb(input_zst, output_lmdb, map_size=int(2**40), batch_txn=1000, max_games=None):
    env = lmdb.open(output_lmdb, map_size=map_size)
    dec = zstd.ZstdDecompressor()
    key_counter = 0
    with open(input_zst, "rb") as fh, env.begin(write=True) as txn:
        with dec.stream_reader(fh) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace", newline="\n")
            games = 0
            while True:
                game = chess.pgn.read_game(text)
                if game is None:
                    break
                games += 1
                board = game.board()
                ply = 0
                for node in tqdm(game.mainline(), desc=f"Game {games}", unit="move", leave=False):
                    if node.move is None:
                        continue
                    board.push(node.move)
                    ply += 1
                    eval_score = parse_eval_from_comment(node.comment)
                    if eval_score is None:
                        continue
                    b = board_to_binary_bytes(board)
                    # pack value: float32 + binary
                    val = struct.pack(">f", float(eval_score)) + b
                    key = key_counter.to_bytes(8, "big")
                    txn.put(key, val)
                    key_counter += 1


                    # periodic commit (to avoid huge txn)
                    if key_counter % batch_txn == 0:
                        txn.commit()
                        txn = env.begin(write=True)

                if max_games and games >= max_games:
                    break
            txn.commit()
    env.close()
    print("Done. wrote", key_counter, "records.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--map-size-gb", type=float, default=150.0, help="LMDB map_size in GB")
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()
    write_lmdb(args.input, args.output, map_size=int(args.map_size_gb * (1024**3)), max_games=args.max_games)
