import math
import random

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
FIXED_AGENT = (3, 1.0, 1.0, 1.0, 1.0, 0.0)


def winner(board):
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def legal(board):
    return [i for i, value in enumerate(board) if value == 0]


def heuristic(board, player, genes):
    result = 0.0
    for line in LINES:
        own = sum(board[i] == player for i in line)
        other = sum(board[i] == -player for i in line)
        if other == 0 and own in (1, 2):
            result += genes[own]
        if own == 0 and other in (1, 2):
            result -= genes[other + 2]
    return result / 16


def search(board, turn, root, depth, genes, counter):
    counter[0] += 1
    won = winner(board)
    if won:
        return float(won * root)
    if not legal(board):
        return 0.0
    if depth == 0:
        return heuristic(board, root, genes)
    values = []
    for move in legal(board):
        board[move] = turn
        values.append(search(board, -turn, root, depth - 1, genes, counter))
        board[move] = 0
    return max(values) if turn == root else min(values)


def best_moves(board, player, genes):
    moves = legal(board)
    values = []
    counter = [0]
    for move in moves:
        board[move] = player
        values.append(search(board, -player, player, genes[0] - 1, genes, counter))
        board[move] = 0
    best = max(values)
    return moves, [move for move, value in zip(moves, values) if abs(value - best) <= 1e-12], counter[0]


def choose(board, player, genes, rng):
    moves, best, nodes = best_moves(board, player, genes)
    options = moves if rng.random() < genes[5] else best
    return rng.choice(options), nodes


def tactical(board, player, rng):
    moves = legal(board)
    for target in (player, -player):
        good = []
        for move in moves:
            board[move] = target
            if winner(board) == target:
                good.append(move)
            board[move] = 0
        if good:
            return rng.choice(good)
    return rng.choice(moves)


def play(genes, opponent, agent_sign, agent_seed, opponent_seed):
    board = [0] * 9
    agent_rng = random.Random(agent_seed)
    opponent_rng = random.Random(opponent_seed)
    turn = 1
    nodes = 0
    while legal(board) and winner(board) == 0:
        if turn == agent_sign:
            move, count = choose(board, turn, genes, agent_rng)
            nodes += count
        elif opponent == 'random':
            move = opponent_rng.choice(legal(board))
        elif opponent == 'tactical':
            move = tactical(board, turn, opponent_rng)
        else:
            move, _ = choose(board, turn, FIXED_AGENT, opponent_rng)
        board[move] = turn
        turn = -turn
    return winner(board) * agent_sign, nodes


def entropy(board, player, genes):
    moves, best, nodes = best_moves(board, player, genes)
    k, g = len(moves), len(best)
    if k == 1:
        return 0.0, nodes
    entropy_sum = 0.0
    for move in moves:
        probability = genes[5] / k + ((1 - genes[5]) / g if move in best else 0)
        if probability:
            entropy_sum -= probability * math.log(probability)
    return entropy_sum / math.log(k), nodes
