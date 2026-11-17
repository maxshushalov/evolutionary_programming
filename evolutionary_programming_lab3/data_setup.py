import json
import random
from pathlib import Path
from game import legal, winner

ROOT = Path(__file__).resolve().parent


def positions(count, seed):
    rng = random.Random(seed)
    found = {}
    attempts = 0
    while len(found) < count:
        attempts += 1
        if attempts > 200000:
            raise RuntimeError('Недостаточно разных позиций')
        board = [0] * 9
        turn = 1
        for _ in range(rng.randrange(0, 7)):
            if winner(board) or not legal(board):
                break
            board[rng.choice(legal(board))] = turn
            turn = -turn
        if not winner(board) and len(legal(board)) >= 2:
            found[tuple(board)] = {'board': board, 'player': turn}
    return list(found.values())


def manifest(repetitions, seed):
    rng = random.Random(seed)
    return [
        {'opponent': opponent, 'sign': sign, 'agent_seed': rng.randrange(2**32), 'opponent_seed': rng.randrange(2**32)}
        for opponent in ('random', 'tactical', 'fixed')
        for sign in (1, -1)
        for _ in range(repetitions)
    ]


def create():
    train = {'positions': positions(12, 290026), 'matches': manifest(2, 290027)}
    test = {'positions': positions(24, 290028), 'matches': manifest(10, 290029)}
    train_boards = {tuple(row['board']) for row in train['positions']}
    test['positions'] = [row for row in positions(100, 290028) if tuple(row['board']) not in train_boards][:24]
    for name, data in (('train', train), ('test', test)):
        (ROOT / 'data' / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    create()
