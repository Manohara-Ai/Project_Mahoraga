import random
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import torch

from alpha_zero import ResNet, MCTS

app = Flask('__name__')
CORS(app)

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count
        self.state = self.get_initial_state()
        self.game_over = False

    def __repr__(self):
        return "TicTacToe"

    def get_initial_state(self):
        self.player = random.choice([1, -1])
        self.turn = 'player' if self.player == 1 else 'bot'
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]
        
        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state
    
class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4
        self.state = self.get_initial_state()
        self.game_over = False
        
    def __repr__(self):
        return "ConnectFour"
        
    def get_initial_state(self):
        self.player = random.choice([1, -1])
        self.turn = 'player' if self.player == 1 else 'bot'
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state
    
    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.row_count
                    or c < 0 
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state
    
t3 = TicTacToe()
c4 = ConnectFour()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t3_model = ResNet(t3, 4, 64, device)
t3_optimizer = torch.optim.Adam(t3_model.parameters(), lr=0.001, weight_decay=0.0001)
t3_args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 2,
    'num_selfPlay_iterations': 500,
    'num_parallel_games': 100,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}
t3_model_path = f"./models/model_{t3}.pth"
t3_model.load_state_dict(torch.load(t3_model_path, map_location=device, weights_only=True))
t3_model.eval()
t3_mcts = MCTS(t3, t3_args, t3_model)

c4_model = ResNet(c4, 9, 128, device)
c4_optimizer = torch.optim.Adam(c4_model.parameters(), lr=0.001, weight_decay=0.0001)
c4_args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 2,
        'num_selfPlay_iterations': 500,
        'num_parallel_games': 100,
        'num_epochs': 4,
        'batch_size': 64,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
}
c4_model_path = f"./models/model_{c4}.pth"
c4_model.load_state_dict(torch.load(c4_model_path, map_location=device, weights_only=True))
c4_model.eval()
c4_mcts = MCTS(c4, c4_args, c4_model)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            data = request.get_json()
            game_name = data.get('game')

            if not game_name:
                return jsonify({"error": "Game not selected"}), 400

            return jsonify({"game": game_name, "status": "Game selected successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    return render_template('index.html')

@app.route('/<game_name>', methods=['GET', 'POST'])
def init_game(game_name):
    try:
        if game_name == 'tictactoe':
            t3.__init__()
            game_constants = {
                'turn': t3.turn,
                'player': t3.player,
                'bot': t3.get_opponent(t3.player),
            }
            if request.args.get('format') == 'json':
                return jsonify(game_constants)
            return render_template('tic_tac_toe.html')

        elif game_name == 'connect4':
            c4.__init__()
            game_constants = {
                'turn': c4.turn,
                'player': c4.player,
                'bot': c4.get_opponent(c4.player),
            }
            if request.args.get('format') == 'json':
                return jsonify(game_constants)
            return render_template('connect_4.html')
        
        else:
            return "Game not found", 404
        
    except Exception as e:
        return "An error occurred while initializing the game.", 500

@app.route('/<game_name>/player-move', methods=['POST'])
def player_move(game_name):
    try:
        if game_name == 'tictactoe':
            if t3.turn == 'player' and t3.game_over == False:
                data = request.get_json()
                cell = int(data['cell'])
                t3.state = t3.get_next_state(t3.state, cell, t3.player)
                value, t3.game_over = t3.get_value_and_terminated(t3.state, cell)
                if t3.game_over:
                    if value  == 1:
                         return jsonify({"message": "Player Wins! Refresh to continue playing!!", 
                                         "state": t3.state.tolist()
                                         }), 200
                    else:
                        return jsonify({"message": "Tie! Refresh to continue playing!!", 
                                        "state": t3.state.tolist()
                                        }), 200
                t3.turn = 'bot'
                return jsonify({
                        "message": f"Move registered : {cell}",
                        "state": t3.state.tolist(),
                    })
            else: 
                return jsonify({
                        "message": "Not player's turn"
                    })
        
        elif game_name == 'connect4':
            if c4.turn == 'player'  and c4.game_over == False:
                data = request.get_json()
                cell = int(data['cell'])
                c4.state = c4.get_next_state(c4.state, cell, c4.player)
                value, c4.game_over = c4.get_value_and_terminated(c4.state, cell)
                if c4.game_over:
                    if value  == 1:
                         return jsonify({"message": "Player Wins! Refresh to continue playing!!", 
                                         "state": c4.state.tolist()
                                         }), 200
                    else:
                        return jsonify({"message": "Tie! Refresh to continue playing!!", 
                                        "state": c4.state.tolist()
                                        }), 200
                c4.turn = 'bot'
                return jsonify({
                        "message": f"Move registered : {cell}",
                        "state": c4.state.tolist(),
                    })
            else:
                return jsonify({
                        "message": "Not player's turn"
                    })
        
        else:
            return "Game not found", 404
        
    except Exception as e:
        return "An error occurred while registering the player move.", 500

@app.route('/<game_name>/ai-move', methods=['POST'])
def ai_move(game_name):
    try:
        if game_name == 'tictactoe':
            if t3.turn == 'bot' and t3.game_over == False:
                neutral_state = t3.change_perspective(t3.state, t3.get_opponent(t3.player))
                mcts_probs = t3_mcts.search(neutral_state)
                action = np.argmax(mcts_probs)
                t3.state = t3.get_next_state(t3.state, action, t3.get_opponent(t3.player))
                value, t3.game_over = t3.get_value_and_terminated(t3.state, action)
                if t3.game_over:
                    if value  == 1:
                        return jsonify({"message": "Ai Wins! Refresh to continue playing!!", 
                                         "state": t3.state.tolist()
                                         }), 200
                    else:
                        return jsonify({"message": "Tie! Refresh to continue playing!!", 
                                        "state": t3.state.tolist()
                                        }), 200
                t3.turn = 'player'
                return jsonify({
                        "message": "Ai made it's move",
                        "state": t3.state.tolist(),
                    })
            else: 
                return jsonify({
                        "message": "Not player's turn"
                    })
        
        elif game_name == 'connect4':
            if c4.turn == 'bot' and c4.game_over == False:
                neutral_state = c4.change_perspective(c4.state, c4.player)
                mcts_probs = c4_mcts.search(neutral_state)
                action = np.argmax(mcts_probs)
                c4.state = c4.get_next_state(c4.state, action, c4.get_opponent(c4.player))
                value, c4.game_over = c4.get_value_and_terminated(c4.state, action)
                if c4.game_over:
                    if value  == 1:
                         return jsonify({"message": "Ai Wins! Refresh to continue playing!!", 
                                         "state": c4.state.tolist()
                                         }), 200
                    else:
                        return jsonify({"message": "Tie! Refresh to continue playing!!", 
                                        "state": c4.state.tolist()
                                        }), 200
                c4.turn = 'player'
                return jsonify({
                        "message": "Ai made it's move",
                        "state": c4.state.tolist(),
                    })
            else:
                return jsonify({
                        "message": "Not player's turn"
                    })
        
        else:
            return "Game not found", 404
        
    except Exception as e:
        return "An error occurred while registering the ai move.", 500
    
if __name__ == '__main__':
    app.run(debug=True)