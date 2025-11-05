from flask import Flask, render_template
from flask_socketio import SocketIO, join_room, leave_room, emit
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory state (single-process only)
lobby_queue = []
games = {}          # room_id -> {board, turn, players, winner, draw}
player_room = {}    # sid -> room_id

WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)
]

def check_winner(board):
    for a,b,c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("join_queue")
def join_queue():
    sid = request.sid
    if sid in lobby_queue:
        return
    lobby_queue.append(sid)

    # Match when 2 players are waiting
    if len(lobby_queue) >= 2:
        p1 = lobby_queue.pop(0)
        p2 = lobby_queue.pop(0)
        room_id = str(uuid.uuid4())

        # Initialize game
        games[room_id] = {
            "board": [""] * 9,
            "turn": "X",
            "players": {"X": p1, "O": p2},
            "winner": None,
            "draw": False
        }
        player_room[p1] = room_id
        player_room[p2] = room_id

        join_room(room_id, sid=p1)
        join_room(room_id, sid=p2)

        # Notify both players of roles and start
        socketio.emit("game_start", {
            "room": room_id,
            "role": "X",
            "opponent": p2,
            "board": games[room_id]["board"],
            "turn": "X"
        }, to=p1)

        socketio.emit("game_start", {
            "room": room_id,
            "role": "O",
            "opponent": p1,
            "board": games[room_id]["board"],
            "turn": "X"
        }, to=p2)

@socketio.on("make_move")
def make_move(data):
    # data: {room, index, role}
    sid = request.sid
    room = data.get("room")
    idx = data.get("index")
    role = data.get("role")

    if room not in games:
        emit("error_msg", {"msg": "Invalid room"})
        return

    game = games[room]

    # Validate player and turn
    if game["players"].get(role) != sid:
        emit("error_msg", {"msg": "Not your role"})
        return
    if game["winner"] or game["draw"]:
        return
    if role != game["turn"]:
        emit("error_msg", {"msg": "Not your turn"})
        return
    if not (0 <= idx <= 8) or game["board"][idx] != "":
        emit("error_msg", {"msg": "Invalid move"})
        return

    # Make move
    game["board"][idx] = role

    # Check winner/draw
    winner = check_winner(game["board"])
    if winner:
        game["winner"] = winner
    elif all(cell != "" for cell in game["board"]):
        game["draw"] = True
    else:
        game["turn"] = "O" if game["turn"] == "X" else "X"

    # Broadcast update
    socketio.emit("board_update", {
        "board": game["board"],
        "turn": game["turn"],
        "winner": game["winner"],
        "draw": game["draw"]
    }, room=room)

@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    # Remove from lobby
    if sid in lobby_queue:
        lobby_queue.remove(sid)
    # Handle in-game disconnect
    room = player_room.pop(sid, None)
    if room and room in games:
        game = games[room]
        # Award win to remaining player if any
        remaining = None
        for role, psid in game["players"].items():
            if psid != sid:
                remaining = psid
        game["winner"] = "X" if game["players"].get("X") == remaining else "O"
        socketio.emit("board_update", {
            "board": game["board"],
            "turn": game["turn"],
            "winner": game["winner"],
            "draw": False
        }, room=room)
        # Clean up room membership (optional)
        # leave_room(room, sid=sid)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
