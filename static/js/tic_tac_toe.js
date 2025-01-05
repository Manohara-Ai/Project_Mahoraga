let player;
let bot;
let turn;
let state;

async function initializeGame() {
    try {
        const response = await fetch('/tictactoe?format=json');
        const data = await response.json();

        turn = data.turn;
        player = data.player;
        bot = data.bot;

        console.log("Player:", player);
        console.log("Bot:", bot);

        if (turn === "bot") {
            fetchAiMove();
        }

    } catch (error) {
        console.error('Error fetching game constants:', error);
    }
}

function updateGameBoard(state) {
    state.forEach((row, rowIndex) => {
        row.forEach((cell, colIndex) => {
            const cellIndex = rowIndex * 3 + colIndex;
            const cellElement = document.querySelector(`[data-cell='${cellIndex}']`);

            if (cellElement) {
                cellElement.textContent = (cell === 1) ? "X" : (cell === -1) ? "O" : "";
                cellElement.classList.remove('player-x', 'player-o');
            
                if (cell === 1) {
                    cellElement.classList.add('player-x');
                    cellElement.removeEventListener('click', handleCellClick);
                } else if (cell === -1) {
                    cellElement.classList.add('player-o');
                    cellElement.removeEventListener('click', handleCellClick);
                } else {
                    cellElement.textContent = "";
                }
            }
        });
    });
}

function fetchAiMove() {
    fetch('/tictactoe/ai-move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        state = data.state;
        console.log(data.message);
        updateGameBoard(state);
        
        if (data.message.includes("Wins") || data.message.includes("Tie")) {
            renderGameMessage(data.message)
            disableAllListeners();
        }
    })
    .catch(error => {
        console.error('Error fetching AI move:', error);
    });
}

function handleCellClick(event) {
    const cellIndex = event.target.getAttribute('data-cell');

    fetch('/tictactoe/player-move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ cell: cellIndex })
    })
    .then(response => response.json())
    .then(data => {
        state = data.state;
        console.log(data.message);
        updateGameBoard(state);

        if (data.message.includes("Wins") || data.message.includes("Tie")) {
            renderGameMessage(data.message)
            disableAllListeners();
        } else {
            fetchAiMove();
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function initializeCellClickListeners() {
    document.querySelectorAll('.cell').forEach(cell => {
        cell.addEventListener('click', handleCellClick);
    });
}

function disableAllListeners() {
    document.querySelectorAll('.cell').forEach(cell => {
        cell.removeEventListener('click', handleCellClick);
    });
    console.log("Game over! Refresh to continue.");
}

function renderGameMessage(message) {
    const resultElement = document.querySelector('.game-result');
    
    const sanitizedMessage = message.toLowerCase().replace(/\s+/g, '-');  
    
    resultElement.textContent = message;
    resultElement.classList.add(sanitizedMessage); 
}

initializeCellClickListeners();
initializeGame();
