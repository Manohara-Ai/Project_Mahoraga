let player;
let bot;
let turn;
let state;

async function initializeGame() {
    try {
        const response = await fetch('/connect4?format=json');
        const data = await response.json();

        turn = data.turn;
        player = data.player;
        bot = data.bot;
        state = data.state || [];

        console.log("Player:", player);
        console.log("Bot:", bot);

        setupTopRowListeners();

        if (turn === "bot") {
            fetchAiMove();
        }

    } catch (error) {
        console.error('Error fetching game constants:', error);
    }
}

function setupTopRowListeners() {
    document.querySelectorAll('.cell').forEach(cell => {
        const cellIndex = parseInt(cell.getAttribute('data-cell'));
        const colIndex = cellIndex % 7; 

        if (cellIndex >= 7) {
            cell.classList.add('disabled-cell');
        } else {
            cell.addEventListener('click', () => handlePlayerMove(colIndex));
        }
    });
}

function handlePlayerMove(colIndex) {
    fetch('/connect4/player-move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ cell: colIndex })
    })
    .then(response => response.json())
    .then(data => {
        state = data.state;
        console.log(data.message);
        updateGameBoard(state);
        
        if (isColumnFull(colIndex)) {
            disableColumn(colIndex);
        }

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

function updateGameBoard(state) {
    state.forEach((row, rowIndex) => {
        row.forEach((cell, colIndex) => {
            const cellIndex = rowIndex * 7 + colIndex;
            const cellElement = document.querySelector(`[data-cell='${cellIndex}']`);

            if (cellElement) {
                cellElement.textContent = (cell === 1) ? "X" : (cell === -1) ? "O" : "";
                cellElement.classList.remove('player-x', 'player-o');
            
                if (cell === 1) {
                    cellElement.classList.add('player-x');
                } else if (cell === -1) {
                    cellElement.classList.add('player-o');
                }
            }
        });
    });
}

function fetchAiMove() {
    fetch('/connect4/ai-move', {
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
        renderGameMessage(data.message)
        console.error('Error fetching AI move:', error);
    });
}

function isColumnFull(colIndex) {
    return state[0][colIndex] !== 0;
}

function disableColumn(colIndex) {
    document.querySelector(`[data-cell='${colIndex}']`).removeEventListener('click', () => handlePlayerMove(colIndex));
}

function disableAllListeners() {
    document.querySelectorAll('.cell').forEach(cell => {
        cell.replaceWith(cell.cloneNode(true));
    });
}

function renderGameMessage(message) {
    const resultElement = document.querySelector('.game-result');
    
    const sanitizedMessage = message.toLowerCase().replace(/\s+/g, '-');  
    
    resultElement.textContent = message;
    resultElement.classList.add(sanitizedMessage); 
}

initializeGame();
