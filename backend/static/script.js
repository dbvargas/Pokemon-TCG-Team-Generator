let isDeckMinimized = false;

function syncDeckWithLocalStorage() {
    const storedDeck = localStorage.getItem('currentDeck');
    if (storedDeck) {
        window.currentDeck = JSON.parse(storedDeck);
        updateDeckDisplay();
    } else {
        localStorage.setItem('currentDeck', JSON.stringify(window.currentDeck || []));
    }
}

function toggleDeckPopup() {
    const popup = document.getElementById('deckPopup');
    const minimizeButton = document.querySelector('.minimize-button');
    
    isDeckMinimized = !isDeckMinimized;
    popup.classList.toggle('minimized');
    
    minimizeButton.textContent = isDeckMinimized ? '+' : '−';
}

function updateDeckDisplay() {
    const deckGrid = document.getElementById('deck-grid');
    const deckCount = document.querySelector('.deck-popup-header h3');
    
    deckGrid.innerHTML = '';
    
    const currentDeck = JSON.parse(localStorage.getItem('currentDeck') || '[]');
    
    document.getElementById('deck-count').textContent = currentDeck.length;
    
    currentDeck.forEach(card => {
        const cardElement = document.createElement('div');
        cardElement.className = 'deck-card';
        cardElement.innerHTML = `
            <img src="${card.imageUrl}" alt="${card.title}" 
                 onclick="openCardModal('${card.id}', '${card.title}')">
            <div class="remove-card" onclick="removeCardFromDeck('${card.id}')">×</div>
        `;
        deckGrid.appendChild(cardElement);
    });
    
    localStorage.setItem('currentDeck', JSON.stringify(currentDeck));
}

document.addEventListener('DOMContentLoaded', () => {
    syncDeckWithLocalStorage();
    updateDeckDisplay();
}); 