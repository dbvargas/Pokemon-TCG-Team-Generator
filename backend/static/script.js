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
    
    // Clear existing cards
    deckGrid.innerHTML = '';
    
    // Get current deck from localStorage
    const currentDeck = JSON.parse(localStorage.getItem('currentDeck') || '[]');
    
    // Update deck count
    document.getElementById('deck-count').textContent = currentDeck.length;
    
    // Group cards by evolution chains or related mechanics
    const cardGroups = groupRelatedCards(currentDeck);
    
    // Add card groups to grid
    cardGroups.forEach(group => {
        if (group.length === 1) {
            // Single cards without relations
            const card = group[0];
            const cardElement = document.createElement('div');
            cardElement.className = 'deck-card';
            cardElement.id = `deck-card-${card.id}`;
            cardElement.innerHTML = `
                <img src="${card.imageUrl}" alt="${card.title}" 
                     onclick="openCardModal('${card.id}', '${card.title}')">
                <div class="remove-card" onclick="removeCardFromDeck('${card.id}')">×</div>
            `;
            deckGrid.appendChild(cardElement);
        } else {
            // Create stacked cards for evolutions/related cards
            const stackElement = document.createElement('div');
            stackElement.className = 'evolution-stack';
            
            // Add 'large' class for stacks with more than 2 cards
            if (group.length > 2) {
                stackElement.classList.add('large');
            }
            
            // Sort cards by their relationship strength
            group.sort((a, b) => (b.score || 0) - (a.score || 0));
            
            // Add each card to the stack with appropriate styling
            group.forEach((card, index) => {
                const cardElement = document.createElement('div');
                cardElement.className = 'deck-card';
                cardElement.id = `deck-card-${card.id}`;
                
                let relationText = '';
                let scoreText = '';
                
                if (index > 0) {
                    relationText = `<div class="card-relation">${card.relation || 'Related'}</div>`;
                    if (card.score) {
                        scoreText = `<div class="similarity-score">${Math.round(card.score * 100)}%</div>`;
                    }
                }
                
                cardElement.innerHTML = `
                    ${relationText}
                    ${scoreText}
                    <img src="${card.imageUrl}" alt="${card.title}" 
                         onclick="openCardModal('${card.id}', '${card.title}')">
                    <div class="remove-card" onclick="removeCardFromDeck('${card.id}')">×</div>
                `;
                
                stackElement.appendChild(cardElement);
            });
            
            deckGrid.appendChild(stackElement);
        }
    });
    
    // Save updated deck to localStorage
    localStorage.setItem('currentDeck', JSON.stringify(currentDeck));
}

// Group cards based on name similarity, evolutions, or related mechanics
function groupRelatedCards(cards) {
    const groups = [];
    const processedCards = new Set();
    
    // Process each card
    cards.forEach(card => {
        // Skip if already processed
        if (processedCards.has(card.id)) return;
        
        // Mark as processed
        processedCards.add(card.id);
        
        // Start a new group with this card
        const group = [card];
        
        // Look for related cards
        cards.forEach(potentialRelated => {
            if (card.id === potentialRelated.id || processedCards.has(potentialRelated.id)) return;
            
            const similarity = calculateCardSimilarity(card, potentialRelated);
            if (similarity > 0.4) { // Threshold for relation
                potentialRelated.score = similarity;
                potentialRelated.relation = getRelationshipType(card, potentialRelated);
                
                group.push(potentialRelated);
                processedCards.add(potentialRelated.id);
            }
        });
        
        groups.push(group);
    });
    
    return groups;
}

// Calculate similarity between cards (0-1)
function calculateCardSimilarity(card1, card2) {
    // Simple name-based similarity for now
    // Extract base name without numbers/special chars
    const baseName1 = card1.title.replace(/\d+$|\sV$|\sEX$|\sGX$|\sVMAX$|\sV-UNION$/i, '').trim();
    const baseName2 = card2.title.replace(/\d+$|\sV$|\sEX$|\sGX$|\sVMAX$|\sV-UNION$/i, '').trim();
    
    // Check if base names match
    if (baseName1 === baseName2) {
        return 0.9; // High similarity for same base name
    }
    
    // Look for partial name matches
    if (baseName1.includes(baseName2) || baseName2.includes(baseName1)) {
        return 0.7; // Medium-high for partial matches
    }
    
    // In a real app, you'd use more advanced similarity algorithms
    return 0;
}

// Determine relationship type based on card properties
function getRelationshipType(card1, card2) {
    // Extract card names and properties
    const baseName1 = card1.title.replace(/\d+$|\sV$|\sEX$|\sGX$|\sVMAX$|\sV-UNION$/i, '').trim();
    const baseName2 = card2.title.replace(/\d+$|\sV$|\sEX$|\sGX$|\sVMAX$|\sV-UNION$/i, '').trim();
    
    // If names match, look for evolution markers
    if (baseName1 === baseName2) {
        if (card2.title.match(/V-UNION|VMAX|VSTAR/i) && !card1.title.match(/V-UNION|VMAX|VSTAR/i)) {
            return 'Evolution';
        }
        if (card2.title.match(/V$/i) && !card1.title.match(/V$/i)) {
            return 'V Form';
        }
        return 'Variant';
    }
    
    return 'Related';
}

document.addEventListener('DOMContentLoaded', () => {
    syncDeckWithLocalStorage();
    updateDeckDisplay();
}); 