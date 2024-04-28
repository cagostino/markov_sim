import numpy as np
import matplotlib.pyplot as plt

# Constants
CONSTANT = 10  # Adjust this to change forgiveness rate
NUM_ROUNDS = 1000  # Total number of rounds in a game
NUM_SIMULATIONS = 1000  # Number of simulations

def forgiveness_strategy(t_since_last_cheating, n_cheating, constant):
    forgive = np.exp(n_cheating*constant/t_since_last_cheating**10) / t_since_last_cheating
    print(forgive)
    return forgive

def simulate_game():
    scores = {"forgiveness": [], "tit_for_tat": []}
    t_since_last_cheating = 1
    n_cheating = 0
    last_move_forgiveness = 1  # Start cooperatively: 1 for cooperate, 0 for defect
    last_move_tft = 1  # Tit-for-Tat also starts cooperatively
    
    for _ in range(NUM_ROUNDS):
        move_forgiveness = 1 if np.random.rand() < forgiveness_strategy(t_since_last_cheating, n_cheating, n_cheating/2) else 0
        move_tft = last_move_forgiveness  # Tit-for-Tat copies last move
        
        # Update scores based on the Prisoner's Dilemma payoff matrix
        if move_forgiveness == move_tft == 1:
            scores["forgiveness"].append(3)  # Reward for mutual cooperation
            scores["tit_for_tat"].append(3)
        elif move_forgiveness == 1 and move_tft == 0:
            scores["forgiveness"].append(0)  # Sucker's payoff
            scores["tit_for_tat"].append(5)  # Temptation payoff
            n_cheating += 1
            t_since_last_cheating = 1
        elif move_forgiveness == 0 and move_tft == 1:
            scores["forgiveness"].append(5)  # Temptation payoff
            scores["tit_for_tat"].append(0)  # Sucker's payoff
        else:
            scores["forgiveness"].append(1)  # Punishment for mutual defection
            scores["tit_for_tat"].append(1)
        
        last_move_forgiveness = move_forgiveness
        t_since_last_cheating += 1
    
    return np.sum(scores["forgiveness"]), np.sum(scores["tit_for_tat"])

# Run simulations
results = {"forgiveness": [], "tit_for_tat": []}
for _ in range(NUM_SIMULATIONS):
    forgiveness_score, tft_score = simulate_game()
    results["forgiveness"].append(forgiveness_score)
    results["tit_for_tat"].append(tft_score)

# Plot results
plt.hist(results["forgiveness"], alpha=0.5, label='Forgiveness Strategy')
plt.hist(results["tit_for_tat"], alpha=0.5, label='Tit-for-Tat')
plt.legend(loc='upper right')
plt.xlabel('Total Score')
plt.ylabel('Frequency')
plt.title('Strategy Performance Comparison')
plt.show()


print(np.mean(results['tit_for_tat']))
print(np.mean(results['forgiveness']))