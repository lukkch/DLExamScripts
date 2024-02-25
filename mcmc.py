import numpy as np

# exam 2023, task 54

def target_dist(i):
    return np.exp(-np.abs(i))

def proposal_dist(i,j):
    return np.exp(-2*(i-j)**2)

start_state = 1.0
end_state_a = 2.0
end_state_b = 0.0

def prob_choosing(old_state, new_state):
    return proposal_dist(new_state, old_state)
    
def prob_accepting(old_state, new_state):
    return min(1.0, (target_dist(new_state)/ target_dist(old_state))
                  * (proposal_dist(old_state, new_state) / proposal_dist(new_state, old_state)))

print(f"P(choosing 2) = {prob_choosing(start_state, end_state_a)}")
print(f"P(choosing 0) = {prob_choosing(start_state, end_state_b)}")
print(f"P(accepting 2) = {prob_accepting(start_state, end_state_a)}")
print(f"P(accepting 0) = {prob_accepting(start_state, end_state_b)}")

prob_end_state_a = prob_choosing(start_state, end_state_a) * prob_accepting(start_state, end_state_a)
prob_end_state_b = prob_choosing(start_state, end_state_b) * prob_accepting(start_state, end_state_b)
print(f"P(2|1) = {prob_end_state_a}")
print(f"P(0|1) = {prob_end_state_b}")
print(f"P(2|1)/P(0|1) = {prob_end_state_a / prob_end_state_b}")

