import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

######################### offline algorithm to produce offline OPT distance #################################
def offline_k_center(points, k):
    # Initialize the first center randomly
    centers = [points[np.random.randint(len(points))]]
    
    while len(centers) < k:
        # Find the point that is the farthest from any center
        next_center = max(points, key=lambda point: min(euclidean_distance(point, center) for center in centers))
        centers.append(next_center)
    
    return centers

def max_distance_to_centers(points, centers):
    max_dist = 0
    for point in points:
        min_dist_to_center = min(euclidean_distance(point, center) for center in centers)
        max_dist = max(max_dist, min_dist_to_center)
    return max_dist

def plot_points_and_centers(points, centers):
    plt.figure(figsize=(8, 6))
    x, y = zip(*points)
    cx, cy = zip(*centers)
    
    plt.scatter(x, y, color='blue', label='Points')
    plt.scatter(cx, cy, color='red', s=100, label='Centers', edgecolors='black')
    plt.title("Offline k-Center Clustering")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate random points
np.random.seed(42)
data_points = np.random.rand(100, 2) * 100  # 100 points in a 100x100 grid

# Number of centers
k = 5

# Solve the offline k-center problem
centers = offline_k_center(data_points, k)

# Calculate the maximum distance to the nearest center
# This value used as input parameter in the online problem
max_dist = max_distance_to_centers(data_points, centers)

# Plot the points and the selected centers
#plot_points_and_centers(data_points, centers)

print("Selected Centers:", centers)
print("Maximum distance to nearest center:", max_dist)


####################################################################################################################################
#
#
################ online positive-body chasing for k-center #########################################################################


# define the radius for the ball to outline the covering/packing constraints
beta = 1.5
r = beta * max_dist

epsilon = 0.25

####################################### functions needed for main method ############################################################

# initialization for online_k_center
def initialization(points):
    x = np.zeros(len(points))

    # start with a set of candidate points of randomly selected 10 points from input
    indices = np.random.choice(points.shape[0], size = 10, replace=False)
    candidates = points[indices, :]
    
    print("\n")
    print("candidate indices:", indices)
    print("center candidates:", candidates)

    # initialize constraint matrix
    constraint_matrix = []

    return x, indices, candidates, constraint_matrix

# update the covering constraints to accommodate insertion and deletion of clients
def update_constraints(C,s, n):
    # update the constraint matrix that for each client, Cx >= 1
    # where row t in C for client t has 0s at entries in s
    new_row = np.zeros(n)
    new_row[s] = 1
    C.append(new_row)
    
    return C

# find all points within the radius of the current client
# this step can be thought of as updating the covering constraint
def find_candidates(candidate_indices, points, client, radius):
    s = []
    for index in candidate_indices:
        if euclidean_distance(points[index], client) <= radius:
            s.append(index)
    
    print("center candidates within radius:", s)
    return s

# check if new client's constraint is satisfied
def check_covering_feasibility(s, x):
    # check if the covering constraint is satisfied
    covering_sum = np.sum(x[s])
    print("total weight in radius:", covering_sum)
    if covering_sum >= 1:
        return True
    else:
        return False


# covering objective function for solving x's
def covering_objective(x, x_old, s, epsilon):

    # we only deal with the x_i's whose coefficient c_i is nonzero
    # print("candidate set passed to objective:", s)
    prev_x_hat = np.zeros(len(s))

    for i in range(len(prev_x_hat)):
        prev_x_hat[i] = x_old[s[i]] + epsilon/(4*len(s))

    
    #print("x_hat values in candidate set:", prev_x_hat)
    
    #prev_x = np.zeros(len(s))
    return np.sum((x + epsilon/(4*len(s))) * np.log((x + epsilon/(4*len(s)))/prev_x_hat) - (x + epsilon/(4*len(s))))


# packing objective function for solving x's
def packing_objective(x, x_old):
    return np.sum(x * np.log(x/x_old) - x)


# covering constraint for solving x's
def covering_constraint(x):
    return np.sum(x) - 1


# packing constraint for solving x's
def packing_constraint(x, epsilon, k):
    return (1 + epsilon) * k - np.sum(x) 


# optimize and update the values of x's when a covering constraint not satisfied
def update_covering_variables(x, s, epsilon):
    # start with x's at t-1
    x0 = np.zeros(len(s))
    for i in range(len(s)):
        x0[i] = x[s[i]]

    # set up constraint for scipy solver for x's
    cons = {'type': 'ineq', 'fun':covering_constraint}

    result = minimize(covering_objective, x0, args=(x, s, epsilon), constraints=cons, method='SLSQP')

    print("updated fractional solutions:", result.x)
    
    # finally, update the values in the original x vector
    x_new = x
    for i in range(len(s)):
        x_new[s[i]] = result.x[i]

    return x_new


# optimize and update the values of x's when a packing constraint not satisfied
def update_packing_variables(x, epsilon, k):
    
    cons = {'type': 'ineq', 'fun':packing_constraint, 'args':(epsilon, k)}

    x0 = x

    result = minimize(packing_objective, x0, args=(x), constraints=cons, method='SLSQP')

    return result.x


########################## rouding algorithm for each fractional solution at time t ###############################


    


############################ main method for online positive-body chasing for k-center ############################

def online_k_center(points, k, r):

    recourse = 0

    # initialize the vector x with all 0s of dimension len(points)
    x, candidate_index, candidates, constraint_mat = initialization(points)

    #index = np.zeros(len(points))
    #x_hat = np.zeros(len(points))

    for t in range(10):
        # when a client arrives, check if there is any violating constraint
        candidate_index = np.append(candidate_index, t)
        candidates = np.append(candidates, points[t])

        print("current client:", points[t])

        s = find_candidates(candidate_index, points, points[t], r)
        # update the covering constraint matrix
        #constraint_mat = update_constraints(constraint_mat, s, len(points))

        # check if covering constraint is violated
        # if so, update the values of x's
        print("covering feasibility?", check_covering_feasibility(s, x))
        if check_covering_feasibility(s, x) == False:
            # covering constraint not satisfied, need to update values of x's in s
            x_new = update_covering_variables(x, s, epsilon)

        # check if packing constraint is violated
        # if so, update the values of x's
        if (np.sum(x) > (1+epsilon) * k):
            print("packing constraint violated!")
            x_new = update_packing_variables(x, epsilon, k)


    
    # update total recourse in l1-norm
    recourse += np.sum(np.abs(x_new - x))
    
    return
    

online_k_center(data_points, k, r)


