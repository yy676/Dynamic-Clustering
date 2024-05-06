import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import copy
import random
import pulp as pl


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

def calculate_diameter(points):
    diameter = 0.0
    num_points = len(points)
    for i in range(num_points):
        for j in range(i+1, num_points):
            distance = euclidean_distance(points[i], points[j])
            if distance > diameter:
                diameter = distance
    return diameter

############################### offline algorithm to produce offline OPT distance #################################

# fractional lp solver for offline k-center
def offline_k_center_lp(points, num_centers):
    num_points = len(points)
    # Distance matrix
    distances = np.sqrt(((points[:, np.newaxis] - points[np.newaxis, :]) ** 2).sum(axis=2))

    # Number of centers
    k = num_centers

    # Create the LP problem
    problem = pl.LpProblem("k-Median Problem", pl.LpMinimize)

    # Decision variables
    x = pl.LpVariable.dicts("x", (range(num_points), range(num_points)), lowBound=0, upBound=1, cat='Continuous')
    y = pl.LpVariable.dicts("y", range(num_points), lowBound=0, upBound=1, cat='Continuous')

    # Objective function
    problem += pl.lpSum(x[i][j] * distances[i][j] for i in range(num_points) for j in range(num_points))

    # Constraints
    for i in range(num_points):
        problem += pl.lpSum(x[i][j] for j in range(num_points)) == 1

    for i in range(num_points):
        for j in range(num_points):
            problem += x[i][j] <= y[j]

    problem += pl.lpSum(y[j] for j in range(num_points)) <= k

    # Solve the problem
    problem.solve()

    if problem.status == pl.LpStatusOptimal:
        print("Optimal cost:", pl.value(problem.objective))
        for j in range(num_points):
            if y[j].varValue > 0.01:  # Only consider y_j significant if > 0.01
                print(f"Center at point {j} with y_{j} = {y[j].varValue:.2f}")
                for i in range(num_points):
                    if x[i][j].varValue > 0.01:
                        print(f"  Point {i} assigned fractionally {x[i][j].varValue:.2f} to center {j}")


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

####################################### online positive-body chasing for k-center ##################################################

# setting parameters needed for online algorithm
beta = 1.5

epsilon = 0.25

####################################### functions needed for main method ############################################################

# initialization for online_k_center
def initialization(points):
    x = np.zeros(len(points))

    # start with a set of candidate points of randomly selected 10 points from input
    #indices = np.random.choice(points.shape[0], size = 10, replace=False)
    #candidates = points[indices, :]
    
    print("\n")
    #print("candidate indices:", indices)
    #print("center candidates:", candidates)

    indices = []
    candidates = []

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
        #print("index in candidate_indicies:", index)
        if euclidean_distance(points[int(index)], client) <= radius:
            s.append(int(index))
    
    print("candidates within radius:", s)
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
    prev_x_hat = np.zeros(len(s))

    for i in range(len(prev_x_hat)):
        prev_x_hat[i] = x_old[s[i]] + epsilon/(4*len(s))
    
    #prev_x = np.zeros(len(s))
    return np.sum((x + epsilon/(4*len(s))) * np.log((x + epsilon/(4*len(s)))/prev_x_hat) - (x + epsilon/(4*len(s))))


# packing objective function for solving x's
def packing_objective(x, x_old):

    # avoiding divide by zero
    #c = 1e-10
    #x_old = np.maximum(x_old, c)
    
    objective = 0
    for i in range(len(x)):
        if x_old[i] != 0:
            objective += x[i] * np.log(x[i]/x_old[i]) - x[i]
    
    return objective
    #return np.sum(x * np.log(x/x_old) - x)


# covering constraint for solving x's
def covering_constraint(x):
    return np.sum(x) - 1


# packing constraint for solving x's
def packing_constraint(x, epsilon, k):
    return (1 + epsilon) * k - np.sum(x) 

def positive_constraint(x):
    return x 


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
    
    cons = [{'type': 'ineq', 'fun':packing_constraint, 'args':(epsilon, k)},
            {'type': 'ineq', 'fun':positive_constraint}]

    x0 = x

    bounds = Bounds(0, np.inf)

    result = minimize(packing_objective, x0, args=(x), constraints=cons, method='SLSQP', bounds=bounds)

    print("updated fraction solutions after packing violation:", result.x)

    return result.x

######################################### helper for rounding ###########################################

# set the parameters for rounding
alpha = 3 + 2 * np.sqrt(2)
delta = np.sqrt(2)

# subroutine to find the balls B_i and B_hait_i for a given center_index
# this subroutine is called whenever the set S is updated
def find_balls(data_points, clients, center_index, radius, radius_hat):

    #print("inside find_ball function")
    B_i = []
    B_i_hat =[]

    for i in range(len(clients)):
        #print("current client:", j)
        #print("current center:", s[i])

        #print("client coordinate:", data_points[j])
        #print("center coordinate:", data_points[s[i]])
        #print("distance:", euclidean_distance(data_points[j], data_points[s[i]]))
        client_coordinate = clients[i][1]
        client_index = clients[i][0]
        #print("distance between client and center:", euclidean_distance(client_coordinate, data_points[center_index]))
        if euclidean_distance(client_coordinate, data_points[center_index]) <= radius:
            B_i.append(client_index)
            
        if euclidean_distance(client_coordinate, data_points[center_index]) <= radius_hat:
            B_i_hat.append(client_index)

    #print("points in B_i:", B_i)
    #print("points in B_i_hat:", B_i_hat)
    
    return B_i, B_i_hat

###################################### main method for online k-center #######################################

def online_k_center(points, k):

    total_recourse = 0

    # initialize the vector x with all 0s of dimension len(points)
    x, candidate_index, candidates, constraint_mat = initialization(points)

    # store each client as a tuple (index in points, coordinates)
    clients = []
    # the coordinates of active clients
    client_points = []
    client_indices = []

    set_of_centers = []
    radius_of_centers = np.zeros(len(x))

    for t in range(len(points)):

        x_old = copy.deepcopy(x)
        
        # at each iteration if the request is an insertion,
        # add new client to the set of points that are known 
        candidate_index = np.append(candidate_index, int(t))
        candidates = np.append(candidates, points[t])

        # variables for rounding
        current_client = (t, points[t])
        clients.append(current_client)
        client_points.append(points[t])
        client_indices.append(t)

        
        print("\n")
        
        print("time t:", t)

        print("current client:", points[t])

        # search for points within the radius of the current client
        # the radius at each t is defined as: min(diam(t), beta * OPT(t))
        diam = calculate_diameter(client_points)
        centers_offline = offline_k_center(client_points, k)
        current_OPT_dist = max_distance_to_centers(client_points, centers_offline)

        print("diam(t):", diam)
        print("curront_OPT(t):", current_OPT_dist)

        s = find_candidates(candidate_index, points, points[t], min(beta * current_OPT_dist, diam))
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
        print("recourse in this round:", np.sum(np.abs(x_new - x_old)))

        total_recourse += np.sum(np.abs(x_new - x_old))
        x = x_new
        
        #################################### rounding procedure begins from here ####################################
        # integrate rounding at each round
        # maintain a set S of open centers at each t
        # calculate for each i in S it's radius = min(beta * OPT(t_i), diag)
        # OPT(t_i) is calculated using the offline algorithm

        # calculate OPT_offline for current active clients
        #centers_offline = offline_k_center(client_points, k)
        #current_OPT_dist = max_distance_to_centers(client_points, centers_offline)
        # diam = calculate_diameter(client_points)
        
        # identify the balls/set of points that are B_i and B_i_hat for each i in set_of_centers
        # while building the balls B_i, drop any i from set_of_centers if it has mass less than 1-epsilon
        #list_of_B_i = []

        print("\n")
        print("Rounding begins")
        print("current set of centers:", set_of_centers)

        #list_of_B_i = []
        list_of_B_i_hat = []
        S = copy.deepcopy(set_of_centers)
        for center in S:
            
            print("\n")
            print("for center:", center)
            print("r_i of this center:", radius_of_centers[center])
            print("r_i_hat of this center:", alpha * min(beta * current_OPT_dist, diam))
            
            B_i, B_i_hat = find_balls(points, clients, center, radius_of_centers[center], alpha * min(beta * current_OPT_dist, diam))
            
            # drop any B_i whose mass is too small
            mass = 0
            for index_of_point in B_i:
                mass += x[index_of_point]
            
            print("mass for cent", mass)

            if mass < 1 - epsilon:
                set_of_centers.remove(center)
                print("center dropped from set")
            else:
                list_of_B_i_hat.append(B_i_hat)
        
        covered_points = set(item for sublist in list_of_B_i_hat for item in sublist)
        print("\n")
        print("covered points:", covered_points)
        print("all clients:", client_indices)
        while len(covered_points) < len(client_indices):
        # find the clients that are not covered by the current set of centers
            uncovered = set(client_indices) - covered_points
            print("uncovered clients:", uncovered)

            j = next(iter(uncovered))
            set_of_centers.append(j)
            # record current_radius
            current_r = min(beta * current_OPT_dist, diam)
            radius_of_centers[j] = current_r

            print("new center {j} added to set", j)

            for center_index in set_of_centers:
                
                ball_dist = radius_of_centers[j] + radius_of_centers[center_index] + delta * min(radius_of_centers[j], radius_of_centers[center_index])
                #print("distance between two centers:", euclidean_distance(points[center_index], points[j]))
                #print("ball disj-radius:", ball_dist)
                if center_index != j and euclidean_distance(points[center_index], points[j]) <= ball_dist:
                    set_of_centers.remove(center_index)
                    print("center {center_index} dropped", center_index)

            # update the the set of B_i_hat
            list_of_B_i_hat = []
            for center in set_of_centers:
                B_i, B_i_hat = find_balls(points, clients, center, radius_of_centers[center], alpha * min(beta * current_OPT_dist, diam))
                list_of_B_i_hat.append(B_i_hat)
            covered_points = set(item for sublist in list_of_B_i_hat for item in sublist)
        
        print("all clients covered!")
        print("selected centers for this round:", set_of_centers)

    return x, total_recourse, set_of_centers



#####################################################################################################################
################################################### main ############################################################
##################################################################################################################### 

## parse inputs
# Generate random points
np.random.seed(42)
all_points = np.random.rand(100, 2) * 100  # 100 points in a 100x100 grid
data_points = random.sample(list(all_points), 50)
#print("data points:", data_points)

# Number of centers
k = 5

# Solve the offline k-center problem
centers = offline_k_center(data_points, k)

# Calculate the maximum distance to the nearest center
# This value used as input parameter in the online problem
max_dist = max_distance_to_centers(data_points, centers)

diam_offline = calculate_diameter(data_points)

'''
print("fractional lp solution:")
data_list = np.array([list(p) for p in data_points])
print("data list:", data_list)
offline_k_center_lp(data_list, k)
'''

# Plot the points and the selected centers
#plot_points_and_centers(data_points, centers)


print("Selected Centers:", centers)
print("Maximum distance to nearest center:", max_dist)

# begin online algorithm
fractional_sol, recourse, centers = online_k_center(data_points, k) 

print("final fractional solution:", fractional_sol)
print("number of centers:", np.sum(fractional_sol))
print("total recourse:", recourse)
print("final selected centers:", centers)







