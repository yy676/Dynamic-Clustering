import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import copy
import random
import pulp


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
beta = 1.25

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
# (potentially needed)
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
    
    #print("candidates within radius:", s)
    return s

# check if new client's constraint is satisfied
def check_covering_feasibility(s, x):
    # check if the covering constraint is satisfied
    covering_sum = np.sum(x[s])
    #print("total weight in radius:", covering_sum)
    if covering_sum >= 1:
        return True
    else:
        return False


# covering objective function for solving x's
def covering_objective(x, x_old, s, epsilon):

   # we only deal with the x_i's whose coefficient c_i is nonzero
    prev_x_hat = np.zeros(len(s))

    for i in range(len(prev_x_hat)):
        prev_x_hat[i] = x_old[s[i]] + epsilon/(4 * len(s))
    
    x_adj = x + epsilon / (4 * len(s)) 
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = x_adj / prev_x_hat
        where_valid = prev_x_hat > 0
        log_term = np.where(where_valid, np.log(np.maximum(ratio, 1e-10)), 0)

    return np.sum(x_adj * log_term - x_adj)


# packing objective function for solving x's
def packing_objective(x, x_old):

    objective = 0
    for i in range(len(x)):
        # additional checking to avoid invalid values
        if x_old[i] != 0 and x[i] > 0:
            ratio = x[i] / x_old[i]
            if ratio > 0:
                objective += x[i] * np.log(ratio) - x[i]
    
    return objective

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
    
    # finally, update the values in the original x vector
    x_new = x
    for i in range(len(s)):
        x_new[s[i]] = result.x[i]

    #print("updated fractional solutions:", x_new)

    return x_new


# optimize and update the values of x's when a packing constraint not satisfied
def update_packing_variables(x, epsilon, k):
    
    cons = [{'type': 'ineq', 'fun':packing_constraint, 'args':(epsilon, k)},
            {'type': 'ineq', 'fun':positive_constraint}]

    x0 = x

    bounds = Bounds(0, np.inf)

    result = minimize(packing_objective, x0, args=(x), constraints=cons, method='SLSQP', bounds=bounds)

    #print("updated fraction solutions after packing violation:", result.x)

    return result.x


# alternative method to update x's when a covering violation occurs
# we use the closed form with the lagrange multiplier here
def update_covering(x, s, epsilon):
    d = len(s)
    log_term = (epsilon/4 + 1) / (np.sum(x[s] + epsilon / 4))

    if log_term > 0:
        y = np.log(log_term)
        x[s] = (x[s] + epsilon / (4 * d)) * np.exp(y) 
    
    return x


# alternative method to update x's when a packing violation occurs
# similarly, we also use the closed form here
def update_packing(x, epsilon, k):
    log_term_bottom = 0
    for i in range(len(x)):
        if x[i] > 0:
            log_term_bottom += x[i]
    
    if log_term_bottom > 0:
        neg_z = np.log(((1 + epsilon) * k) / log_term_bottom)
        x = x * np.exp(neg_z)
    
    return x


############################ method used to compute OPT_rec at time t #################################

# arguments: 
#   C: a list of length T that consists of violated covering constraints at time t; 
#      C(t) is the indices of non-zero c_i's that appear in a violated covering constraint at time t: 
#      i.e., C(t) * x < 1; 
#      C(t) is an empty set if no violation at t
#   P: row t is all 1 vector if packing constraint is violated at time t; empty set otherwise
#   covering_t, packing_t: sets that store the t's when violations happen
#   T: the number of steps so far

def compute_OPT_rec(C, P, covering_t, packing_t, t, k, epsilon):

    # Problem data and parameters
    T = t + 1  # Number of time periods
    n = t + 1  # Number of variables per period
    #weights = {t: [1]*n for t in range(0, T)}  # Example weights

    # Create the LP problem object
    lp_prob = pulp.LpProblem("OPT_recourse", pulp.LpMinimize)

    # Decision variables x_i^t and l_i^t
    x = pulp.LpVariable.dicts("x", (range(n), range(T)), lowBound=0)
    l = pulp.LpVariable.dicts("l", (range(n), range(T)), lowBound=0)

    # Objective function
    lp_prob += pulp.lpSum(l[i][t] for t in range(T) for i in range(n))

    # Constraints
    for t in covering_t:
        subset_indices = C[t]
        #print("violating covering constraint(subset of indices):", subset_indices)
        lp_prob += pulp.lpSum(x[i][t] for i in subset_indices) >= 1, f"CoverageConstraint_{t}"
    for t in packing_t:
        lp_prob += pulp.lpSum(x[i][t] for i in range(n)) <= (1 + epsilon) * k, f"PerformanceConstraint_{t}"
    for i in range(n):
        lp_prob += x[i][0] <= l[i][0]
    for t in range(1, T):
        for i in range(n):
            lp_prob += (x[i][t] - x[i][t-1]) <= l[i][t], f"ChangeConstraint_Pos_{t}_{i}"
            lp_prob += (x[i][t-1] - x[i][t]) <= l[i][t], f"ChangeConstraint_Neg_{t}_{i}"

    # Solve the problem
    lp_prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Output results
    #print("Status:", pulp.LpStatus[lp_prob.status])
    #for var in lp_prob.variables():
        #print(f"{var.name} = {var.varValue}")
    #print("Total OPT_recourse = ", pulp.value(lp_prob.objective))
    return pulp.value(lp_prob.objective)


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

    # stores the number of total requests
    # will be modified when removal requests are added
    num_requests = len(points)

    # initialize variables needed for computing OPT_rec:
    # a t-by-n list of lists C, whose number of rows grows as t increases,
    # at each time t, if a covering constraint is violated,
    # append to C the vector c(t) such that the constraint c(t) * x(t) < 1,
    # otherwise, append an empty list
    # A matrix P for packing constraints is defined similarly
    # auxiliary sets violated_covering_t, violated_packing_t that store the values of t when a 
    # covering/packing constraint is violated at t
    C_list = []
    P_list = []
    violated_covering_t = []
    violated_packing_t = []
         
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

        #print("\n")
        #print("time t:", t)
        #print("current client:", points[t])

        # search for points within the radius of the current client
        # the radius at each t is defined as: min(diam(t), beta * OPT(t))
        diam = calculate_diameter(client_points)
        centers_offline = offline_k_center(client_points, k)
        current_OPT_dist = max_distance_to_centers(client_points, centers_offline)

        #print("diam(t):", diam)
        #print("curront_OPT_dist:", current_OPT_dist)

        s = find_candidates(candidate_index, points, points[t], min(beta * current_OPT_dist, diam))
        # update the covering constraint matrix
        #constraint_mat = update_constraints(constraint_mat, s, len(points))

        # check if covering constraint is violated
        # if so, update the values of x's
        #print("covering feasibility?", check_covering_feasibility(s, x))
        if check_covering_feasibility(s, x) == False:
            # covering constraint not satisfied, need to update values of x's in s
            # add the set s to C_list for computing OPT_rec
            # record t in violated_covering_t
            #print("Solving covering constraints...")
            C_list.append(s)
            violated_covering_t.append(t)
            x_new = update_covering_variables(x, s, epsilon)
        else:
            # no covering constraints are violated, add empty list at row t 
            C_list.append([])

        # check if packing constraint is violated
        # if so, update the values of x's
        if (np.sum(x) > (1+epsilon) * k):
            #print("Solving packing constraints")
            # p(t) will just be an all 1 vector of length t
            p_vector = np.ones(t)
            P_list.append(p_vector)
            violated_packing_t.append(t)
            x_new = update_packing_variables(x, epsilon, k)
        else:
            P_list.append([])
        
        OPT_recourse = compute_OPT_rec(C_list, P_list, violated_covering_t, violated_packing_t, t, k, epsilon)

        # update total recourse in l1-norm
        #print("recourse in this round:", np.sum(np.abs(x_new - x_old)))

        total_recourse += np.sum(np.abs(x_new - x_old))
        #print("total online recourse so far:", total_recourse)
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

        #print("\n")
        #print("Rounding begins")
        #print("current set of centers:", set_of_centers)

        #list_of_B_i = []
        list_of_B_i_hat = []
        S = copy.deepcopy(set_of_centers)
        for center in S:
            
            #print("\n")
            #print("for center:", center)
            #print("r_i of this center:", radius_of_centers[center])
            #print("r_i_hat of this center:", alpha * min(beta * current_OPT_dist, diam))
            
            B_i, B_i_hat = find_balls(points, clients, center, radius_of_centers[center], alpha * min(beta * current_OPT_dist, diam))
            
            # drop any B_i whose mass is too small
            mass = 0
            for index_of_point in B_i:
                mass += x[index_of_point]
            
            #print("mass for cent", mass)

            if mass < 1 - epsilon:
                set_of_centers.remove(center)
                #print("center dropped from set")
            else:
                list_of_B_i_hat.append(B_i_hat)
        
        covered_points = set(item for sublist in list_of_B_i_hat for item in sublist)
        #print("\n")
        #print("covered points:", covered_points)
        #print("all clients:", client_indices)
        while len(covered_points) < len(client_indices):
        # find the clients that are not covered by the current set of centers
            uncovered = set(client_indices) - covered_points
            #print("uncovered clients:", uncovered)

            j = next(iter(uncovered))
            set_of_centers.append(j)
            # record current_radius
            current_r = min(beta * current_OPT_dist, diam)
            radius_of_centers[j] = current_r

            #print("new center added to set:", j)

            for center_index in set_of_centers:
                
                ball_dist = radius_of_centers[j] + radius_of_centers[center_index] + delta * min(radius_of_centers[j], radius_of_centers[center_index])
                #print("distance between two centers:", euclidean_distance(points[center_index], points[j]))
                #print("ball disj-radius:", ball_dist)
                if center_index != j and euclidean_distance(points[center_index], points[j]) <= ball_dist:
                    set_of_centers.remove(center_index)
                    #print("center {center_index} dropped", center_index)

            # update the the set of B_i_hat
            list_of_B_i_hat = []
            for center in set_of_centers:
                B_i, B_i_hat = find_balls(points, clients, center, radius_of_centers[center], alpha * min(beta * current_OPT_dist, diam))
                list_of_B_i_hat.append(B_i_hat)
            covered_points = set(item for sublist in list_of_B_i_hat for item in sublist)
        
        #print("all clients covered!")
        #print("selected centers for this round:", set_of_centers)

    return x, total_recourse, set_of_centers, OPT_recourse


#####################################################################################################################
################################################### main ############################################################
##################################################################################################################### 

## parse inputs
# Generate random points
np.random.seed(42)
all_points = np.random.rand(100, 2) * 100  # 100 points in a 100x100 grid
data_points = random.sample(list(all_points), 100)
#print("data points:", data_points)

# Number of centers
k = 5

# Solve the offline k-center problem
centers = offline_k_center(data_points, k)

# Calculate the maximum distance to the nearest center
# This value used as input parameter in the online problem
max_dist = max_distance_to_centers(data_points, centers)

#diam_offline = calculate_diameter(data_points)

'''
print("fractional lp solution:")
data_list = np.array([list(p) for p in data_points])
print("data list:", data_list)
offline_k_center_lp(data_list, k)
'''

# Plot the points and the selected centers
#plot_points_and_centers(data_points, centers)

#print("Selected Centers:", centers)
print("\n")
print("Offline maximum distance to nearest center:", max_dist)

# begin online algorithm
fractional_sol, recourse, centers, OPT_rec = online_k_center(data_points, k) 

print("-----------final online results-----------")
print("k = ", k)
print("beta = ", beta)
print("epsilon = ", epsilon)
print("final fractional solution:", fractional_sol)
print("number of centers (sum of fractional x's):", np.sum(fractional_sol))
print("OPT recourse:", OPT_rec)
print("total online recourse:", recourse)
print("final selected centers:", centers)







