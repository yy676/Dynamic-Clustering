import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import copy
import random
import pulp

############################################# set up parameters ###################################################
beta = 1.5

epsilon = 0.25

alpha = 11
delta = 2
gamma = float(10 / 9)
eta = 1 / np.log((alpha - gamma * (1 + delta))/ (2 * gamma))

cost = 10

################################################ useful methods ###################################################
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

def approx_facility_location(points, cost):
    return

# LP relaxation for facility location
def lp_relaxation_facility_location(points, candidate_locations, cost):

    num_clients = len(points)
    num_facilities = len(candidate_locations)
    
    # calculate distance matrix
    dist_mat = np.zeros((num_clients, num_facilities))
    for i in range(num_facilities):
        for j in range(num_clients):
            dist_mat[i][j] = euclidean_distance(candidate_locations[i], points[j])
    
    problem = pulp.LpProblem("Facility Location", pulp.LpMinimize)

    # Decision variables
    x = [pulp.LpVariable(f"x_{i}", 0, 1) for i in range(num_facilities)]
    y = [[pulp.LpVariable(f"y_{i}_{j}", 0, 1) for j in range(num_clients)] for i in range(num_facilities)]

    # Objective function
    problem += pulp.lpSum([cost * x[i] for i in range(num_facilities)]) + \
               pulp.lpSum([dist_mat[i][j] * y[i][j] for i in range(num_facilities) for j in range(num_clients)])

    # Constraints
    for j in range(num_clients):
        problem += pulp.lpSum([y[i][j] for i in range(num_facilities)]) >= 1, f"ServiceConstraint_{j}"

    for i in range(num_facilities):
        for j in range(num_clients):
            problem += y[i][j] <= x[i], f"OpenConstraint_{i}_{j}"

    # Solve the problem
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    # Print results
    #for v in problem.variables():
        #print(f"{v.name} = {v.varValue}")

    print(f"Total cost = {pulp.value(problem.objective)}")

    return pulp.value(problem.objective)

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


################################# functions needed for main method ####################################

############################ method used to compute OPT_rec at time t #################################

# arguments: 
# Compute the OPT_rec LP and use this solution for rounding 
# T: the number of steps so far

def compute_OPT_rec(t, candidate_locations, distances, cost, OPT, beta, epsilon):

    # Problem data and parameters
    T = t + 1  # Number of time periods
    num_facilities = len(candidate_locations)
    num_clients = len(distances[0])
    #weights = {t: [1]*n for t in range(0, T)}  # Example weights

    # Create the LP problem object
    lp_prob = pulp.LpProblem("OPT_recourse", pulp.LpMinimize)

    # Decision variables x_i^t and l_i^t
    x = pulp.LpVariable.dicts("x", (range(T), range(num_facilities)), lowBound=0)
    y = [[[pulp.LpVariable(f"y_time_{t}_facility_{i}_client_{j}", lowBound=0)
       for t in range(T)]
       for i in range(num_facilities)]
       for j in range(num_clients)]
    l = pulp.LpVariable.dicts("l", (range(T), range(num_facilities)), lowBound=0)

    # Objective function
    lp_prob += pulp.lpSum(l[t][i] for i in range(num_facilities) for t in range(T)) 

    # Constraints
    for t in range(T):
        lp_prob += pulp.lpSum(y[t][i][j] for i in range(num_facilities) for j in range(num_clients)) >= 1, f"CoverageConstraint_{t}"
        lp_prob += pulp.lpSum(cost * x[t][i] + distances[i][j] * y[t][i][j] for i in range(num_facilities) for j in range(num_clients)) <= beta * OPT * (1 + epsilon)
        for i in range(num_facilities):
        # include the recourse at t = 0
            for j in range(num_clients):
                lp_prob += x[t][i] >= y[t][i][j]
    
    for t in range(1, T):
        for i in range(num_facilities):
            lp_prob += x[t][i] - x[t-1][i] <= l[t][i]
            lp_prob += x[t-1][i] - x[t][i] <= l[t][i]
    
    # Solve the problem
    lp_prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Output results
    
    # printing x and l matrices for debugging
    x_mat = np.zeros((T, num_facilities))
    l_mat = np.zeros((T, num_facilities))

    for i in range(T):
        for j in range(num_facilities):
            x_mat [i, j] = x[i][j].varValue
            l_mat[i, j] = l[i][j].varValue
    
    print("X_OPT:\n" , x_mat)
    #print("l: \n", l_mat)

    #print("Total OPT_recourse = ", pulp.value(lp_prob.objective))
    

    return pulp.value(lp_prob.objective), x_mat[-1], y[-1]


######################################### helper for rounding ###########################################

# subroutine to find the balls B_i and B_hait_i for a given center_index
# this subroutine is called whenever the set S is updated
def find_ball(client, dist_mat, facility_indices, radius):

    B_j = []
    for facility in facility_indices:
        if dist_mat[facility][client] <= radius[client]:
            B_j.append(facility)

    return B_j

###################################### main method for online k-center ####################################

def online_facility_location(requests, points, cost):

    total_rounding_recourse = 0

    num_clients = len(points)
    num_facilities = len(points)

    # initialize the vector x with all 0s of dimension len(points)
    x = np.zeros(num_facilities)
    y = np.zeros((num_facilities, num_clients))

    client_indices = []
    client_coordinates = []
    open_facilities = []
    active_clients = []

    # initialize variables needed for computing OPT_rec:
    # a t-by-n list of lists C, whose number of rows grows as t increases,
    # at each time t, if a covering constraint is violated,
    # append to C the vector c(t) such that the constraint c(t) * x(t) < 1,
    # otherwise, append an empty list
    # A matrix P for packing constraints is defined similarly
    # auxiliary sets violated_covering_t, violated_packing_t that store the values of t when a 
    # covering/packing constraint is violated at t

    B = [[] for _ in range(len(active_clients))]
    radius = np.zeros(num_clients)
    facility_of_client = np.zeros(num_clients)

    t = 0  # for indexing the data points     
    for r in range(len(requests)):

        x_old = copy.deepcopy(x)

        #print("\n")
        #print("---------request:", r)
        
        # at each iteration if the request is an insertion,
        # add new client to the set of points that are known 
        #candidate_index = np.append(candidate_index, int(t))
        #candidates = np.append(candidates, points[t])
        if requests[r] == -1:
            
            # random sample an active client to 
            if len(client_indices) > 0:
                client_to_remove = random.randint(0, t)
                while client_to_remove in open_facilities or client_to_remove not in client_indices:
                    client_to_remove = random.randint(0, t)
                
                x[client_to_remove] = 0
                
                if client_to_remove in client_indices:
                    client_indices.remove(client_to_remove)
            continue
            
        # the request is an insertion, add the new client and deal with any constraint violations
        # set of active clients
        client_indices.append(t)
        client_coordinates.append(points[t])

        facility_locations = client_coordinates
        client_locations = client_coordinates

        num_clients = len(client_locations)
        num_facilities = len(facility_locations)

        facility_indices = client_indices
        
        #print("\n")
        #print("t= ", t)
        #print("current client:", client_coordinates[t])

        # search for points within the radius of the current client
        # the radius at each t is defined as: min(diam(t), beta * OPT(t))
        #diam = calculate_diameter(client_points)
        #centers_offline = offline_k_center(client_points, k)

        # populate distance matrix at t:
        # calculate distance matrix
        dist_mat = np.zeros((num_facilities, num_clients))
        for i in range(num_facilities):
            for j in range(num_clients):
                dist_mat[i][j] = euclidean_distance(facility_locations[i], client_locations[j])
        
        current_OPT_cost = lp_relaxation_facility_location(client_locations, facility_locations, cost)

        #print("diam(t):", diam)
        print("curront_OPT_cost:", current_OPT_cost)
        
        OPT_recourse, x_OPT, y_OPT = compute_OPT_rec(t, facility_locations, dist_mat, cost, current_OPT_cost, beta, epsilon)

        print("fractional solution this round:", x_OPT)
        
        #################################### rounding procedure begins from here ####################################

        R_j = np.zeros(num_clients)
        for client in client_indices:
            for i in range(len(num_facilities)):
                R_j[client] += y_OPT[i][client] * dist_mat[i][client]
                B[client] = find_ball(client, dist_mat, facility_indices, radius) 

            if np.sum(x_OPT[B[client]]) < 1 / alpha:
                active_clients.remove(client)
                open_facilities.remove(facility_of_client[client])
        
        uncovered_clients = []
        for client in client_indices:
            covered = False
            for facility in open_facilities:
                if dist_mat[facility][client] <= R_j[client]:
                    covered = True
            if covered == False:
                uncovered_clients.append(client)
        
        if len(uncovered_clients) > 0:
            A_t = copy.deepcopy(active_clients)
            for client in uncovered_clients:
                active_clients.append(client)
                radius[client] = gamma * R_j[client]
                B_j = find_ball(client, dist_mat, facility_indices, radius)
                # pick one to be the associated facility since costs are uniform
                facility = B_j[0]
                facility_of_client[client] = facility
                open_facilities.append(facility)
                
                for diff_client in A_t:
                    if diff_client != client and dist_mat[client][diff_client] <= radius[client] + radius[diff_client] + delta * min(radius[client], radius[diff_client]):
                        active_clients.remove(diff_client)
                        open_facilities.remove(facility_of_client[diff_client])
        
        t += 1

    return x_OPT, open_facilities, OPT_recourse, total_rounding_recourse


#####################################################################################################################
################################################### main ############################################################
##################################################################################################################### 

## parse inputs
# Generate random points
np.random.seed(42)
all_points = np.random.rand(200, 2) * 100  # 100 points in a 100x100 grid
data_points = random.sample(list(all_points), 10)
#candidate_locations = data_points
#plot_points(data_points)
#print(data_points)

# We'll add 20% of the amout of data to be removal requests
# to simulate dynamic streaming.
# For simplicity, whenever we encounter a removal request,
# we randomly sample an active client point that is not in the set of centers
# In our request array. a +1 indicates an insertion of a client;
# -1 indicates a removal.
requests = np.ones(int(len(data_points)))
#removals = np.random.choice(range(0, len(data_points)+ 1), int(len(data_points)*0.2), replace=False)
#print(removals)
#requests[removals] = -1
#print(requests)

# First Solve the offline k-center problem
# Calculate LP
fractional_sol, open_facilities, OPT_rec, recourse = online_facility_location(requests, data_points, cost)

print("\n")
#print("Approx maximum distance to nearest center:", max_dist_approx)

# begin online algorithm
#fractional_sol, recourse, centers, OPT_rec = online_k_center(requests, data_points, k) 

print("-----------final online results-----------")
print("beta = ", beta)
print("epsilon = ", epsilon)
print("final fractional solution:", fractional_sol)
print("OPT recourse:", OPT_rec)
print("total online recourse:", recourse)
#print("final selected centers:", centers)

# (optional) for plotting
#for i in range(len(centers)):
    #center_coordinates[i] = data_points[centers[i]]
#plot_points_and_centers(data_points, center_coordinates)

OPT_cost = lp_relaxation_facility_location(data_points, data_points, cost)
print("OPT cost from lp relaxation:", OPT_cost)

print("max online cost:", OPT_cost)

print("alpha * beta * offline OPT cost:", alpha * beta * OPT_cost)

# Plot the points and the selected centers
#plot_points_and_centers(data_points, approx_centers)







