import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import pulp
from sklearn.datasets import make_blobs


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

# greedy approximation to compute OPT_distance
def offline_k_center(points, k):
    # Initialize the first center randomly
    centers = [points[np.random.randint(len(points))]]
    
    while len(centers) < k:
        # Find the point that is the farthest from any center
        next_center = max(points, key=lambda point: min(euclidean_distance(point, center) for center in centers))
        centers.append(next_center)
    
    return centers

# LP relaxation for offline k-center's OPT_distance 
def lp_relaxation_k_center(points, k):
    num_points = len(points)
    prob = pulp.LpProblem("k_Center", pulp.LpMinimize)

    # populate distance matrix
    dist_mat = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            dist_mat[i, j] = euclidean_distance(points[i], points[j])
        print(max(dist_mat[i]))
    
    #print("distance matrix:\n", dist_mat)

    # Variables
    x = pulp.LpVariable.dicts("x", (range(num_points), range(num_points)), lowBound=0, upBound=1, cat=pulp.LpContinuous)
    y = pulp.LpVariable.dicts("y", range(num_points), lowBound=0, upBound=1, cat=pulp.LpContinuous)
    z = pulp.LpVariable("z", lowBound=0, cat=pulp.LpContinuous)

    # Objective
    prob += z, "Maximum distance to nearest center"

    # Constraints
    for i in range(num_points):
        prob += pulp.lpSum(x[i][j] for j in range(num_points)) == 1, f"Assign_{i}"
        for j in range(num_points):
            prob += x[i][j] <= y[j], f"Link_{i}_{j}"
            prob += dist_mat[i][j] * x[i][j] <= z, f"Distance_{i}_{j}"

    prob += pulp.lpSum(y[j] for j in range(num_points)) == k, "Number_of_centers"

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Print the results
    #print("Status:", pulp.LpStatus[prob.status])
    #print("The minimized maximum distance z is:", pulp.value(z))
    '''
    print("y[j] values:")
    for j in range(num_points):
        print(f"y[{j}] = {y[j].varValue}")
    '''

    #y_sum = pulp.lpSum(y[j].varValue for j in range(num_points))
    
    # print out the x matrix for debugging
    x_mat = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            x_mat [i, j] = x[i][j].varValue
    #print("x_mat:\n", x_mat)

    z = np.zeros(num_points)
    for i in range(num_points):
        for j in range(num_points):
            z[i] += x_mat[i][j] * dist_mat[i][j]
        #print(z[i])

    #print("sum of y:", y_sum)
    
    result = np.max(z)
    
    return result


# used in conjunction with greedy approximation to calculate the max distance 
# from a point to its nearest center
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

def plot_points(points):
    plt.figure(figsize=(8, 6))
    x, y = zip(*points)
    #cx, cy = zip(*centers)
    
    plt.scatter(x, y, color='blue', label='Points')
    #plt.scatter(cx, cy, color='red', s=100, label='Centers', edgecolors='black')
    plt.title("Offline k-Center Clustering")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

####################################### online positive-body chasing for k-center ##################################################

# setting parameters needed for online algorithm
beta = 1

epsilon = 0.25

####################################### functions needed for main method ############################################################

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
def find_candidates(client_indices, points, client, radius):
    s = []
    #print("candidate set:", client_indices)
    #print("points:", points)
    for index in client_indices:
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


# alternative method to update x's when a covering violation occurs
# we use the closed form with the lagrange multiplier here
def update_covering(x, s, epsilon):
    d = len(s)
    log_term = (epsilon/4 + 1) / (np.sum(x[s]) + epsilon / 4)

    if log_term > 1:
        y = np.log(log_term)
        x[s] = (x[s] + epsilon / (4 * d)) * np.exp(y) - (epsilon / (4 * d))

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
#   C: a list of length T that consists of covering constraints at time t; 
#      C(t) is the indices of non-zero c_i's that appear in a covering constraint at time t: 
#      i.e., C(t) * x < 1; 
#   P: a list of packing contraints at each t defined similarly to C
#   t: index of current iteration (starts from 0)
def compute_OPT_rec(C_list, P_list, t, k, epsilon, client_indices):

    # Problem data and parameters
    T = t + 1  # Number of time periods
    n = t + 1  # Number of total variables (including removed clients)
    # Create the LP problem object
    lp_prob = pulp.LpProblem("OPT_recourse", pulp.LpMinimize)

    # Decision variables x_i^t and l_i^t
    x = pulp.LpVariable.dicts("x", (range(T), range(n)), lowBound=0, upBound=1, cat=pulp.LpContinuous)
    l = pulp.LpVariable.dicts("l", (range(T), range(n)), lowBound=0, cat=pulp.LpContinuous)
    #z = pulp.LpVariable("z", lowBound=0)

    # Objective function
    lp_prob += pulp.lpSum(l[t][i] for i in range(n) for t in range(T))

    # Constraints
    for t in range(len(C_list)):
        C_t = C_list[t]
        #print("all covering constraints at this iteration:", C_t)
        for c in range(len(C_t)):
            subset_indices = C_t[c]
            #print("violating covering constraint(subset of indices):", subset_indices)
            lp_prob += pulp.lpSum(x[t][i] for i in subset_indices) >= 1  
    for t in range(len(P_list)):
        P = P_list[t]
        #print("Packing constraints this iteration", P)
        #print("number of active points this iteration:", len(P))
    
        #print("value of packing_min this iteration:", packing_min)
        #lp_prob += z <= (1 + epsilon) * k
        #lp_prob += z <= len(P)
        lp_prob += pulp.lpSum(x[t][i] for i in P) <= (1 + epsilon) * k  
        # make sure that removed points are set to 0
        for i in range(len(client_indices)):
            if i not in P:
                lp_prob += x[t][i] == 0
    
    for i in range(n):
        # include the recourse at t = 0
        lp_prob += x[0][i] <= l[0][i]
    for t in range(1, T):
        for i in range(n):
            lp_prob += (x[t][i] - x[t-1][i]) <= l[t][i] #, f"ChangeConstraint_Pos_{t}_{i}"
            lp_prob += (x[t-1][i] - x[t][i]) <= l[t][i] #, f"ChangeConstraint_Neg_{t}_{i}"

    # Solve the problem
    lp_prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Output results
    #print("The packing target:", pulp.value(z))
    
    # printing x and l matrices for debugging
    x_mat = np.zeros((T, n))
    l_mat = np.zeros((T, n))

    for i in range(T):
        for j in range(n):
            x_mat [i, j] = x[i][j].varValue
            l_mat[i, j] = l[i][j].varValue
            
    
    #print("Result from offline OPT_rec:")
    #print("X_OPT: \n", x_mat)
    #print("l: \n", l_mat)

    #print("OPT_recourse this iteration = ", pulp.value(lp_prob.objective))
    
    # return the last row of the x matrix as the fractional solution for rounding
    return pulp.value(lp_prob.objective), x_mat[-1]


######################################### helper for rounding ###########################################

# set the parameters for rounding
alpha = 1 + 2 * np.sqrt(2)
delta = np.sqrt(2)

# subroutine to find the balls B_i and B_hait_i for a given center_index
# this subroutine is called whenever the set S is updated
def find_balls(data_points, client_indices, center_index, radius, radius_hat):

    #print("inside find_ball function")
    B_i = []
    B_i_hat =[]

    for index in client_indices:
        #print("current client:", j)
        #print("current center:", s[i])

        #print("client coordinate:", data_points[j])
        #print("center coordinate:", data_points[s[i]])
        #print("distance:", euclidean_distance(data_points[j], data_points[s[i]]))
        #client_coordinate = clients[i][1]
        #client_index = clients[i][0]
        #print("distance between client and center:", euclidean_distance(client_coordinate, data_points[center_index]))
        if euclidean_distance(data_points[index], data_points[center_index]) <= radius:
            B_i.append(index)
            
        if euclidean_distance(data_points[index], data_points[center_index]) <= radius_hat:
            B_i_hat.append(index)

    #print("points in B_i:", B_i)
    print("points in B_i_hat:", B_i_hat)
    
    return B_i, B_i_hat

###################################### main method for online k-center #######################################

def online_k_center(requests, points, k):

    total_recourse = 0
    total_integer_recourse = 0

    # initialize the vector x with all 0s of dimension len(points)
    x = np.zeros(len(points))

    # store each client as a tuple (index in points, coordinates)
    clients = []
    # the coordinates of active clients
    client_points = []
    client_indices = []

    set_of_centers = []
    radius_of_centers = np.zeros(len(x))

    # (optional) may needed for dynamic streaming
    # stores the number of total requests
    # will be modified when removal requests are added
    num_requests = len(points)

    # initialize variables needed for computing OPT_rec:
    # a t-by-n list of lists C, whose number of rows grows as t increases,
    # each row stores the covering constraints of time t
    # A matrix P for packing constraints is defined similarly
    C_list = []
    P_list = []
    #violated_covering_t = []
    #violated_packing_t = []

    #x_closed = np.zeros(len(x))

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
            if len(client_indices) and len(client_indices) > len(set_of_centers) > 0:
                client_to_remove = random.randint(0, t)
                while client_to_remove in set_of_centers and client_to_remove not in client_indices:
                    client_to_remove = random.randint(0, t)
                
                x[client_to_remove] = 0
                #print(client_indices)
                #print("client to remove:", client_to_remove)
                if client_to_remove in client_indices:
                    client_indices.remove(client_to_remove)
                
                    item_to_remove = points[client_to_remove]
                    for i in range(len(client_points)):
                        if np.array_equal(client_points[i], item_to_remove):
                            client_points.pop(i)
                            break

                    for client in clients:
                        if client[0] == client_to_remove:
                            clients.remove(client)
                    #print("removal request")
                    #print("client removed:", client_to_remove)
            continue
        
        # the request is an insertion, add the new client and deal with any constraint violations
        # set of active clients
        current_client = (t, points[t])
        current_client_coordinates = points[t]
        clients.append(current_client)
        client_points.append(points[t])
        #print(client_points)
        client_indices.append(t)
        #print(client_indices)
        
        
        #print("\n")
        #print("t= ", t)
        #print("current client:", points[t])

        # search for points within the radius of the current client
        # the radius at each t is defined as: min(diam(t), beta * OPT(t))
        diam = calculate_diameter(client_points)
        centers_offline = offline_k_center(client_points, k)
        approx_dist = max_distance_to_centers(client_points, centers_offline)
        current_OPT_dist = lp_relaxation_k_center(client_points, k)

        #print("diam(t):", diam)
        print("current_OPT_dist:", current_OPT_dist)
        #print("approx OPT dist:", approx_dist)

        # stores all covering constraints at t
        C_t =[]
        s = find_candidates(client_indices, points, current_client_coordinates, min(beta * current_OPT_dist, diam))
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
            #violated_covering_t.append(t)
            x_new = update_covering(x, s, epsilon)
            #print("updated solution from closed-form (covering):", x_new)
        #else:
            # no covering constraints are violated, add empty list at row t 
            #C_list.append([])

        # check if packing constraint is violated
        # if so, update the values of x's
        if (np.sum(x) > (1+epsilon) * k):
            #print("Solving packing constraints")
            # the packing constraint takes the sum over all active clients
            #P_list.append(client_indices)
            #violated_packing_t.append(t)
            x_new = update_packing(x, epsilon, k)
            #print("updated solution from closed-form (packing):", x_new)
        #else:
            #P_list.append([])

        # record the packing constraint for this iteration
        #print("active clients this round (packing constraint):", client_indices)
        active_set = []
        active_set = copy.deepcopy(client_indices)
        #packing_min = min((1 + epsilon) * k, len(active_set))
        P_list.append(active_set)
        #print("list of packing constraints so far:", P_list)

        # recheck again the covering constraint's for all points
        #print("\nrecheck for covering violations...")

        #print("current client set:", client_indices)
        for client_index in client_indices:
            #print("for client:", client_index)
            s_set = find_candidates(client_indices, points, points[client_index], min(beta * current_OPT_dist, diam))
            C_t.append(s_set)

            if (check_covering_feasibility(s_set, x_new) == False):
                #print("re-adjusting for covering constraints...")
                x_new = update_covering(x_new, s_set, epsilon)
                #print("re-updated solution after resolving covering constraint:", x_new)
        
        # record all covering constraints for this round
        C_list.append(C_t)
        #print("all covering constraints at this iteration:", C_t)
        
        #print("\nComputing OPT recourse...")
        OPT_recourse, x_OPT = compute_OPT_rec(C_list, P_list, t, k, epsilon, client_indices)

        # update total recourse in l1-norm
        total_recourse += np.sum(np.abs(x_new - x_old))
        #print("total online recourse so far:", total_recourse)
        x = x_new

        #print("fractional solution this round:", x)
        #print("number of k this round:", np.sum(x_new))    
        
        #################################### rounding procedure begins from here ####################################
        # integrate rounding at each round
        # maintain a set S (initially empty) of open centers at each t
        # r_i of each 'center' is stored at radius_of_centers[center]
        
        # identify the balls/set of points that are B_i and B_i_hat for each i in set_of_centers
        # while building the balls B_i, drop any i from set_of_centers if it has mass less than 1-epsilon

        #print("\n")
        #print("Rounding begins")
        #print("current set of centers:", set_of_centers)

        #list_of_B_i = []
        list_of_B_i_hat = []
        S = copy.deepcopy(set_of_centers)
        for center in S:

            if center not in client_indices: # this means 'center' was a removed point at some t
                set_of_centers.remove(center)
                total_integer_recourse += 1
                #print("updated set of centers:", set_of_centers)
                continue
            
            print("\n")
            print("for center:", center)
            #print("r_i of this center:", radius_of_centers[center])
            print("r_i_hat of this center:", alpha * min(beta * current_OPT_dist, diam))
            
            B_i, B_i_hat = find_balls(data_points, client_indices, center, radius_of_centers[center], alpha * min(beta * current_OPT_dist, diam))
            
            # drop any B_i whose mass is too small
            mass = 0
            for index_of_point in B_i:
                mass += x_OPT[index_of_point]  # currently using the solution returned by OPT_rec for rounding
            
            print("mass for center", mass)

            if mass < 1 - epsilon:
                set_of_centers.remove(center)
                total_integer_recourse += 1
                print("center dropped from set:", center)
            else:
                list_of_B_i_hat.append(B_i_hat)
        
        covered_points = set(item for sublist in list_of_B_i_hat for item in sublist)
        #print("\n")
        print("covered points:", covered_points)
        print("all clients:", client_indices)
        while len(covered_points) < len(client_indices):
        # find the clients that are not covered by the current set of centers
            uncovered = set(client_indices) - covered_points
            print("uncovered clients:", uncovered)

            j = next(iter(uncovered))
            set_of_centers.append(j)
            set_of_centers = list(set(set_of_centers)) # get rid of duplicates in the list
            total_integer_recourse += 1
            # record current_radius
            current_r = min(beta * current_OPT_dist, diam)
            radius_of_centers[j] = current_r

            print("new center added to set:", j)
            print("With radius_i:", current_r)

            S = copy.deepcopy(set_of_centers)
            for center_index in S:
                ball_dist = radius_of_centers[j] + radius_of_centers[center_index] + delta * min(radius_of_centers[j], radius_of_centers[center_index])
                print("distance between two centers:", euclidean_distance(points[center_index], points[j]))
                print("ball disjoint-radius:", ball_dist)
                if center_index != j and euclidean_distance(points[center_index], points[j]) <= ball_dist:
                    set_of_centers.remove(center_index)
                    total_integer_recourse += 1
                    print("center {center_index} dropped", center_index)

            # update the the set of B_i_hats
            print("update ball...")
            list_of_B_i_hat = []
            for center in set_of_centers:
                B_i, B_i_hat = find_balls(points, client_indices, center, radius_of_centers[center], alpha * min(beta * current_OPT_dist, diam))
                list_of_B_i_hat.append(B_i_hat)
            covered_points = set(item for sublist in list_of_B_i_hat for item in sublist)
            print("covered points:", covered_points)
        
        print("all clients covered!")
        #print("selected centers for this round:", set_of_centers)
        #print("total integer recourse so far:", total_integer_recourse)

        t += 1

    return x, total_recourse, set_of_centers, OPT_recourse, total_integer_recourse


#####################################################################################################################
################################################### main ############################################################
##################################################################################################################### 

################################################## parse inputs #####################################################

# Method 1: Generate random points
np.random.seed(42)
random_points = np.random.rand(100, 2) * 100  # 100 points in a 100x100 grid
#plot_points(data_points)
#print(data_points)

# Method 2: generate test points in clusters 
# Settings for the clusters
n_samples = 100          # Total number of points
n_features = 2           # Number of dimensions (2D)
centers = 4              # Number of clusters
cluster_std = 1.0        # Standard deviation of clusters
cluster_points, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=42)
print(y)

# set this variable to either random_points or cluster_points
data_points = cluster_points


# Preliminary simulation of dynamic streaming:
# We'll add 20% of the amout of data to be removal requests
# to simulate dynamic streaming.
# For simplicity, whenever we encounter a removal request,
# we randomly sample an active client point that is not in the set of centers
# In our request array. a +1 indicates an insertion of a client;
# -1 indicates a removal.
requests = np.ones(int(len(data_points) * 1.2))
removals = np.random.choice(range(0, len(data_points)+ 1), int(len(data_points)*0.2), replace=False)
#print(removals)
requests[removals] = -1
#print(requests)


# Number of centers
k = 4

# Solve the offline k-center problem
approx_centers = offline_k_center(data_points, k)
#print("approx centers:", approx_centers)

# Calculate the maximum distance to the nearest center
# This value used as input parameter in the online problem
max_dist = lp_relaxation_k_center(data_points, k)
max_dist_approx = max_distance_to_centers(data_points, approx_centers)

# Plot the points and the selected centers
#plot_points_and_centers(data_points, approx_centers)

print("\n")
#print("Approx maximum distance to nearest center:", max_dist_approx)
print("OPT distance from lp relaxation:", max_dist)

# begin calculation of the online problem
fractional_sol, recourse, centers, OPT_rec, total_int_recourse = online_k_center(requests, data_points, k) 

print("-----------final online results-----------")
print("k = ", k)
print("beta = ", beta)
print("epsilon = ", epsilon)
print("alpha:", alpha)
#print("final fractional solution:", fractional_sol)
print("number of centers (sum of fractional x's):", np.sum(fractional_sol))
print("OPT recourse:", OPT_rec)
print("total fractional online recourse:", recourse)
print("final selected centers:", centers)
print("total rounding recourse:", total_int_recourse)

center_coordinates = random.sample(list(data_points), len(centers))
for i in range(len(centers)):
    center_coordinates[i] = data_points[centers[i]]
max_online_dist = max_distance_to_centers(data_points, center_coordinates)

print("max online distance:", max_online_dist)

print("alpha * beta * offline max distance:", alpha * beta * max_dist)

# Plot the points and the selected centers
plot_points_and_centers(data_points, center_coordinates)








