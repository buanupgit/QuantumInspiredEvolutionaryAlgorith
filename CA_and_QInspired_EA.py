import time
import random
import numpy as np
import matplotlib.pyplot as plt

# ================================================================================
# Configuration Parameters
# ================================================================================
SEED = 42                    # Random seed for reproducible results
POP_SIZE = 25               # Population size for both algorithms
GENERATIONS = 500           # Maximum number of generations to evolve
CROSSOVER_RATE = 0.9        # Probability of crossover operation
MUTATION_RATE = 0.2         # Base mutation rate (adaptive during evolution)
ROTATION_ANGLE = 0.08 * np.pi  # Base rotation angle for quantum-inspired updates
ELITISM = True              # Enable elitist selection to preserve best solutions
PLOT_EVERY = 50            # Display progress every N generations

# Initialize random number generators for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# ================================================================================
# Problem Instance Setup
# ================================================================================
# Vehicle Routing Problem configuration with fixed depot and random customers
np.random.seed(42)  # Ensures consistent customer placement across runs

DEPOT = np.array([50.0, 50.0])  # Central depot location
N_CUSTOMERS = 10                # Number of customers to serve (configurable)

# Generate customer coordinates uniformly distributed in a 100x100 grid
CUSTOMERS = np.random.uniform(low=0.0, high=100.0, size=(N_CUSTOMERS, 2))

# ================================================================================
# Vehicle Routing Problem Class
# ================================================================================
class VRPInstance:
    """
    Encapsulates the Vehicle Routing Problem instance with geometric data
    and utility methods for route evaluation.
    
    This class manages:
    - Depot and customer locations
    - Distance matrix computation
    - Route distance calculation
    - Multi-objective fitness evaluation
    """
    def __init__(self, customers, depot):
        """
        Initialize VRP instance with customer locations and depot.
        
        Args:
            customers: Array of customer coordinates
            depot: Depot coordinate array
        """
        self.depot = np.array(depot)
        self.customers = np.array(customers)
        self.n = len(customers)
        
        # Build comprehensive distance matrix including depot (index 0) and all customers
        pts = [self.depot] + list(self.customers)
        self.dist = np.zeros((self.n + 1, self.n + 1))
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                self.dist[i, j] = np.linalg.norm(np.array(pts[i]) - np.array(pts[j]))

    def route_distance(self, route):
        """
        Calculate total distance for a route starting and ending at depot.
        
        Args:
            route: List of customer indices (1-based) defining the route
            
        Returns:
            Total distance including depot-to-first and last-to-depot segments
        """
        if not route:
            return 0.0
        
        # Distance from depot to first customer
        d = self.dist[0, route[0]]
        
        # Sum distances between consecutive customers
        for i in range(len(route) - 1):
            d += self.dist[route[i], route[i + 1]]
        
        # Distance from last customer back to depot
        d += self.dist[route[-1], 0]
        return d

    def objectives_from_routes(self, r1, r2):
        """
        Compute multi-objective fitness from two complete routes.
        
        Args:
            r1, r2: Two vehicle routes as lists of customer indices
            
        Returns:
            Tuple of (total_distance, workload_imbalance)
        """
        d1 = self.route_distance(r1)
        d2 = self.route_distance(r2)
        return d1 + d2, abs(d1 - d2)

    def objectives_from_seq_split(self, seq, split):
        """
        Convert permutation sequence and split point into two routes for evaluation.
        
        Args:
            seq: Permutation of all customer indices
            split: Index where to split sequence into two routes
            
        Returns:
            Multi-objective fitness tuple
        """
        r1 = seq[:split]
        r2 = seq[split:]
        return self.objectives_from_routes(r1, r2)

# Create global VRP instance for use by both algorithms
vrp = VRPInstance(CUSTOMERS, DEPOT)

# ================================================================================
# Multi-Objective Optimization Utilities
# ================================================================================
def dominates(a, b):
    """
    Check if solution 'a' Pareto-dominates solution 'b'.
    
    Args:
        a, b: Objective vectors to compare
        
    Returns:
        True if 'a' dominates 'b' (better in all objectives, strictly better in at least one)
    """
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def non_dominated_sort(objs):
    """
    Perform fast non-dominated sorting as used in NSGA-II algorithm.
    
    Args:
        objs: List of objective vectors for all solutions
        
    Returns:
        Tuple of (fronts, ranks) where:
        - fronts: List of fronts, each containing solution indices
        - ranks: Pareto rank for each solution (0 = best front)
    """
    N = len(objs)
    S = [[] for _ in range(N)]  # Solutions dominated by solution p
    n = [0] * N                 # Number of solutions dominating solution p
    rank = [0] * N              # Pareto rank of each solution
    fronts = [[]]               # List of fronts
    
    # Build domination relationships
    for p in range(N):
        for q in range(N):
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1
        
        # Solutions with no dominators belong to first front
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    
    # Build subsequent fronts
    i = 0
    while fronts[i]:
        next_f = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_f.append(q)
        i += 1
        fronts.append(next_f)
    
    return fronts[:-1], rank  # Remove empty last front

def crowding_distance(front, objs):
    """
    Calculate crowding distance for diversity preservation within Pareto fronts.
    
    Args:
        front: List of solution indices in the same Pareto front
        objs: Objective vectors for all solutions
        
    Returns:
        Dictionary mapping solution indices to their crowding distances
    """
    l = len(front)
    if l == 0:
        return {}
    if l == 1:
        return {front[0]: float('inf')}
    
    M = len(objs[0])  # Number of objectives
    cd = {i: 0.0 for i in front}
    
    # Calculate crowding distance for each objective
    for m in range(M):
        # Sort solutions by m-th objective value
        vals = sorted(front, key=lambda i: objs[i][m])
        
        # Boundary solutions get infinite distance
        cd[vals[0]] = cd[vals[-1]] = float('inf')
        
        vmin = objs[vals[0]][m]
        vmax = objs[vals[-1]][m]
        
        # Skip if all values are identical
        if vmax - vmin < 1e-12:
            continue
        
        # Assign distances proportional to objective space gaps
        for k in range(1, l - 1):
            cd[vals[k]] += (objs[vals[k + 1]][m] - objs[vals[k - 1]][m]) / (vmax - vmin)
    
    return cd

def hypervolume_2d(front, ref_point):
    """
    Calculate 2D hypervolume indicator for solution quality assessment.
    
    Args:
        front: List of objective vectors on the Pareto front
        ref_point: Reference point for hypervolume calculation
        
    Returns:
        Hypervolume value (higher is better for minimization problems)
    """
    if not front:
        return 0.0
    
    # Sort points by first objective for sweep line algorithm
    pts = sorted(front, key=lambda x: x[0])
    hv = 0.0
    prev_f1 = ref_point[0]
    
    # Calculate area dominated by each point
    for f1, f2 in pts:
        width = prev_f1 - f1
        height = ref_point[1] - f2
        if width > 0 and height > 0:
            hv += width * height
        prev_f1 = f1
    
    return hv

def spacing_metric(front):
    """
    Calculate spacing metric to measure distribution uniformity of Pareto front.
    
    Args:
        front: List of objective vectors on the Pareto front
        
    Returns:
        Spacing value (lower is better - indicates more uniform distribution)
    """
    if len(front) < 2:
        return 0.0
    
    # Calculate minimum distance to nearest neighbor for each solution
    dists = []
    for i in range(len(front)):
        min_d = float('inf')
        for j in range(len(front)):
            if i == j:
                continue
            d = np.linalg.norm(np.array(front[i]) - np.array(front[j]))
            if d < min_d:
                min_d = d
        dists.append(min_d)
    
    # Return standard deviation of distances
    mean_d = np.mean(dists)
    return np.sqrt(np.mean((dists - mean_d) ** 2))

# ================================================================================
# Classical Evolutionary Algorithm with Anti-Stagnation
# ================================================================================
def order_crossover(p1, p2):
    """
    Perform Order Crossover (OX) operation for permutation representation.
    
    This crossover preserves the relative ordering of elements while
    combining genetic material from both parents.
    
    Args:
        p1, p2: Parent permutations
        
    Returns:
        Offspring permutation maintaining valid customer assignments
    """
    n = len(p1)
    
    # Select random crossover segment
    a, b = sorted(random.sample(range(n), 2))
    
    # Initialize child and copy segment from parent 1
    child = [None] * n
    child[a:b] = p1[a:b]
    
    # Fill remaining positions with elements from parent 2 in order
    fill = [g for g in p2 if g not in child]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    
    return child

def swap_mutation(seq, intensive=False):
    """
    Apply swap mutation to permutation sequence.
    
    Args:
        seq: Permutation sequence to mutate
        intensive: If True, applies multiple swaps for stronger perturbation
        
    Returns:
        Mutated sequence with preserved element set
    """
    if len(seq) < 2:
        return seq
    
    if intensive:
        # Apply multiple swaps for escape from local optima
        num_swaps = random.randint(2, 4)
        for _ in range(num_swaps):
            i, j = random.sample(range(len(seq)), 2)
            seq[i], seq[j] = seq[j], seq[i]
    else:
        # Standard single swap mutation
        i, j = random.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]
    
    return seq

def split_mutation(split, n_customers, intensive=False):
    """
    Mutate the split point that divides customers between two vehicles.
    
    Args:
        split: Current split point index
        n_customers: Total number of customers
        intensive: If True, allows larger jumps for diversification
        
    Returns:
        New split point ensuring both routes have at least one customer
    """
    if intensive:
        # Larger perturbation for stagnation recovery
        delta = random.randint(-n_customers//4, n_customers//4)
    else:
        # Conservative local search-style mutation
        delta = random.choice([-2, -1, 1, 2])
    
    new_split = split + delta
    
    # Ensure split maintains valid route structure
    new_split = max(1, min(n_customers - 1, new_split))
    return new_split

def init_classical_population(pop_size, n_customers):
    """
    Initialize population for classical evolutionary algorithm.
    
    Args:
        pop_size: Number of individuals to generate
        n_customers: Number of customers in the problem
        
    Returns:
        List of (sequence, split) tuples representing initial solutions
    """
    pop = []
    for _ in range(pop_size):
        # Generate random permutation of customer indices
        seq = list(np.random.permutation(np.arange(1, n_customers + 1)))
        # Random split point ensuring both routes are non-empty
        split = random.randint(1, n_customers - 1)
        pop.append((seq, split))
    return pop

def classical_ea(vrp, pop_size=POP_SIZE, gens=GENERATIONS):
    """
    Classical Multi-Objective Evolutionary Algorithm with advanced features.
    
    Features implemented:
    - NSGA-II-style selection with Pareto ranking and crowding distance
    - Anti-stagnation mechanism with adaptive parameters
    - Strong elitism to preserve best solutions
    - Dynamic diversity injection during stagnation periods
    
    Args:
        vrp: VRP problem instance
        pop_size: Population size
        gens: Number of generations
        
    Yields:
        Progress snapshots containing current best solution and metrics
    """
    # Initialize population and tracking variables
    pop = init_classical_population(pop_size, vrp.n)
    archive = []  # Store all evaluated solutions
    convergence = []  # Track hypervolume over generations
    start = time.time()
    
    # Global best tracking for elitism
    global_best = None
    global_best_fitness = float('inf')
    
    # Stagnation detection parameters
    stagnation_counter = 0
    stagnation_threshold = 30
    last_improvement_fitness = float('inf')

    for g in range(gens + 1):
        # Evaluate current population
        objs = [vrp.objectives_from_seq_split(s, sp) for s, sp in pop]
        
        # Archive all solutions for final analysis
        for i, (s, sp) in enumerate(pop):
            r1 = s[:sp]
            r2 = s[sp:]
            archive.append((s[:], sp, objs[i], r1, r2))
        
        # Update global best and monitor improvement
        best_idx = min(range(len(pop)), key=lambda i: objs[i][0])
        current_best_fitness = objs[best_idx][0]
        
        if current_best_fitness < global_best_fitness:
            global_best = (pop[best_idx][0][:], pop[best_idx][1])
            global_best_fitness = current_best_fitness
            
            # Check for meaningful improvement (threshold-based)
            if current_best_fitness < last_improvement_fitness * 0.999:
                stagnation_counter = 0
                last_improvement_fitness = current_best_fitness
            else:
                stagnation_counter += 1
        else:
            stagnation_counter += 1
        
        # Detect and handle stagnation
        is_stagnated = stagnation_counter >= stagnation_threshold
        
        if is_stagnated:
            print(f"Gen {g}: Stagnation detected! Applying diversity enhancement...")
            stagnation_counter = 0  # Reset stagnation counter
        
        # Multi-objective ranking and diversity assessment
        fronts, ranks = non_dominated_sort(objs)
        cd = {}
        for f in fronts:
            cd.update(crowding_distance(f, objs))
        
        # Calculate convergence metric (hypervolume)
        if fronts:
            front_objs = [objs[i] for i in fronts[0]]
            ref = (max(o[0] for o in front_objs) * 1.1, 
                   max(o[1] for o in front_objs) * 1.1)
            hv = hypervolume_2d(front_objs, ref)
        else:
            hv = 0.0
        convergence.append(hv)
        
        # Provide progress snapshot
        if g % PLOT_EVERY == 0:
            if global_best:
                best_r1 = global_best[0][:global_best[1]]
                best_r2 = global_best[0][global_best[1]:]
            else:
                best_seq, best_split = pop[best_idx]
                best_r1 = best_seq[:best_split]
                best_r2 = best_seq[best_split:]
            yield ('classical', g, best_r1, best_r2, archive[:], convergence[:], time.time() - start)
        
        # Generate next generation
        if g < gens:
            new_pop = []
            
            # Elitism: preserve global best solution
            if ELITISM and global_best:
                new_pop.append(global_best)
            
            # Diversity injection during stagnation
            if is_stagnated:
                num_immigrants = int(pop_size * 0.3)  # Inject 30% fresh solutions
                for _ in range(num_immigrants):
                    seq = list(np.random.permutation(np.arange(1, vrp.n + 1)))
                    split = random.randint(1, vrp.n - 1)
                    new_pop.append((seq, split))
            
            # Adaptive mutation rate based on stagnation proximity
            adaptive_mutation_rate = MUTATION_RATE
            if stagnation_counter > stagnation_threshold // 2:
                # Gradually increase mutation as stagnation approaches
                adaptive_mutation_rate = min(0.8, MUTATION_RATE * (1 + stagnation_counter / stagnation_threshold))
            
            # Generate offspring through selection, crossover, and mutation
            while len(new_pop) < pop_size:
                # Tournament selection based on Pareto rank and crowding distance
                t_size = 3  # Tournament size
                tournament = random.sample(range(len(pop)), t_size)
                p1_idx = min(tournament, key=lambda i: (ranks[i], -cd.get(i, 0)))
                p1 = pop[p1_idx]
                
                tournament = random.sample(range(len(pop)), t_size)
                p2_idx = min(tournament, key=lambda i: (ranks[i], -cd.get(i, 0)))
                p2 = pop[p2_idx]
                
                # Order crossover for permutation
                if random.random() < CROSSOVER_RATE:
                    child_seq = order_crossover(p1[0], p2[0])
                    # Inherit split from superior parent or use average
                    if ranks[p1_idx] < ranks[p2_idx]:
                        child_split = p1[1]
                    elif ranks[p2_idx] < ranks[p1_idx]:
                        child_split = p2[1]
                    else:
                        # Average split with small random perturbation
                        avg_split = (p1[1] + p2[1]) // 2
                        child_split = max(1, min(vrp.n - 1, 
                                                avg_split + random.randint(-1, 1)))
                else:
                    child_seq = p1[0][:]
                    child_split = p1[1]
                
                # Adaptive mutation with intensity based on stagnation
                if random.random() < adaptive_mutation_rate:
                    child_seq = swap_mutation(child_seq, intensive=is_stagnated)
                
                # Split point mutation
                if random.random() < adaptive_mutation_rate * 0.5:
                    child_split = split_mutation(child_split, vrp.n, intensive=is_stagnated)
                
                new_pop.append((child_seq, child_split))
            
            # Environmental selection using NSGA-II survivor selection
            combined = pop + new_pop
            combined_objs = [vrp.objectives_from_seq_split(s, sp) for s, sp in combined]
            fronts, _ = non_dominated_sort(combined_objs)
            
            next_pop = []
            for f in fronts:
                if len(next_pop) + len(f) <= pop_size:
                    # Include entire front if space permits
                    next_pop.extend([combined[i] for i in f])
                else:
                    # Select individuals with highest crowding distance
                    cd_f = crowding_distance(f, combined_objs)
                    sorted_f = sorted(f, key=lambda i: cd_f[i], reverse=True)
                    remaining = pop_size - len(next_pop)
                    next_pop.extend([combined[i] for i in sorted_f[:remaining]])
                    break
            
            pop = next_pop
    
    return archive, convergence, time.time() - start

# ================================================================================
# Quantum-Inspired Evolutionary Algorithm with Anti-Stagnation
# ================================================================================
class QuantumInspiredEA:
    """
    Quantum-Inspired Evolutionary Algorithm for multi-objective VRP optimization.
    
    Key features:
    - Quantum bit (qubit) representation with probability amplitudes
    - Adaptive rotation operators for solution improvement
    - Multi-objective fitness evaluation and Pareto-based selection
    - Anti-stagnation mechanisms with quantum perturbation
    - Strong elitism to ensure monotonic progress
    
    The algorithm maintains a population of quantum individuals, each represented
    by probability amplitudes that encode both route assignment and customer ordering.
    """

    def __init__(self, vrp, pop_size=POP_SIZE, generations=GENERATIONS,
                 crossover_rate=CROSSOVER_RATE,
                 base_rotation_angle=ROTATION_ANGLE,
                 mutation_prob=0.05):
        """
        Initialize Quantum-Inspired EA with problem parameters.
        
        Args:
            vrp: VRP problem instance
            pop_size: Population size
            generations: Maximum generations
            crossover_rate: Probability of crossover
            base_rotation_angle: Base angle for quantum rotations
            mutation_prob: Base probability for amplitude mutations
        """
        self.vrp = vrp
        self.n = vrp.n
        self.n_qbits = self.n * 2  # n bits for assignment, n bits for ordering
        self.population_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.base_rotation_angle = base_rotation_angle
        self.base_mutation_prob = mutation_prob
        
        # Global best tracking for monotonic improvement
        self.global_best = None
        self.global_best_fitness = float('inf')
        
        # Stagnation detection and recovery
        self.stagnation_counter = 0
        self.stagnation_threshold = 30
        self.last_improvement_fitness = float('inf')

    def initialize_qbit(self):
        """
        Initialize quantum bit with maximum superposition state.
        
        Returns:
            Tuple of (alpha, beta) amplitude arrays representing equal probabilities
        """
        alpha = np.full(self.n_qbits, 1.0 / np.sqrt(2.0))
        beta  = np.full(self.n_qbits, 1.0 / np.sqrt(2.0))
        return alpha, beta
    
    def perturb_qbit(self, alpha, beta, strength=0.3):
        """
        Apply quantum perturbation to escape local optima during stagnation.
        
        Args:
            alpha, beta: Current quantum amplitudes
            strength: Perturbation intensity (0.0 to 1.0)
            
        Returns:
            Perturbed and normalized amplitude arrays
        """
        a = alpha.copy()
        b = beta.copy()
        
        # Apply random rotations to selected qubits
        for i in range(len(a)):
            if np.random.rand() < strength:
                # Random rotation with substantial angle for diversification
                theta = np.random.uniform(-np.pi/2, np.pi/2)
                c = np.cos(theta)
                s = np.sin(theta)
                na = c * a[i] - s * b[i]
                nb = s * a[i] + c * b[i]
                a[i], b[i] = na, nb
        
        # Normalize to maintain quantum state validity
        norm = np.sqrt(a**2 + b**2)
        a /= (norm + 1e-10)
        b /= (norm + 1e-10)
        
        return a, b

    def measure(self, alpha, beta):
        """
        Collapse quantum state to classical bit string (measurement operation).
        
        Args:
            alpha, beta: Quantum amplitude arrays
            
        Returns:
            Tuple of (assignment_bits, order_keys) defining the solution
        """
        # Calculate measurement probabilities from amplitudes
        probs = alpha**2
        
        # Assignment bits determine which vehicle serves each customer
        assign_probs = probs[:self.n]
        assignment_bits = (np.random.rand(self.n) < assign_probs).astype(int)
        
        # Order keys determine customer sequence within routes
        order_probs = probs[self.n:]
        order_keys = order_probs  # Direct probability use for consistent ordering
        
        return assignment_bits, order_keys

    def decode_from_measure(self, assignment_bits, order_keys):
        """
        Convert measured quantum state to concrete vehicle routes.
        
        Args:
            assignment_bits: Binary assignment of customers to vehicles
            order_keys: Continuous values determining customer ordering
            
        Returns:
            Tuple of (route1, route2) containing customer sequences
        """
        # Separate customers by vehicle assignment
        route1 = [i+1 for i in range(self.n) if assignment_bits[i] == 0]
        route2 = [i+1 for i in range(self.n) if assignment_bits[i] == 1]
        
        # Sort customers within each route by their order keys
        route1.sort(key=lambda c: order_keys[c-1])
        route2.sort(key=lambda c: order_keys[c-1])
        
        return route1, route2

    def fitness_tuple(self, r1, r2):
        """
        Evaluate multi-objective fitness for route pair.
        
        Args:
            r1, r2: Vehicle routes as customer index lists
            
        Returns:
            Tuple of (total_distance, workload_imbalance)
        """
        return self.vrp.objectives_from_routes(r1, r2)

    def amplitude_mutation(self, alpha, beta, mutation_prob):
        """
        Apply small random rotations to quantum amplitudes for local exploration.
        
        Args:
            alpha, beta: Current quantum amplitudes
            mutation_prob: Probability of mutating each qubit
            
        Returns:
            Mutated and normalized amplitude arrays
        """
        a = alpha.copy()
        b = beta.copy()
        
        # Apply small rotations to selected qubits
        for i in range(len(a)):
            if np.random.rand() < mutation_prob:
                # Small Gaussian perturbation for fine-tuning
                theta = np.random.normal(0, 0.1)
                c = np.cos(theta)
                s = np.sin(theta)
                na = c * a[i] - s * b[i]
                nb = s * a[i] + c * b[i]
                a[i], b[i] = na, nb
        
        # Maintain quantum state normalization
        norm = np.sqrt(a**2 + b**2)
        a /= (norm + 1e-10)
        b /= (norm + 1e-10)
        
        return a, b

    def crossover(self, p1, p2):
        """
        Perform quantum crossover by mixing amplitude information.
        
        Args:
            p1, p2: Parent quantum individuals as (alpha, beta) tuples
            
        Returns:
            Offspring quantum individual with combined genetic material
        """
        a1, b1 = p1
        a2, b2 = p2
        
        if np.random.rand() < self.crossover_rate:
            # Two-point crossover for quantum amplitudes
            i, j = sorted(np.random.choice(self.n_qbits, 2, replace=False))
            ca = a1.copy()
            cb = b1.copy()
            
            # Exchange amplitude segments between parents
            ca[i:j] = a2[i:j]
            cb[i:j] = b2[i:j]
            
            # Normalize resulting quantum state
            norm = np.sqrt(ca**2 + cb**2)
            ca /= (norm + 1e-10)
            cb /= (norm + 1e-10)
            
            return ca, cb
        
        return a1.copy(), b1.copy()

    def rotation_update(self, alpha, beta, best_bits, cur_bits, generation, is_stagnated=False):
        """
        Apply quantum rotation towards best known solution.
        
        This is the core quantum update mechanism that rotates amplitudes
        to increase probability of generating better solutions.
        
        Args:
            alpha, beta: Current quantum amplitudes
            best_bits: Bit representation of global best solution
            cur_bits: Bit representation of current solution
            generation: Current generation number
            is_stagnated: Whether population is stagnating
            
        Returns:
            Updated quantum amplitudes after rotation
        """
        best_assign, best_order = best_bits
        cur_assign, cur_order = cur_bits
        
        # Adaptive rotation schedule: exploration → exploitation
        progress = generation / self.generations
        # Start conservative (10% base), increase to aggressive (150% base)
        adaptive_angle = self.base_rotation_angle * (0.1 + 1.4 * progress)
        
        # Amplify rotation during stagnation to force exploration
        if is_stagnated:
            adaptive_angle *= 2.0
        
        a = alpha.copy()
        b = beta.copy()
        
        # Rotate assignment qubits towards optimal assignments
        for i in range(self.n):
            if best_assign[i] != cur_assign[i]:
                # Determine rotation direction based on target state
                if best_assign[i] == 1:
                    theta = adaptive_angle
                else:
                    theta = -adaptive_angle
                
                # Apply rotation to increase probability of desired state
                c = np.cos(theta)
                s = np.sin(theta)
                idx = i
                na = c * a[idx] - s * b[idx]
                nb = s * a[idx] + c * b[idx]
                a[idx], b[idx] = na, nb
        
        # Rotate ordering qubits towards optimal sequence
        for i in range(self.n):
            idx = self.n + i
            diff = best_order[i] - cur_order[i]
            if abs(diff) > 1e-6:
                # Rotation magnitude proportional to required change
                theta = adaptive_angle * np.sign(diff) * min(abs(diff), 1.0)
                c = np.cos(theta)
                s = np.sin(theta)
                na = c * a[idx] - s * b[idx]
                nb = s * a[idx] + c * b[idx]
                a[idx], b[idx] = na, nb
        
        # Normalize to maintain valid quantum state
        norm = np.sqrt(a**2 + b**2)
        a /= (norm + 1e-10)
        b /= (norm + 1e-10)
        
        return a, b

    def run(self):
        """
        Execute the complete Quantum-Inspired Evolutionary Algorithm.
        
        This method orchestrates the entire optimization process:
        - Population initialization with quantum superposition
        - Iterative evolution through measurement, selection, and update
        - Anti-stagnation mechanisms for maintaining diversity
        - Progress tracking and performance monitoring
        
        Yields:
            Progress snapshots containing generation data and best solutions
        """
        # Initialize quantum population in maximum superposition
        population = [self.initialize_qbit() for _ in range(self.population_size)]
        archive = []  # Repository of all evaluated solutions
        convergence = []  # Hypervolume tracking for convergence analysis
        t0 = time.time()
        
        # Global optimization tracking
        best_ever = None
        best_ever_fitness = float('inf')
        
        for g in range(self.generations + 1):
            # Stagnation assessment and intervention
            is_stagnated = self.stagnation_counter >= self.stagnation_threshold
            
            if is_stagnated:
                print(f"Gen {g}: Quantum stagnation detected! Applying quantum perturbation...")
                self.stagnation_counter = 0  # Reset stagnation monitoring
            
            # Adaptive mutation based on evolutionary pressure
            adaptive_mutation_prob = self.base_mutation_prob
            if self.stagnation_counter > self.stagnation_threshold // 2:
                # Progressively increase mutation as stagnation approaches
                adaptive_mutation_prob = min(0.3, self.base_mutation_prob * 
                                           (1 + self.stagnation_counter / self.stagnation_threshold))
            
            # Quantum measurement and classical evaluation
            measured_list = []
            objs = []
            
            for alpha, beta in population:
                # Collapse quantum state to classical solution
                assign_bits, order_keys = self.measure(alpha, beta)
                r1, r2 = self.decode_from_measure(assign_bits, order_keys)
                
                # Evaluate solution quality
                obj = self.fitness_tuple(r1, r2)
                measured_list.append((assign_bits, order_keys, r1, r2))
                objs.append(obj)
                
                # Archive solution for final analysis
                archive.append((None, None, obj, r1, r2))
            
            # Identify generation champion
            best_idx = min(range(len(objs)), key=lambda i: objs[i][0])
            gen_best = measured_list[best_idx]
            gen_best_fitness = objs[best_idx][0]
            
            # Update global best and monitor improvement
            if gen_best_fitness < best_ever_fitness:
                best_ever = gen_best
                best_ever_fitness = gen_best_fitness
                
                # Assess improvement significance for stagnation detection
                if gen_best_fitness < self.last_improvement_fitness * 0.999:
                    self.stagnation_counter = 0
                    self.last_improvement_fitness = gen_best_fitness
                else:
                    self.stagnation_counter += 1
            else:
                self.stagnation_counter += 1
            
            # Multi-objective ranking for selection pressure
            fronts, ranks = non_dominated_sort(objs)
            
            # Convergence assessment via hypervolume indicator
            if fronts:
                front_objs = [objs[i] for i in fronts[0]]
                ref = (max(o[0] for o in front_objs) * 1.1, 
                       max(o[1] for o in front_objs) * 1.1)
                hv = hypervolume_2d(front_objs, ref)
            else:
                hv = 0.0
            convergence.append(hv)
            
            # Progress reporting
            if g % PLOT_EVERY == 0:
                yield ('quantum', g, best_ever[2], best_ever[3], 
                       archive[:], convergence[:], time.time() - t0)
            
            # Evolutionary transition to next generation
            if g < self.generations:
                new_pop = []
                
                # Elite preservation strategy
                if ELITISM and best_idx < len(population):
                    new_pop.append(population[best_idx])
                
                # Stagnation recovery through quantum perturbation
                if is_stagnated:
                    num_perturbed = int(self.population_size * 0.3)  # 30% population refresh
                    for _ in range(num_perturbed):
                        # Apply strong quantum perturbation to random individual
                        idx = random.randint(0, len(population) - 1)
                        alpha, beta = population[idx]
                        perturbed_a, perturbed_b = self.perturb_qbit(alpha, beta, strength=0.5)
                        new_pop.append((perturbed_a, perturbed_b))
                
                # Offspring generation through quantum operations
                while len(new_pop) < self.population_size:
                    # Tournament selection based on Pareto dominance
                    idx1 = random.randint(0, self.population_size - 1)
                    idx2 = random.randint(0, self.population_size - 1)
                    p1_idx = idx1 if ranks[idx1] <= ranks[idx2] else idx2
                    
                    idx3 = random.randint(0, self.population_size - 1)
                    idx4 = random.randint(0, self.population_size - 1)
                    p2_idx = idx3 if ranks[idx3] <= ranks[idx4] else idx4
                    
                    p1 = population[p1_idx]
                    p2 = population[p2_idx]
                    
                    # Quantum crossover operation
                    child_a, child_b = self.crossover(p1, p2)
                    
                    # Amplitude mutation for local search
                    child_a, child_b = self.amplitude_mutation(child_a, child_b, adaptive_mutation_prob)
                    
                    # Quantum rotation towards global optimum
                    cur_assign, cur_order = self.measure(child_a, child_b)[:2]
                    child_a, child_b = self.rotation_update(
                        child_a, child_b,
                        (best_ever[0], best_ever[1]),
                        (cur_assign, cur_order),
                        g,
                        is_stagnated
                    )
                    
                    new_pop.append((child_a, child_b))
                
                population = new_pop
        
        return archive, convergence, time.time() - t0

# ================================================================================
# Visualization and Analysis Utilities
# ================================================================================
def plot_routes_subplot(vrp, ax, r1, r2, title):
    """
    Visualize VRP solution with routes and performance metrics.
    
    Creates a comprehensive plot showing depot, customers, vehicle routes,
    and key performance indicators in a single subplot.
    
    Args:
        vrp: VRP problem instance
        ax: Matplotlib subplot axes
        r1, r2: Vehicle routes as customer index lists
        title: Plot title string
    """
    # Plot depot as distinctive red square
    ax.scatter(vrp.depot[0], vrp.depot[1], marker='s', s=80, color='red', label='Depot')
    
    # Plot customers with numerical labels
    for i, coord in enumerate(vrp.customers, start=1):
        ax.scatter(coord[0], coord[1], color='black', s=40)
        ax.text(coord[0] + 0.8, coord[1] + 0.8, str(i), fontsize=9)

    def draw_route(route, color):
        """Draw vehicle route with depot connections."""
        if not route:
            return
        # Create path: depot → customers → depot
        pts = [vrp.depot] + [vrp.customers[c - 1] for c in route] + [vrp.depot]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker='o', color=color, linewidth=2, markersize=4)

    # Draw routes with distinct colors
    draw_route(r1, 'green')
    draw_route(r2, 'orange')

    # Calculate and display performance metrics
    total, imbalance = vrp.objectives_from_routes(r1, r2)
    ax.set_title(title, pad=18)
    ax.text(0.5, 1.00, f'Total Distance: {total:.2f}   Imbalance: {imbalance:.2f}',
            transform=ax.transAxes, ha='center', va='top', fontsize=9)
    
    # Configure plot appearance
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

def plot_combined_metrics(classic_vals, quant_vals, metric_names, title='Algorithm Performance Comparison'):
    """
    Generate comprehensive bar chart comparing algorithm performance metrics.
    
    Args:
        classic_vals: Performance values for classical algorithm
        quant_vals: Performance values for quantum-inspired algorithm
        metric_names: List of metric names for x-axis labels
        title: Chart title
    """
    N = len(metric_names)
    x = np.arange(N)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create grouped bar chart
    b1 = ax.bar(x - width/2, classic_vals, width, label='Classical EA', alpha=0.8)
    b2 = ax.bar(x + width/2, quant_vals, width, label='Quantum-Inspired EA', alpha=0.8)

    # Configure chart appearance
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=30, ha='right')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    def annotate_bars(bars, vals):
        """Add value labels on top of bars."""
        for rect, v in zip(bars, vals):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                txt = 'N/A'
                y = 0.0
            else:
                txt = f'{v:.4f}' if isinstance(v, float) else f'{v}'
                y = v
            ax.text(rect.get_x() + rect.get_width()/2, y + max(0.01, 0.01*abs(y)), 
                   txt, ha='center', va='bottom', fontsize=10, fontweight='bold')

    annotate_bars(b1, classic_vals)
    annotate_bars(b2, quant_vals)
    plt.tight_layout()
    plt.show()

# ================================================================================
# Main Experimental Framework
# ================================================================================
def run_both_and_compare(vrp):
    """
    Execute comprehensive comparison between Classical and Quantum-Inspired EAs.
    
    This function orchestrates the complete experimental protocol:
    1. Parallel execution of both algorithms
    2. Real-time visualization of progress
    3. Comprehensive performance analysis
    4. Statistical comparison and visualization
    
    Args:
        vrp: VRP problem instance
        
    Returns:
        Dictionary containing detailed results from both algorithms
    """
    # Initialize algorithm instances
    classical_gen = classical_ea(vrp, pop_size=POP_SIZE, gens=GENERATIONS)
    qea = QuantumInspiredEA(vrp, pop_size=POP_SIZE, generations=GENERATIONS)
    quantum_gen = qea.run()

    # Initialize result tracking
    all_class_archive = []
    all_quant_archive = []
    c_conv = []
    q_conv = []
    c_time = 0.0
    q_time = 0.0

    # Execute algorithms with synchronized progress reporting
    n_steps = (GENERATIONS // PLOT_EVERY) + 1
    for _ in range(n_steps):
        # Retrieve synchronized progress from both algorithms
        c_res = next(classical_gen)
        q_res = next(quantum_gen)
        
        # Verify generation synchronization
        assert c_res[1] == q_res[1]
        gen = c_res[1]
        
        # Extract results from both algorithms
        _, _, c_r1, c_r2, c_archive, c_conv_local, c_time = c_res
        _, _, q_r1, q_r2, q_archive, q_conv_local, q_time = q_res

        # Update global tracking
        all_class_archive = c_archive
        all_quant_archive = q_archive
        c_conv = c_conv_local
        q_conv = q_conv_local

        # Generate comparative visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_routes_subplot(vrp, axes[0], c_r1, c_r2, f'Classical EA — Generation {gen}')
        plot_routes_subplot(vrp, axes[1], q_r1, q_r2, f'Quantum-Inspired EA — Generation {gen}')
        plt.suptitle(f'Algorithm Comparison at Generation {gen}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()

    # ============================================================================
    # Comprehensive Post-Processing and Analysis
    # ============================================================================
    
    # Prepare solution archives for analysis
    class_sols = all_class_archive
    quant_sols = []
    
    # Standardize quantum solution format for analysis
    for entry in all_quant_archive:
        seq, split, obj, r1, r2 = entry
        if seq is None:
            # Reconstruct sequence representation for quantum solutions
            seq_placeholder = r1 + r2
            split_placeholder = len(r1)
            quant_sols.append((seq_placeholder, split_placeholder, obj, r1, r2))
        else:
            quant_sols.append(entry)

    # Extract objective vectors for multi-objective analysis
    class_objs = [s[2] for s in class_sols]
    quant_objs = [s[2] for s in quant_sols]

    # Perform Pareto front analysis
    class_fronts, _ = non_dominated_sort(class_objs) if class_objs else ([], [])
    quant_fronts, _ = non_dominated_sort(quant_objs) if quant_objs else ([], [])

    class_front = [class_sols[i] for i in class_fronts[0]] if class_fronts else []
    quant_front = [quant_sols[i] for i in quant_fronts[0]] if quant_fronts else []

    def get_best_solution(sols):
        """Extract solution with minimum total distance."""
        if not sols:
            return None
        return min(sols, key=lambda x: x[2][0])

    class_best = get_best_solution(class_sols)
    quant_best = get_best_solution(quant_sols)

    # Calculate performance metrics
    combined = class_objs + quant_objs
    if combined:
        ref = (max(o[0] for o in combined) * 1.05, max(o[1] for o in combined) * 1.05)
    else:
        ref = (1.0, 1.0)

    class_hv = hypervolume_2d([s[2] for s in class_front], ref) if class_front else 0.0
    quant_hv = hypervolume_2d([s[2] for s in quant_front], ref) if quant_front else 0.0

    class_spacing = spacing_metric([s[2] for s in class_front]) if class_front else 0.0
    quant_spacing = spacing_metric([s[2] for s in quant_front]) if quant_front else 0.0

    # Prepare convergence data for comparison
    L = min(len(c_conv), len(q_conv)) if c_conv and q_conv else 0
    c_conv_trim = c_conv[:L]
    q_conv_trim = q_conv[:L]

    # ============================================================================
    # Results Summary and Reporting
    # ============================================================================
    print('\n' + '='*60)
    print('COMPREHENSIVE ALGORITHM COMPARISON SUMMARY')
    print('='*60)
    
    if class_best:
        print(f'Classical EA Best Solution:')
        print(f'  • Total Distance: {class_best[2][0]:.4f}')
        print(f'  • Route Imbalance: {class_best[2][1]:.4f}')
    
    if quant_best:
        print(f'Quantum-Inspired EA Best Solution:')
        print(f'  • Total Distance: {quant_best[2][0]:.4f}')
        print(f'  • Route Imbalance: {quant_best[2][1]:.4f}')
    
    print(f'Multi-Objective Performance:')
    print(f'  • Classical Hypervolume: {class_hv:.6f}')
    print(f'  • Quantum Hypervolume: {quant_hv:.6f}')
    print(f'  • Classical Front Spacing: {class_spacing:.6f}')
    print(f'  • Quantum Front Spacing: {quant_spacing:.6f}')
    
    print(f'Computational Efficiency:')
    print(f'  • Classical Runtime: {c_time:.3f} seconds')
    print(f'  • Quantum Runtime: {q_time:.3f} seconds')
    print('='*60)

    # ============================================================================
    # Advanced Visualization Suite
    # ============================================================================
    
    # Pareto Front Comparison Plot
    plt.figure(figsize=(10,7))
    if class_front:
        co = np.array([s[2] for s in class_front])
        plt.scatter(co[:,0], co[:,1], c='red', s=60, alpha=0.7, 
                   label=f'Classical Pareto Front ({len(class_front)} solutions)')
    if quant_front:
        qo = np.array([s[2] for s in quant_front])
        plt.scatter(qo[:,0], qo[:,1], c='blue', s=60, alpha=0.7,
                   label=f'Quantum Pareto Front ({len(quant_front)} solutions)')
    
    plt.xlabel('Total Distance', fontsize=12)
    plt.ylabel('Route Imbalance', fontsize=12)
    plt.title('Pareto Front Comparison: Multi-Objective Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Convergence Analysis Plot
    if L > 0:
        gens = np.arange(L) * PLOT_EVERY
        plt.figure(figsize=(12,6))
        plt.plot(gens, c_conv_trim, label='Classical EA', linewidth=2, marker='o', markersize=4)
        plt.plot(gens, q_conv_trim, label='Quantum-Inspired EA', linewidth=2, marker='s', markersize=4)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Hypervolume Indicator', fontsize=12)
        plt.title('Algorithm Convergence Analysis', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Comprehensive Performance Metrics Comparison
    metric_names = ['Best Total Distance', 'Best Imbalance', 'Hypervolume', 'Spacing', 'Runtime (s)']
    class_vals = [
        float(class_best[2][0]) if class_best else np.nan,
        float(class_best[2][1]) if class_best else np.nan,
        float(class_hv),
        float(class_spacing),
        float(c_time)
    ]
    quant_vals = [
        float(quant_best[2][0]) if quant_best else np.nan,
        float(quant_best[2][1]) if quant_best else np.nan,
        float(quant_hv),
        float(quant_spacing),
        float(q_time)
    ]

    plot_combined_metrics(class_vals, quant_vals, metric_names, 
                         title='Comprehensive Algorithm Performance Comparison')

    # Return structured results for further analysis
    return {
        'classical': {
            'archive': class_sols, 
            'pareto_front': class_front, 
            'hypervolume': class_hv, 
            'spacing': class_spacing, 
            'runtime': c_time
        },
        'quantum': {
            'archive': quant_sols, 
            'pareto_front': quant_front, 
            'hypervolume': quant_hv, 
            'spacing': quant_spacing, 
            'runtime': q_time
        }
    }

# ================================================================================
# Main Execution Entry Point
# ================================================================================
if __name__ == '__main__':
    print(f'Initiating Multi-Objective VRP Optimization Comparison')
    print(f'Configuration: {N_CUSTOMERS} customers, {POP_SIZE} population, {GENERATIONS} generations')
    print(f'Progress visualization frequency: every {PLOT_EVERY} generations')
    print('='*70)
    
    results = run_both_and_compare(vrp)
    
    print('\nExperimental comparison completed successfully.')
    print('Detailed results and visualizations have been generated.')