# Quantum-Inspired Evolutionary Algorithm for Vehicle Routing Problem

## Overview

This repository contains the implementation of a Quantum-Inspired Evolutionary Algorithm (QEA) to solve the Vehicle Routing Problem (VRP), along with a baseline Classical Evolutionary Algorithm for comparison. The project demonstrates how quantum-inspired mechanisms can enhance traditional evolutionary search algorithms in solving combinatorial optimization problems in the logistics domain.

## Problem Definition

### Vehicle Routing Problem (VRP) Scenario
- **Depot**: 1 central depot at coordinates (50, 50)
- **Vehicles**: 2 vehicles (represented as 2 routes)
- **Customers**: 10 customer locations randomly distributed in a 100x100 grid
- **Objective**: Multi-objective optimization to:
  1. Minimize total distance traveled by both vehicles
  2. Minimize workload imbalance between the two routes

### Problem Constraints
- Each customer must be visited exactly once
- Both vehicles start and end at the depot
- Each route must have at least one customer

### Example Dataset
The implementation uses a reproducible random seed (42) to generate consistent customer locations for fair algorithm comparison. Coordinates are generated uniformly within a 100x100 grid, and distances are calculated using Euclidean distance.

## Algorithm Implementations

### 1. Classical Evolutionary Algorithm (Baseline)

A multi-objective evolutionary algorithm implementing NSGA-II principles:

**Representation**: 
- Permutation of all customers + split point indicating where to divide customers between two vehicles

**Key Operations**:
- **Selection**: Tournament selection with Pareto ranking and crowding distance
- **Crossover**: Order crossover (OX) preserving customer sequence validity
- **Mutation**: Swap mutation for customer sequences, perturbation for split points
- **Environmental Selection**: NSGA-II survivor selection maintaining Pareto front diversity

**Advanced Features**:
- **Elitism**: Global best solution preservation
- **Anti-stagnation**: Detection of convergence stagnation with adaptive responses
- **Adaptive mutation**: Increased mutation rates when approaching stagnation
- **Population injection**: Fresh random solutions during stagnation periods

### 2. Quantum-Inspired Evolutionary Algorithm (QEA)

A novel approach using quantum computing principles adapted for classical hardware:

**Quantum Representation**: 
- Q-bits with amplitude pairs (α, β) encoding solution space
- 2n q-bits: n for customer-to-vehicle assignment, n for ordering within routes

**Quantum Operators**:
- **Measurement**: Probabilistic collapse of quantum superposition to classical solutions
- **Rotation Gates**: Adaptive rotation toward best solutions with angle scheduling
- **Quantum Crossover**: Amplitude-based crossover preserving quantum properties  
- **Quantum Mutation**: Small rotation perturbations maintaining superposition

**Quantum-Inspired Features**:
- **Adaptive Rotation**: Exploration-focused early (small angles) → exploitation-focused late (large angles)
- **Quantum Perturbation**: Strong amplitude perturbation for stagnation escape
- **Superposition Encoding**: Probabilistic solution representation enabling multiple possibilities
- **Quantum Elitism**: Preservation of best quantum states across generations

**Stagnation Recovery**:
- **Detection**: Monitoring fitness improvement over generations
- **Response**: Quantum perturbation of 30% population with increased rotation angles

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.7+
- NumPy (≥1.19.0) - for numerical computations and quantum amplitude operations
- Matplotlib (≥3.3.0) - for route visualization and performance analysis

## Usage

### Running the Comparison
```bash
python CA_and_QInspired_EA.py
```

### Configuration Parameters
Modify these constants at the top of the file:
```python
POP_SIZE = 25           # Population size (appropriate for 10-customer problem)
GENERATIONS = 500       # Number of generations  
N_CUSTOMERS = 10        # Number of customer locations
CROSSOVER_RATE = 0.9    # Crossover probability
MUTATION_RATE = 0.2     # Base mutation probability
PLOT_EVERY = 25         # Visualization frequency (every N generations)
```

### Expected Runtime
- Approximate execution time: 2-3 minutes
- Progress visualizations displayed every 25 generations
- Final analysis plots generated at completion

## Output and Visualization

### Real-time Progress
1. **Route Visualizations**: Side-by-side comparison every 25 generations
   - Green routes: Vehicle 1 paths
   - Orange routes: Vehicle 2 paths  
   - Red square: Depot location
   - Numbered points: Customer locations

### Final Analysis
2. **Performance Metrics Dashboard**:
   - Best total distance found by each algorithm
   - Workload imbalance between vehicles
   - Hypervolume indicator (Pareto front quality)
   - Spacing metric (solution diversity)
   - Computational runtime comparison

3. **Pareto Front Visualization**: Scatter plot comparing non-dominated solutions
4. **Convergence Analysis**: Hypervolume progression over generations
5. **Performance Comparison Chart**: Grouped bar chart of all metrics

## Algorithm Design Rationale

### Classical EA Design Choices

**Multi-objective Approach**: NSGA-II framework chosen for its proven effectiveness in handling trade-offs between total distance and workload balance.

**Permutation + Split Representation**: Natural encoding for VRP ensuring all customers visited exactly once while allowing flexible route divisions.

**Anti-stagnation Mechanisms**: Essential for small population sizes (25 individuals) to prevent premature convergence:
- Stagnation detection after 15 generations without meaningful improvement
- Adaptive mutation rate increases (up to 4x base rate)
- Population diversification through random immigrant injection

### Quantum-Inspired Design Choices

**Q-bit Amplitude Encoding**: Enables natural representation of solution uncertainty and multiple possibilities in superposition.

**Adaptive Rotation Strategy**: 
- Early generations: Small rotation angles (0.1× base) for broad exploration
- Late generations: Large rotation angles (1.5× base) for fine-tuned exploitation
- Stagnation response: Extra large angles (2× adaptive) for escape

**Measurement Without Noise**: Direct use of amplitude probabilities ensures deterministic decoding while maintaining quantum properties.

**Quantum Perturbation**: Strong amplitude perturbation (50% strength) applied to 30% of population during stagnation, providing quantum-inspired diversification mechanism.

## Performance Evaluation Methodology

### Metrics Selection
1. **Solution Quality**: Best total distance (primary logistics objective)
2. **Balance Quality**: Workload imbalance (fairness objective)  
3. **Pareto Efficiency**: Hypervolume indicator for multi-objective optimization quality
4. **Solution Diversity**: Spacing metric ensuring well-distributed Pareto fronts
5. **Computational Efficiency**: Runtime comparison for practical applicability

### Statistical Validity
- Reproducible random seed ensures fair comparison
- Identical problem instances for both algorithms
- Same computational budget (population size × generations)

### Expected Performance Characteristics

**Classical EA Strengths**:
- Proven convergence properties
- Effective local exploitation through directed mutation
- Strong performance on established VRP benchmarks

**Quantum-Inspired EA Advantages**:
- Enhanced exploration through quantum superposition
- Novel escape mechanisms via quantum perturbation
- Adaptive balance between exploration and exploitation
- Potential for discovering unique solution regions

## File Structure
```
QuantumInspiredEA/
├── README.md                    # Comprehensive documentation
├── requirements.txt             # Python dependencies
├── QUICKSTART.md               # Quick setup guide
├── .gitignore                  # Git exclusions
└── CA_and_QInspired_EA.py      # Main implementation with clean, descriptive comments
```

## Technical Implementation Details

### Quantum-Inspired Mechanisms

**Q-bit Initialization**: Equal superposition state (α = β = 1/√2) representing maximum uncertainty.

**Quantum Measurement Process**:
1. Calculate measurement probabilities from α² amplitudes
2. Assignment bits: Binary decisions for customer-to-vehicle mapping
3. Ordering keys: Continuous values determining visit sequence within routes

**Rotation Gate Implementation**:
```
α' = cos(θ) × α - sin(θ) × β
β' = sin(θ) × α + cos(θ) × β
```
With adaptive angle θ based on generation progress and stagnation status.

**Amplitude Normalization**: Maintains quantum property constraint α² + β² = 1 after all operations.

### Anti-Stagnation Architecture

Both algorithms implement sophisticated stagnation detection:
- **Improvement Threshold**: 0.1% fitness improvement required to reset stagnation counter
- **Graduated Response**: Increasing intervention strength as stagnation approaches
- **Recovery Mechanisms**: Population diversification, adaptive parameters, quantum perturbation

## Scientific Contribution

This implementation provides a controlled experimental environment for comparing:
- Traditional evolutionary approaches vs. quantum-inspired methods
- Different stagnation recovery mechanisms
- Multi-objective optimization techniques in logistics
- Scalability of quantum-inspired algorithms on classical hardware

The quantum-inspired features demonstrate practical application of quantum computing concepts:
- **Superposition**: Multiple solution possibilities encoded simultaneously
- **Measurement**: Probabilistic extraction of classical solutions

## Academic Context

This project bridges quantum computing theory and practical logistics optimization, providing insights into:
- Quantum-inspired algorithm design for combinatorial problems
- Performance trade-offs between exploration and exploitation
- Stagnation recovery in small population evolutionary algorithms
- Multi-objective optimization in constrained logistics scenarios

## License

This project is provided for educational and research purposes.

## Author

Developed as part of a quantum-inspired optimization research initiative, demonstrating the practical application of quantum computing principles to classical logistics problems.
