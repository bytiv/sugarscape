"""
Sugarscape Agent-Based Model
An implementation of the Sugarscape model by Epstein & Axtell
for studying collective and social intelligence.
"""

import numpy as np
import random
from typing import List, Tuple, Optional


class Agent:
    """
    Agent class representing an individual in the Sugarscape environment.
    
    Attributes:
        id: Unique identifier for the agent
        x, y: Position on the grid
        sugar: Current sugar wealth
        vision: How far the agent can see
        metabolism: Sugar consumption per time step
        max_age: Maximum lifespan
        age: Current age
        alive: Whether the agent is alive
    """
    
    agent_counter = 0
    
    def __init__(self, x: int, y: int, sugar: int, vision: int, 
                 metabolism: int, max_age: int):
        # Assign a unique id to each agent (static counter)
        self.id = Agent.agent_counter
        Agent.agent_counter += 1
        self.x = x
        self.y = y
        self.sugar = sugar
        self.vision = vision
        self.metabolism = metabolism
        self.max_age = max_age
        self.age = 0
        self.alive = True
    
    def __repr__(self):
        return f"Agent({self.id}, pos=({self.x},{self.y}), sugar={self.sugar})"


class Sugarscape:
    """
    Sugarscape environment with agents and resources.
    
    The environment is a 2D grid where each cell contains sugar that grows over time.
    Agents move around collecting sugar to survive.
    """
    
    def __init__(self, width: int = 50, height: int = 50, 
                 initial_agents: int = 100, reproduction: bool = False):
        """
        Initialize the Sugarscape environment.
        
        Args:
            width: Grid width
            height: Grid height
            initial_agents: Number of initial agents
            reproduction: Whether agents can reproduce
        """
        self.width = width
        self.height = height
        self.reproduction_enabled = reproduction
        
        # Initialize sugar grid with two peaks (like original Sugarscape)
        self.sugar_grid = self._initialize_sugar_grid()
        self.max_sugar = np.copy(self.sugar_grid)  # Maximum capacity for each cell
        self.growback_rate = 1  # Sugar regrowth per time step
        
        # Initialize agents
        self.agents: List[Agent] = []
        self._initialize_agents(initial_agents)
        
        # Track statistics
        self.time_step = 0
        self.total_agents_born = initial_agents
        self.total_agents_died = 0
    
    def _initialize_sugar_grid(self) -> np.ndarray:
        """
        Create sugar distribution with two resource "peaks".

        The original Sugarscape model places richer sugar quantities in
        specific regions of the grid. This helper computes a simple
        radial falloff from two peak locations so cells near a peak
        have more sugar (resource) than distant cells.
        """
        grid = np.zeros((self.height, self.width))
        
        # Create two sugar peaks (top-left and bottom-right quadrants)
        peak1_x, peak1_y = self.width // 4, self.height // 4
        peak2_x, peak2_y = 3 * self.width // 4, 3 * self.height // 4
        
        for i in range(self.height):
            for j in range(self.width):
                # Distance from peaks
                dist1 = np.sqrt((i - peak1_y)**2 + (j - peak1_x)**2)
                dist2 = np.sqrt((i - peak2_y)**2 + (j - peak2_x)**2)
                
                # Sugar level based on distance from nearest peak
                sugar1 = max(0, 4 - dist1 / 5)
                sugar2 = max(0, 4 - dist2 / 5)
                grid[i, j] = max(sugar1, sugar2)
        
        return grid
    
    def _initialize_agents(self, num_agents: int):
        """
        Create `num_agents` Agent instances at random grid locations.

        Each agent is given randomized trait values (initial sugar,
        vision, metabolism, max_age) sampled from reasonable ranges.
        Agents are appended to `self.agents` and kept in the environment
        for the duration of the simulation (alive flag is used to mark
        deaths while preserving history).
        """
        for _ in range(num_agents):
            # Random position
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            
            # Random attributes
            initial_sugar = random.randint(5, 25)
            vision = random.randint(1, 6)
            metabolism = random.randint(1, 4)
            max_age = random.randint(60, 100)
            
            agent = Agent(x, y, initial_sugar, vision, metabolism, max_age)
            self.agents.append(agent)
    
    def get_cell_sugar(self, x: int, y: int) -> float:
        """Return the sugar amount available at coordinates `(x, y)`.

        Note: grid indexing uses `[y, x]` because the first dimension is
        the row (height) and the second is the column (width).
        """
        return self.sugar_grid[y, x]
    
    def get_visible_cells(self, agent: Agent) -> List[Tuple[int, int, float]]:
        """
        Get all cells visible to an agent (in cardinal directions).
        
        Returns:
            List of tuples (x, y, sugar_level)
        """
        visible = []

        # Look in four cardinal directions (N, S, E, W). Agents see up to
        # `agent.vision` cells away. Grid wraps around using modulo so the
        # environment is toroidal.
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for distance in range(1, agent.vision + 1):
                new_x = (agent.x + dx * distance) % self.width
                new_y = (agent.y + dy * distance) % self.height
                
                # Check if cell is occupied
                occupied = any(a.alive and a.x == new_x and a.y == new_y 
                             for a in self.agents)
                if not occupied:
                    sugar = self.get_cell_sugar(new_x, new_y)
                    visible.append((new_x, new_y, sugar))
        
        return visible
    
    def move_agent(self, agent: Agent):
        """
        Move the agent to the best visible unoccupied cell.

        Rules implemented:
        - Agent inspects visible cells (cardinal directions) and prefers the
          cell with the largest sugar value.
        - If multiple cells have equal sugar, prefer the closest one (Manhattan
          distance). If still tied, break ties randomly.
        - If no unoccupied cell is visible, the agent stays in place.
        """
        visible_cells = self.get_visible_cells(agent)

        if not visible_cells:
            return  # No available move (all visible cells occupied)

        # Select cell(s) with the maximum sugar value
        max_sugar = max(cell[2] for cell in visible_cells)
        best_cells = [cell for cell in visible_cells if cell[2] == max_sugar]

        # If there are multiple best cells, pick the one(s) with smallest distance
        if len(best_cells) > 1:
            distances = [abs(cell[0] - agent.x) + abs(cell[1] - agent.y)
                         for cell in best_cells]
            min_dist = min(distances)
            best_cells = [best_cells[i] for i, d in enumerate(distances) if d == min_dist]

        # Final tie-breaker: random choice
        chosen_cell = random.choice(best_cells)
        agent.x, agent.y = chosen_cell[0], chosen_cell[1]
    
    def agent_eat(self, agent: Agent):
                """
                Agent consumes all sugar at its current cell and then metabolizes.

                - Add the sugar at the current location to the agent's sugar store.
                - Clear the sugar at that cell (it's consumed).
                - Subtract the agent's metabolism (sugar cost per step). If sugar
                    falls to zero or below the agent will die during the step check.
                """
                sugar_here = self.sugar_grid[agent.y, agent.x]
                agent.sugar += sugar_here
                self.sugar_grid[agent.y, agent.x] = 0  # Sugar consumed

                # Agent consumes sugar to maintain itself (metabolism)
                agent.sugar -= agent.metabolism
    
    def agent_reproduce(self, agent: Agent) -> Optional[Agent]:
        """
        Agent reproduces if conditions are met.
        
        Conditions:
        - Agent must have enough sugar (>= 50)
        - Agent must be of reproductive age (age > 10 and age < max_age - 10)
        
        Returns:
            New agent if reproduction occurs, None otherwise
        """
        # If reproduction is globally disabled, skip
        if not self.reproduction_enabled:
            return None
        
        if (agent.sugar >= 50 and 
            agent.age > 10 and 
            agent.age < agent.max_age - 10 and
            random.random() < 0.05):  # 5% chance per step
            
            # Find empty adjacent cell
            adjacent = [
                ((agent.x + dx) % self.width, (agent.y + dy) % self.height)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            ]
            
            empty_cells = [
                (x, y) for x, y in adjacent
                if not any(a.alive and a.x == x and a.y == y for a in self.agents)
            ]
            
            if empty_cells:
                # Create offspring
                x, y = random.choice(empty_cells)
                
                # Inherit traits with some variation
                child_vision = agent.vision + random.randint(-1, 1)
                child_vision = max(1, min(6, child_vision))
                
                child_metabolism = agent.metabolism + random.randint(-1, 1)
                child_metabolism = max(1, min(4, child_metabolism))
                
                child_max_age = agent.max_age + random.randint(-10, 10)
                child_max_age = max(40, min(120, child_max_age))
                
                # Parent gives half their sugar to child
                child_sugar = agent.sugar // 2
                agent.sugar = agent.sugar // 2
                
                child = Agent(x, y, child_sugar, child_vision, 
                            child_metabolism, child_max_age)
                self.total_agents_born += 1
                return child
        
        return None
    
    def step(self):
        """
        Advance the environment by one discrete time step.

        Per-step sequence for each alive agent (in randomized order):
        1. Move to a preferred cell.
        2. Eat sugar at the new location.
        3. Age one time step and check for death (starvation or old age).
        4. Potentially reproduce if conditions are met.

        After agents act, new offspring are appended, sugar regrows across
        the grid, and death counters/statistics are updated.
        """
        self.time_step += 1

        # Randomize agent order to avoid update-order bias
        active_agents = [a for a in self.agents if a.alive]
        random.shuffle(active_agents)

        new_agents = []

        for agent in active_agents:
            if not agent.alive:
                continue

            # Agent decides where to move and then consumes sugar
            self.move_agent(agent)
            self.agent_eat(agent)
            agent.age += 1

            # Death conditions: no sugar left or exceeded lifespan
            if agent.sugar <= 0 or agent.age >= agent.max_age:
                agent.alive = False
                self.total_agents_died += 1

            # Reproduction (if still alive after metabolism)
            if agent.alive:
                child = self.agent_reproduce(agent)
                if child:
                    new_agents.append(child)

        # Add any newborn agents to the population list
        self.agents.extend(new_agents)

        # Regrow sugar across the landscape toward each cell's capacity
        self.grow_sugar()
    
    def grow_sugar(self):
        """
        Incrementally regrow sugar at each cell until it reaches the cell's
        maximum capacity (`self.max_sugar`). This simulates resource renewal
        in the environment.
        """
        for i in range(self.height):
            for j in range(self.width):
                if self.sugar_grid[i, j] < self.max_sugar[i, j]:
                    self.sugar_grid[i, j] = min(
                        self.sugar_grid[i, j] + self.growback_rate,
                        self.max_sugar[i, j]
                    )
    
    def get_statistics(self) -> dict:
        """
        Compute and return a small dictionary of population-level statistics
        useful for monitoring and saving simulation progress.

        Statistics include current `time_step`, living `population`, average
        sugar (wealth) among living agents, average age, and cumulative
        totals for births and deaths.
        """
        alive_agents = [a for a in self.agents if a.alive]

        if not alive_agents:
            return {
                'time_step': self.time_step,
                'population': 0,
                'avg_sugar': 0,
                'avg_age': 0,
                'total_born': self.total_agents_born,
                'total_died': self.total_agents_died,
            }

        return {
            'time_step': self.time_step,
            'population': len(alive_agents),
            'avg_sugar': np.mean([a.sugar for a in alive_agents]),
            'avg_age': np.mean([a.age for a in alive_agents]),
            'total_born': self.total_agents_born,
            'total_died': self.total_agents_died,
        }
    
    def get_agent_data(self) -> List[dict]:
        """
        Return a list of dictionaries representing each currently alive
        agent. This is suitable for writing to CSVs for later analysis and
        visualization. Each dictionary contains spatial location, traits,
        and current state values.
        """
        alive_agents = [a for a in self.agents if a.alive]
        return [
            {
                'agent_id': a.id,
                'x': a.x,
                'y': a.y,
                'sugar': a.sugar,
                'vision': a.vision,
                'metabolism': a.metabolism,
                'age': a.age,
                'max_age': a.max_age,
            }
            for a in alive_agents
        ]
