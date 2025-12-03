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
        """Create sugar distribution with two peaks."""
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
        """Initialize agents at random positions."""
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
        """Get sugar level at a cell."""
        return self.sugar_grid[y, x]
    
    def get_visible_cells(self, agent: Agent) -> List[Tuple[int, int, float]]:
        """
        Get all cells visible to an agent (in cardinal directions).
        
        Returns:
            List of tuples (x, y, sugar_level)
        """
        visible = []
        
        # Look in four cardinal directions
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
        """Move agent to the visible cell with most sugar."""
        visible_cells = self.get_visible_cells(agent)
        
        if not visible_cells:
            return  # Agent can't move
        
        # Find cell(s) with maximum sugar
        max_sugar = max(cell[2] for cell in visible_cells)
        best_cells = [cell for cell in visible_cells if cell[2] == max_sugar]
        
        # If multiple cells have same sugar, choose closest
        if len(best_cells) > 1:
            distances = [abs(cell[0] - agent.x) + abs(cell[1] - agent.y) 
                        for cell in best_cells]
            min_dist = min(distances)
            best_cells = [best_cells[i] for i, d in enumerate(distances) 
                         if d == min_dist]
        
        # Move to chosen cell (random if still tied)
        chosen_cell = random.choice(best_cells)
        agent.x, agent.y = chosen_cell[0], chosen_cell[1]
    
    def agent_eat(self, agent: Agent):
        """Agent eats sugar at current location."""
        sugar_here = self.sugar_grid[agent.y, agent.x]
        agent.sugar += sugar_here
        self.sugar_grid[agent.y, agent.x] = 0  # Sugar consumed
        
        # Metabolize
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
        """Execute one time step of the simulation."""
        self.time_step += 1
        
        # Randomize agent order each step
        active_agents = [a for a in self.agents if a.alive]
        random.shuffle(active_agents)
        
        new_agents = []
        
        for agent in active_agents:
            if not agent.alive:
                continue
            
            # Agent acts
            self.move_agent(agent)
            self.agent_eat(agent)
            agent.age += 1
            
            # Check for death
            if agent.sugar <= 0 or agent.age >= agent.max_age:
                agent.alive = False
                self.total_agents_died += 1
            
            # Check for reproduction
            if agent.alive:
                child = self.agent_reproduce(agent)
                if child:
                    new_agents.append(child)
        
        # Add new agents
        self.agents.extend(new_agents)
        
        # Grow back sugar
        self.grow_sugar()
        
        # Clean up dead agents (keep for statistics but mark as dead)
        # self.agents = [a for a in self.agents if a.alive]
    
    def grow_sugar(self):
        """Grow sugar back towards maximum capacity."""
        for i in range(self.height):
            for j in range(self.width):
                if self.sugar_grid[i, j] < self.max_sugar[i, j]:
                    self.sugar_grid[i, j] = min(
                        self.sugar_grid[i, j] + self.growback_rate,
                        self.max_sugar[i, j]
                    )
    
    def get_statistics(self) -> dict:
        """Get current simulation statistics."""
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
        """Get data for all alive agents."""
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
