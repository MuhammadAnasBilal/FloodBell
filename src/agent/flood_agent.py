"""
Step 7: Intelligent Agent -- A* Search for Flood Evacuation Planning
====================================================================
PEAS Framework:
  Performance: Minimize casualties, minimize evacuation time, maximize resource coverage
  Environment: Pakistan regions with flood risk, shelters, roads (partially observable, stochastic, sequential)
  Actuators:   Issue alerts, open shelters, deploy teams, begin evacuation, allocate supplies
  Sensors:     ML predictions (FloodProbability), feature values, population data

Search Algorithm: A* with heuristic based on remaining unmet needs
"""

import heapq
import copy
import math


class FloodState:
    """Represents a state in the flood disaster response search space."""
    def __init__(self, risk_level="medium", population=100000, evacuated=0,
                 shelters_open=0, shelters_needed=2, teams_deployed=0,
                 teams_needed=3, supplies_allocated=False, alert_level="NONE",
                 medical_teams=0, medical_needed=1, routes_cleared=False):
        self.risk_level = risk_level
        self.population = population
        self.evacuated = evacuated
        self.shelters_open = shelters_open
        self.shelters_needed = shelters_needed
        self.teams_deployed = teams_deployed
        self.teams_needed = teams_needed
        self.supplies_allocated = supplies_allocated
        self.alert_level = alert_level
        self.medical_teams = medical_teams
        self.medical_needed = medical_needed
        self.routes_cleared = routes_cleared

    def is_goal(self):
        return (self.evacuated >= self.population and
                self.shelters_open >= self.shelters_needed and
                self.teams_deployed >= self.teams_needed and
                self.supplies_allocated and
                self.medical_teams >= self.medical_needed)

    def to_tuple(self):
        return (self.evacuated, self.shelters_open, self.teams_deployed,
                self.supplies_allocated, self.alert_level, self.medical_teams,
                self.routes_cleared)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def __lt__(self, other):
        return False


def _set(state, **kwargs):
    new_state = copy.deepcopy(state)
    for k, v in kwargs.items():
        setattr(new_state, k, v)
    return new_state


ACTIONS = {
    "ISSUE_ALERT": {
        "description": "Issue flood alert to the population",
        "cost": 1,
        "preconditions": lambda s: s.alert_level == "NONE",
        "effect": lambda s: _set(s, alert_level=("YELLOW" if s.risk_level == "low" else
                                                   "ORANGE" if s.risk_level in ["medium", "high"] else "RED"))
    },
    "UPGRADE_ALERT": {
        "description": "Upgrade alert level to RED",
        "cost": 1,
        "preconditions": lambda s: s.alert_level in ["YELLOW", "ORANGE"] and s.risk_level in ["high", "critical"],
        "effect": lambda s: _set(s, alert_level="RED")
    },
    "OPEN_SHELTER": {
        "description": "Open an emergency shelter for evacuees",
        "cost": 3,
        "preconditions": lambda s: s.shelters_open < s.shelters_needed and s.alert_level != "NONE",
        "effect": lambda s: _set(s, shelters_open=s.shelters_open + 1)
    },
    "DEPLOY_RESCUE_TEAM": {
        "description": "Deploy a rescue team to the affected area",
        "cost": 4,
        "preconditions": lambda s: s.teams_deployed < s.teams_needed and s.alert_level != "NONE",
        "effect": lambda s: _set(s, teams_deployed=s.teams_deployed + 1)
    },
    "DEPLOY_MEDICAL_TEAM": {
        "description": "Deploy a medical team to provide healthcare",
        "cost": 3,
        "preconditions": lambda s: s.medical_teams < s.medical_needed and s.alert_level != "NONE",
        "effect": lambda s: _set(s, medical_teams=s.medical_teams + 1)
    },
    "CLEAR_EVACUATION_ROUTES": {
        "description": "Clear and secure evacuation routes",
        "cost": 2,
        "preconditions": lambda s: not s.routes_cleared and s.alert_level != "NONE",
        "effect": lambda s: _set(s, routes_cleared=True)
    },
    "BEGIN_EVACUATION": {
        "description": "Begin evacuating population to shelters",
        "cost": 5,
        "preconditions": lambda s: (s.shelters_open > 0 and s.teams_deployed > 0 and
                                     s.routes_cleared and s.evacuated < s.population),
        "effect": lambda s: _set(s, evacuated=min(s.population,
                                                   s.evacuated + int(s.population * 0.4 * s.teams_deployed / max(s.teams_needed, 1))))
    },
    "ALLOCATE_SUPPLIES": {
        "description": "Allocate food, water, and medical supplies to shelters",
        "cost": 3,
        "preconditions": lambda s: s.shelters_open > 0 and not s.supplies_allocated,
        "effect": lambda s: _set(s, supplies_allocated=True)
    },
}


def heuristic(state):
    """Admissible heuristic: estimates minimum remaining cost to reach goal."""
    h = 0
    if state.alert_level == "NONE":
        h += 1
    h += max(0, state.shelters_needed - state.shelters_open) * 3
    h += max(0, state.teams_needed - state.teams_deployed) * 4
    h += max(0, state.medical_needed - state.medical_teams) * 3
    if not state.routes_cleared:
        h += 2
    if not state.supplies_allocated:
        h += 3 if state.shelters_open > 0 else 6
    if state.evacuated < state.population:
        evac_per_round = max(1, int(state.population * 0.4))
        h += math.ceil((state.population - state.evacuated) / evac_per_round) * 5
    return h


def astar_search(initial_state, max_iterations=5000):
    """A* Search Algorithm for flood evacuation planning."""
    counter = 0
    frontier = []
    g = 0
    h = heuristic(initial_state)
    heapq.heappush(frontier, (g + h, counter, initial_state, [], g))
    visited = set()
    visited.add(initial_state.to_tuple())
    nodes_explored = 0

    while frontier and nodes_explored < max_iterations:
        f, _, current_state, path, g_cost = heapq.heappop(frontier)
        nodes_explored += 1
        if current_state.is_goal():
            return {"success": True, "path": path, "total_cost": g_cost,
                    "nodes_explored": nodes_explored, "final_state": current_state.to_dict()}

        for action_name, action in ACTIONS.items():
            if action["preconditions"](current_state):
                new_state = action["effect"](current_state)
                state_tuple = new_state.to_tuple()
                if state_tuple not in visited:
                    visited.add(state_tuple)
                    new_g = g_cost + action["cost"]
                    new_path = path + [{"action": action_name,
                                         "description": action["description"],
                                         "cost": action["cost"],
                                         "state_after": new_state.to_dict()}]
                    counter += 1
                    heapq.heappush(frontier, (new_g + heuristic(new_state), counter, new_state, new_path, new_g))

    return {"success": False, "path": path if path else [], "total_cost": 0,
            "nodes_explored": nodes_explored, "message": "Search exhausted"}


def create_initial_state(flood_probability, features_dict=None):
    features = features_dict or {}
    if flood_probability > 0.7:
        risk_level, population, sn, tn, mn = "critical", 150000, 5, 6, 3
    elif flood_probability > 0.55:
        risk_level, population, sn, tn, mn = "high", 100000, 3, 4, 2
    elif flood_probability > 0.4:
        risk_level, population, sn, tn, mn = "medium", 60000, 2, 2, 1
    else:
        risk_level, population, sn, tn, mn = "low", 30000, 1, 1, 1
    pop_score = features.get("PopulationScore", 5)
    population = int(population * (pop_score / 5.0))
    if features.get("DeterioratingInfrastructure", 5) > 7: tn += 1
    if features.get("CoastalVulnerability", 5) > 7: sn += 1
    return FloodState(risk_level=risk_level, population=population,
                      shelters_needed=sn, teams_needed=tn, medical_needed=mn)


def run_agent(flood_probability, features_dict=None):
    """Main entry point for the Flask app."""
    initial_state = create_initial_state(flood_probability, features_dict)
    result = astar_search(initial_state)
    actions_list = [{"action": s["action"], "description": s["description"], "cost": s["cost"]}
                    for s in result.get("path", [])]
    return {
        "initial_state": create_initial_state(flood_probability, features_dict).to_dict(),
        "actions": actions_list,
        "total_cost": result.get("total_cost", 0),
        "nodes_explored": result.get("nodes_explored", 0),
        "search_successful": result.get("success", False),
        "final_state": result.get("final_state", {}),
        "risk_level": initial_state.risk_level
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  STEP 7: Intelligent Agent -- A* Search Evacuation Planning")
    print("=" * 70)
    for prob, label, features in [
        (0.2, "Low Risk", {"PopulationScore": 4, "MonsoonIntensity": 3}),
        (0.5, "Medium Risk", {"PopulationScore": 6, "MonsoonIntensity": 6}),
        (0.85, "Critical Risk", {"PopulationScore": 8, "MonsoonIntensity": 9,
                                  "DeterioratingInfrastructure": 8, "CoastalVulnerability": 7}),
    ]:
        print(f"\n  Scenario: {label} (FloodProbability = {prob})")
        result = run_agent(prob, features)
        print(f"  Nodes explored: {result['nodes_explored']}, Cost: {result['total_cost']}")
        print(f"  Actions:")
        for i, a in enumerate(result['actions'], 1):
            print(f"    {i}. [{a['action']}] {a['description']} (cost: {a['cost']})")
