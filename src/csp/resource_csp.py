"""
Step 8: Constraint Satisfaction Problem -- Resource Allocation
==============================================================
Solver: Backtracking with Forward Checking + MRV heuristic
"""

import copy


class CSP:
    """Generic CSP framework with backtracking, forward checking, and MRV."""
    def __init__(self):
        self.variables = []
        self.domains = {}
        self.constraints = []
        self.trace = []

    def add_variable(self, name, domain):
        self.variables.append(name)
        self.domains[name] = list(domain)

    def add_constraint(self, scope, check_fn, description=""):
        self.constraints.append((scope, check_fn, description))

    def is_consistent(self, var, value, assignment):
        test = dict(assignment)
        test[var] = value
        for scope, check_fn, desc in self.constraints:
            if all(v in test for v in scope):
                if not check_fn(test):
                    self.trace.append({"type": "constraint_violated", "variable": var,
                                       "value": value, "constraint": desc})
                    return False
        return True

    def select_unassigned_variable(self, assignment):
        unassigned = [v for v in self.variables if v not in assignment]
        if not unassigned: return None
        return min(unassigned, key=lambda v: sum(1 for val in self.domains[v]
                                                  if self.is_consistent(v, val, assignment)))

    def forward_check(self, var, value, assignment, domains):
        new_domains = {k: list(v) for k, v in domains.items()}
        test = dict(assignment)
        test[var] = value
        for other_var in self.variables:
            if other_var in test or other_var == var: continue
            pruned = [val for val in new_domains[other_var] if self.is_consistent(other_var, val, test)]
            if not pruned: return None
            new_domains[other_var] = pruned
        return new_domains

    def backtrack(self, assignment, domains, depth=0):
        if len(assignment) == len(self.variables):
            return dict(assignment)
        var = self.select_unassigned_variable(assignment)
        if var is None: return None
        self.trace.append({"type": "selecting_variable", "variable": var, "depth": depth})
        for value in domains[var]:
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                self.trace.append({"type": "assignment", "variable": var, "value": value, "depth": depth})
                new_domains = self.forward_check(var, value, assignment, domains)
                if new_domains is not None:
                    result = self.backtrack(assignment, new_domains, depth + 1)
                    if result is not None: return result
                del assignment[var]
                self.trace.append({"type": "backtrack", "variable": var, "value": value, "depth": depth})
        return None

    def solve(self):
        self.trace = []
        return self.backtrack({}, {k: list(v) for k, v in self.domains.items()})


def build_flood_csp(risk_level, num_zones=4, num_shelters=3, num_teams=5, num_supplies=4):
    csp = CSP()
    if risk_level == "critical":
        num_zones, num_shelters, num_teams, num_supplies = max(num_zones,5), max(num_shelters,4), max(num_teams,7), max(num_supplies,5)
        shelter_capacity, min_teams = 3, 1
    elif risk_level == "high":
        num_zones, num_shelters, num_teams = max(num_zones,4), max(num_shelters,3), max(num_teams,5)
        shelter_capacity, min_teams = 3, 1
    else:
        num_zones, num_shelters, num_teams, num_supplies = max(num_zones,3), max(num_shelters,2), max(num_teams,3), max(num_supplies,3)
        shelter_capacity, min_teams = 4, 0

    zones = [f"Zone_{i+1}" for i in range(num_zones)]
    shelters = [f"Shelter_{chr(65+i)}" for i in range(num_shelters)]
    teams = [f"Team_{i+1}" for i in range(num_teams)]
    supplies = [f"Supply_{i+1}" for i in range(num_supplies)]
    priorities = list(range(1, num_zones + 1))

    for z in zones: csp.add_variable(f"shelter_{z}", shelters)
    for t in teams: csp.add_variable(f"team_{t}", zones)
    for s in supplies: csp.add_variable(f"supply_{s}", shelters)
    for z in zones: csp.add_variable(f"priority_{z}", priorities)

    # Shelter capacity
    shelter_vars = [f"shelter_{z}" for z in zones]
    for shelter in shelters:
        def make_cap(sn, c, sv=shelter_vars):
            def check(a): return sum(1 for v in sv if a.get(v)==sn) <= c
            return check
        csp.add_constraint(tuple(shelter_vars), make_cap(shelter, shelter_capacity),
                          f"Shelter {shelter} capacity <= {shelter_capacity}")

    # Adjacent zones different shelters
    for i in range(len(zones)-1):
        v1, v2 = f"shelter_{zones[i]}", f"shelter_{zones[i+1]}"
        def make_diff(a, b):
            def check(assignment): return assignment.get(a) != assignment.get(b) if a in assignment and b in assignment else True
            return check
        csp.add_constraint((v1, v2), make_diff(v1, v2), f"Adjacent zones use different shelters")

    # All different priorities
    priority_vars = [f"priority_{z}" for z in zones]
    def all_diff(a):
        vals = [a[v] for v in priority_vars if v in a]
        return len(vals) == len(set(vals))
    csp.add_constraint(tuple(priority_vars), all_diff, "All zones different priorities")

    # Min teams per zone
    if min_teams > 0:
        team_vars = [f"team_{t}" for t in teams]
        for zone in zones:
            def make_min(zn, mt, tv=team_vars):
                def check(a):
                    if not all(v in a for v in tv): return True
                    return sum(1 for v in tv if a.get(v)==zn) >= mt
                return check
            csp.add_constraint(tuple(team_vars), make_min(zone, min_teams), f"{zone} min {min_teams} teams")

    # Supply coverage
    supply_vars = [f"supply_{s}" for s in supplies]
    for z in zones:
        sv = f"shelter_{z}"
        def make_sup(s_var, s_list=supply_vars):
            def check(a):
                if s_var not in a: return True
                if not all(v in a for v in s_list): return True
                return any(a.get(v)==a[s_var] for v in s_list)
            return check
        csp.add_constraint((sv,)+tuple(supply_vars), make_sup(sv), f"Shelter for {z} gets supplies")

    return csp, {"zones": zones, "shelters": shelters, "teams": teams, "supplies": supplies}


def solve_csp(risk_level, num_zones=4, num_shelters=3, num_teams=5, num_supplies=4):
    """Main entry point for Flask app."""
    csp, config = build_flood_csp(risk_level, num_zones, num_shelters, num_teams, num_supplies)
    solution = csp.solve()
    if solution:
        sa, ta, su, ep = {}, {}, {}, {}
        for var, val in solution.items():
            if var.startswith("shelter_"): sa[var.replace("shelter_","")] = val
            elif var.startswith("team_"): ta[var.replace("team_","")] = val
            elif var.startswith("supply_"): su[var.replace("supply_","")] = val
            elif var.startswith("priority_"): ep[var.replace("priority_","")] = val
        eo = [z for z, p in sorted(ep.items(), key=lambda x: x[1])]
        ts = csp.trace
        return {"solved": True, "constraints_satisfied": True,
                "shelter_assignments": sa, "team_assignments": ta,
                "supply_allocations": su, "evacuation_priorities": ep,
                "evacuation_order": eo,
                "config": {"num_zones": len(config["zones"]), "num_shelters": len(config["shelters"]),
                           "num_teams": len(config["teams"]), "risk_level": risk_level},
                "trace_summary": {"total_steps": len(ts),
                                  "assignments": sum(1 for t in ts if t["type"]=="assignment"),
                                  "backtracks": sum(1 for t in ts if t["type"]=="backtrack"),
                                  "violations": sum(1 for t in ts if t["type"]=="constraint_violated")},
                "trace": ts[:30]}
    return {"solved": False, "constraints_satisfied": False,
            "message": "No valid allocation found",
            "trace_summary": {"total_steps": len(csp.trace),
                              "backtracks": sum(1 for t in csp.trace if t["type"]=="backtrack")}}


if __name__ == "__main__":
    print("=" * 70)
    print("  STEP 8: CSP -- Resource Allocation")
    print("=" * 70)
    for risk in ["low", "high", "critical"]:
        print(f"\n  Risk: {risk.upper()}")
        r = solve_csp(risk)
        if r["solved"]:
            print(f"  Solved! Shelters: {r['shelter_assignments']}")
            print(f"  Teams: {r['team_assignments']}")
            print(f"  Evacuation order: {' -> '.join(r['evacuation_order'])}")
            print(f"  Stats: {r['trace_summary']['assignments']} assignments, {r['trace_summary']['backtracks']} backtracks")
        else:
            print(f"  No solution found.")
