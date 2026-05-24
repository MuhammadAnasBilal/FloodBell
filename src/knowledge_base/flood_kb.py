"""
Step 9: Knowledge-Based System -- Forward Chaining Inference
=============================================================
12 domain-specific IF-THEN rules for flood disaster management.
Inference Engine: Forward chaining until no new facts derived.
"""

RULES = [
    {"id": "R1", "name": "Critical Risk Classification",
     "condition": lambda f: f.get("FloodProbability", 0) > 0.7,
     "action": lambda f: f.update({"risk_level": "critical"}),
     "new_facts": {"risk_level": "critical"},
     "description": "IF FloodProbability > 0.7 THEN risk_level = critical"},
    {"id": "R2", "name": "High Risk Classification",
     "condition": lambda f: 0.4 < f.get("FloodProbability", 0) <= 0.7,
     "action": lambda f: f.update({"risk_level": "high"}),
     "new_facts": {"risk_level": "high"},
     "description": "IF 0.4 < FloodProbability <= 0.7 THEN risk_level = high"},
    {"id": "R3", "name": "Low Risk Classification",
     "condition": lambda f: f.get("FloodProbability", 0) <= 0.4,
     "action": lambda f: f.update({"risk_level": "low"}),
     "new_facts": {"risk_level": "low"},
     "description": "IF FloodProbability <= 0.4 THEN risk_level = low"},
    {"id": "R4", "name": "Immediate Evacuation Required",
     "condition": lambda f: f.get("risk_level") == "critical" and f.get("PopulationScore", 0) > 5,
     "action": lambda f: f.update({"immediate_evacuation": True}),
     "new_facts": {"immediate_evacuation": True},
     "description": "IF risk_level = critical AND PopulationScore > 5 THEN immediate_evacuation = True"},
    {"id": "R5", "name": "Dam Breach Warning",
     "condition": lambda f: f.get("risk_level") == "critical" and f.get("DamsQuality", 0) > 6,
     "action": lambda f: f.update({"dam_breach_warning": True}),
     "new_facts": {"dam_breach_warning": True},
     "description": "IF risk_level = critical AND DamsQuality > 6 THEN dam_breach_warning = True"},
    {"id": "R6", "name": "Urban Flooding Alert",
     "condition": lambda f: f.get("MonsoonIntensity", 0) > 7 and f.get("DrainageSystems", 0) > 5,
     "action": lambda f: f.update({"urban_flooding_alert": True}),
     "new_facts": {"urban_flooding_alert": True},
     "description": "IF MonsoonIntensity > 7 AND DrainageSystems > 5 THEN urban_flooding_alert = True"},
    {"id": "R7", "name": "Open All Shelters",
     "condition": lambda f: f.get("immediate_evacuation") == True,
     "action": lambda f: f.update({"open_all_shelters": True}),
     "new_facts": {"open_all_shelters": True},
     "description": "IF immediate_evacuation = True THEN open_all_shelters = True"},
    {"id": "R8", "name": "Evacuate Downstream Communities",
     "condition": lambda f: f.get("dam_breach_warning") == True,
     "action": lambda f: f.update({"evacuate_downstream": True}),
     "new_facts": {"evacuate_downstream": True},
     "description": "IF dam_breach_warning = True THEN evacuate_downstream = True"},
    {"id": "R9", "name": "Landslide Alert",
     "condition": lambda f: f.get("Deforestation", 0) > 6 and f.get("Landslides", 0) > 5,
     "action": lambda f: f.update({"landslide_alert": True}),
     "new_facts": {"landslide_alert": True},
     "description": "IF Deforestation > 6 AND Landslides > 5 THEN landslide_alert = True"},
    {"id": "R10", "name": "Block Mountain Routes",
     "condition": lambda f: f.get("landslide_alert") == True,
     "action": lambda f: f.update({"block_mountain_routes": True}),
     "new_facts": {"block_mountain_routes": True},
     "description": "IF landslide_alert = True THEN block_mountain_routes = True"},
    {"id": "R11", "name": "Deploy Emergency Pumps",
     "condition": lambda f: f.get("urban_flooding_alert") == True and f.get("IneffectiveDisasterPreparedness", 0) > 5,
     "action": lambda f: f.update({"deploy_emergency_pumps": True}),
     "new_facts": {"deploy_emergency_pumps": True},
     "description": "IF urban_flooding_alert AND IneffectiveDisasterPreparedness > 5 THEN deploy_emergency_pumps"},
    {"id": "R12", "name": "Activate National Disaster Response",
     "condition": lambda f: f.get("open_all_shelters") == True and f.get("evacuate_downstream") == True,
     "action": lambda f: f.update({"activate_national_disaster_response": True}),
     "new_facts": {"activate_national_disaster_response": True},
     "description": "IF open_all_shelters AND evacuate_downstream THEN activate_national_disaster_response"},
]


def forward_chaining(initial_facts):
    facts = dict(initial_facts)
    trace = []
    fired_rules = set()
    for iteration in range(50):
        new_this_round = False
        for rule in RULES:
            if rule["id"] in fired_rules: continue
            try:
                if rule["condition"](facts):
                    old_facts = dict(facts)
                    rule["action"](facts)
                    new_keys = {k for k in facts if k not in old_facts or facts[k] != old_facts[k]}
                    if new_keys:
                        fired_rules.add(rule["id"])
                        new_this_round = True
                        trace.append({"step": len(trace)+1, "iteration": iteration+1,
                                      "rule_id": rule["id"], "rule_name": rule["name"],
                                      "rule_description": rule["description"],
                                      "new_facts": {k: facts[k] for k in new_keys},
                                      "total_facts_count": len(facts)})
            except: continue
        if not new_this_round: break
    return facts, trace


def generate_recommendations(facts):
    recs = []
    risk = facts.get("risk_level", "unknown")
    if risk == "critical":
        recs.append({"priority": "CRITICAL", "icon": "!!", "text": "CRITICAL FLOOD RISK -- Immediate action required."})
    elif risk == "high":
        recs.append({"priority": "HIGH", "icon": "!", "text": "HIGH FLOOD RISK -- Prepare for potential evacuation."})
    elif risk == "low":
        recs.append({"priority": "LOW", "icon": "i", "text": "LOW FLOOD RISK -- Continue routine monitoring."})
    if facts.get("immediate_evacuation"):
        recs.append({"priority": "CRITICAL", "icon": "!!", "text": "IMMEDIATE EVACUATION: Dense population in critical zone. Begin mass evacuation."})
    if facts.get("dam_breach_warning"):
        recs.append({"priority": "CRITICAL", "icon": "!!", "text": "DAM BREACH WARNING: Alert downstream communities."})
    if facts.get("open_all_shelters"):
        recs.append({"priority": "HIGH", "icon": "!", "text": "OPEN ALL SHELTERS: Activate all emergency shelters."})
    if facts.get("evacuate_downstream"):
        recs.append({"priority": "HIGH", "icon": "!", "text": "EVACUATE DOWNSTREAM: Move communities to higher ground."})
    if facts.get("urban_flooding_alert"):
        recs.append({"priority": "HIGH", "icon": "!", "text": "URBAN FLOODING ALERT: Deploy pumping stations."})
    if facts.get("deploy_emergency_pumps"):
        recs.append({"priority": "HIGH", "icon": "!", "text": "DEPLOY EMERGENCY PUMPS to low-lying urban areas."})
    if facts.get("landslide_alert"):
        recs.append({"priority": "HIGH", "icon": "!", "text": "LANDSLIDE ALERT: Issue warnings for hilly areas."})
    if facts.get("block_mountain_routes"):
        recs.append({"priority": "HIGH", "icon": "!", "text": "BLOCK MOUNTAIN ROUTES: Close mountain roads."})
    if facts.get("activate_national_disaster_response"):
        recs.append({"priority": "CRITICAL", "icon": "!!", "text": "ACTIVATE NDMA: Request federal emergency support."})
    if not recs:
        recs.append({"priority": "INFO", "icon": "i", "text": "No specific alerts. Continue monitoring."})
    return recs


def run_inference(flood_probability, features_dict=None):
    """Main entry point for Flask app."""
    initial_facts = {"FloodProbability": flood_probability}
    if features_dict: initial_facts.update(features_dict)
    final_facts, trace = forward_chaining(initial_facts)
    recommendations = generate_recommendations(final_facts)
    input_keys = set(["FloodProbability"] + list((features_dict or {}).keys()))
    derived = {k: v for k, v in final_facts.items() if k not in input_keys}
    return {
        "initial_facts": {k: v for k, v in initial_facts.items()
                         if k in ["FloodProbability","MonsoonIntensity","PopulationScore",
                                  "DamsQuality","Deforestation","Landslides","DrainageSystems",
                                  "IneffectiveDisasterPreparedness"]},
        "derived_facts": derived, "facts": final_facts, "trace": trace,
        "rules_fired": len(trace), "recommendations": recommendations,
        "rules_reference": [{"id": r["id"], "description": r["description"]} for r in RULES]
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  STEP 9: Knowledge-Based System -- Forward Chaining")
    print("=" * 70)
    print("\n  Example 1: HIGH-RISK CASE")
    r1 = run_inference(0.82, {"MonsoonIntensity": 8, "PopulationScore": 7, "DamsQuality": 7,
                               "Deforestation": 8, "Landslides": 6, "DrainageSystems": 6,
                               "IneffectiveDisasterPreparedness": 7})
    for s in r1["trace"]:
        print(f"    Step {s['step']}: [{s['rule_id']}] {s['rule_name']} -> {s['new_facts']}")
    print(f"  Recommendations: {len(r1['recommendations'])}")
    for rec in r1["recommendations"]:
        print(f"    [{rec['priority']}] {rec['text']}")

    print("\n  Example 2: LOW-RISK CASE")
    r2 = run_inference(0.25, {"MonsoonIntensity": 3, "PopulationScore": 4, "DamsQuality": 2})
    for s in r2["trace"]:
        print(f"    Step {s['step']}: [{s['rule_id']}] {s['rule_name']} -> {s['new_facts']}")
    print(f"  Recommendations: {len(r2['recommendations'])}")

    print(f"\n  Comparison: High-risk fired {r1['rules_fired']} rules, Low-risk fired {r2['rules_fired']} rules")
