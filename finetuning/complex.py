import random
import json
from datasets import Dataset

def random_id():
    return f"UE{random.randint(100000, 999999)}"

# List of 10 very complex 5G issues
COMPLEX_ISSUES = [
    {
        "name": "Multi-step RRC and Handover Failure",
        "log_template": """
        [RRC] [ERROR] RRC Connection Setup Failure
        Cause: {rrc_cause}
        UE ID: {ue_id}
        Cell ID: {cell_id}
        PCI: {pci}
        SINR: {sinr} dB
        RSRP: {rsrp} dBm

        [HO] [ERROR] Handover Failure
        Cause: {ho_cause}
        Source Cell ID: {source_cell}
        Target Cell ID: {target_cell}
        UE ID: {ue_id}
        RSRP (Source): {rsrp_source} dBm
        RSRP (Target): {rsrp_target} dBm
        """,
        "causes": ["Radio Link Failure", "Network Congestion", "Weak Target Cell Signal"],
        "resolution": "Optimize radio parameters, reduce congestion, and tune handover thresholds."
    },
    {
        "name": "Correlated Transport and UE Drop",
        "log_template": """
        [TRANSPORT] [CRITICAL] S1-U Path Failure
        Cause: {transport_cause}
        gNB ID: {gnb_id}
        UPF IP: {upf_ip}
        Packet Loss: {packet_loss}%

        [UE_MGMT] [ALERT] Massive UE Drop Detected
        Total Connected UEs Before Drop: {before_drop}
        Dropped UEs: {dropped_ues}
        Cause: {ue_drop_cause}
        CPU Utilization: {cpu_utilization}%
        """,
        "causes": ["Fiber Link Failure", "Backhaul Congestion", "CPU Overload"],
        "resolution": "Check fiber integrity, optimize backhaul routing, and scale CPU resources."
    },
    {
        "name": "Interference-Induced Handover Loop",
        "log_template": """
        [PHY] [WARNING] High Interference Detected
        Cell ID: {cell_id}
        PCI: {pci}
        Interference Level: {interference_level} dBm
        Suspected Source: {suspect_cell}

        [HO] [ERROR] Handover Failure Due to Interference
        UE ID: {ue_id}
        Source Cell ID: {source_cell}
        Target Cell ID: {target_cell}
        RSRP (Source): {rsrp_source} dBm
        RSRP (Target): {rsrp_target} dBm
        """,
        "causes": ["External RF Interference", "Power Control Issues"],
        "resolution": "Adjust frequency planning, implement interference cancellation techniques."
    },
    {
        "name": "Slicing Resource Exhaustion",
        "log_template": """
        [NS] [ERROR] Network Slice Allocation Failure
        UE ID: {ue_id}
        Requested Slice: {slice_type} (Slice ID: {slice_id})
        Cause: {slice_cause}
        """,
        "causes": ["No Available Resources", "Incorrect Slice Mapping", "Policy Enforcement Failure"],
        "resolution": "Reallocate resources dynamically and optimize slice configurations."
    },
    {
        "name": "Neighbor Cell Conflict",
        "log_template": """
        [HO] [ERROR] Handover Rejection Due to Neighbor Conflict
        UE ID: {ue_id}
        Source Cell ID: {source_cell}
        Target Cell ID: {target_cell}
        PCI Conflict Detected: {pci}
        """,
        "causes": ["Duplicate PCI Assignment", "Misconfigured Neighbor Relations"],
        "resolution": "Correct PCI allocations and optimize neighbor relations."
    },
    {
        "name": "Synchronization Failure",
        "log_template": """
        [SYNC] [CRITICAL] gNB Timing Synchronization Failure
        gNB ID: {gnb_id}
        Cause: {sync_cause}
        """,
        "causes": ["Clock Drift", "Backhaul Latency", "Sync Signal Loss"],
        "resolution": "Improve clock synchronization and minimize backhaul delays."
    },
    {
        "name": "Unexpected Call Drop",
        "log_template": """
        [CALL] [ERROR] Unexpected Call Drop
        UE ID: {ue_id}
        Cell ID: {cell_id}
        PCI: {pci}
        Cause: {call_cause}
        """,
        "causes": ["Network Load", "Interference", "Resource Preemption"],
        "resolution": "Increase capacity, manage resource allocation dynamically."
    },
    {
        "name": "Paging Storm",
        "log_template": """
        [PAGING] [ALERT] Excessive Paging Requests Detected
        gNB ID: {gnb_id}
        Paging Load: {paging_load}%
        Cause: {paging_cause}
        """,
        "causes": ["UE Misbehavior", "Core Network Overload"],
        "resolution": "Mitigate paging load by adjusting thresholds and monitoring UE behavior."
    },
    {
        "name": "Control Plane Congestion",
        "log_template": """
        [CONTROL] [ERROR] High Signaling Load Detected
        gNB ID: {gnb_id}
        CPU Utilization: {cpu_utilization}%
        Cause: {control_cause}
        """,
        "causes": ["Excessive UE Registration", "Frequent Session Modifications"],
        "resolution": "Optimize session handling and distribute signaling load."
    },
    {
        "name": "Backhaul Link Instability",
        "log_template": """
        [BACKHAUL] [ERROR] Frequent Link Fluctuations Detected
        gNB ID: {gnb_id}
        Packet Loss: {packet_loss}%
        Cause: {backhaul_cause}
        """,
        "causes": ["Fluctuating Fiber Quality", "Routing Instability"],
        "resolution": "Enhance fiber stability and improve routing resilience."
    }
]

def generate_complex_log():
    issue = random.choice(COMPLEX_ISSUES)
    log_entry = issue["log_template"].format(
        ue_id=random_id(),
        cell_id=random.randint(1001, 1099),
        pci=random.randint(1, 512),
        sinr=random.randint(-20, 10),
        rsrp=random.randint(-130, -80),
        source_cell=random.randint(1001, 1099),
        target_cell=random.randint(1001, 1099),
        rsrp_source=random.randint(-100, -80),
        rsrp_target=random.randint(-130, -90),
        interference_level=random.randint(-90, -60),
        suspect_cell=random.randint(1001, 1099),
        gnb_id=random.randint(5000, 5999),
        upf_ip=f"10.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
        packet_loss=random.randint(50, 99),
        before_drop=random.randint(300, 800),
        dropped_ues=random.randint(50, 200),
        cpu_utilization=random.randint(80, 99),
        **{k: random.choice(issue["causes"]) for k in issue["log_template"].count("{")}
    )
    return {"log": log_entry, "issue": issue["name"], "resolution": issue["resolution"]}
