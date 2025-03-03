import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Issue categories and their details (expanded)
ISSUES = [
    {
        "name": "RRC Connection Failure",
        "log_template": "[RRC] [ERROR] RRC Connection Setup Failure\nCause: {cause}\nUE ID: {ue_id}\nCell ID: {cell_id}\nPCI: {pci}\nSINR: {sinr} dB\nRSRP: {rsrp} dBm",
        "causes": ["Radio Link Failure (RLF)", "Network Congestion", "Authentication Failure"],
        "resolution": "Check signal strength, reduce congestion, and ensure proper authentication handling."
    },
    {
        "name": "Handover Failure",
        "log_template": "[HO] [ERROR] Handover Failure\nCause: {cause}\nSource Cell ID: {source_cell}\nTarget Cell ID: {target_cell}\nUE ID: {ue_id}\nRSRP (Source): {rsrp_source} dBm\nRSRP (Target): {rsrp_target} dBm",
        "causes": ["Weak Target Cell Signal", "Neighbor Cell Misconfiguration", "High Network Load"],
        "resolution": "Adjust handover thresholds, optimize neighbor lists, and ensure target cell availability."
    },
    {
        "name": "Interference Issue",
        "log_template": "[PHY] [WARNING] High Interference Detected\nCell ID: {cell_id}\nPCI: {pci}\nInterference Level: {interference_level} dBm\nSuspected Source: {suspect_cell}",
        "causes": ["Overlapping Frequencies", "External RF Interference", "Power Control Issues"],
        "resolution": "Optimize frequency planning, adjust power levels, and use advanced interference mitigation techniques."
    },
    {
        "name": "Transport Network Failure",
        "log_template": "[TRANSPORT] [CRITICAL] S1-U Path Failure\nCause: {cause}\ngNB ID: {gnb_id}\nUPF IP: {upf_ip}\nPacket Loss: {packet_loss}%",
        "causes": ["IP Route Unreachable", "Backhaul Congestion", "Fiber Link Failure"],
        "resolution": "Check routing tables, troubleshoot transport connectivity, and ensure redundancy in fiber links."
    },
    {
        "name": "Massive UE Drop",
        "log_template": "[UE_MGMT] [ALERT] Massive UE Drop Detected\nTotal Connected UEs Before Drop: {before_drop}\nDropped UEs: {dropped_ues}\nCause: {cause}\nCPU Utilization: {cpu_utilization}%",
        "causes": ["CPU Overload", "Network Signaling Storm", "Software Crash"],
        "resolution": "Improve load balancing, optimize signaling mechanisms, and upgrade network software."
    },
    {
        "name": "Slicing Configuration Error",
        "log_template": "[NS] [ERROR] Network Slice Allocation Failure\nUE ID: {ue_id}\nRequested Slice: {slice_type} (Slice ID: {slice_id})\nCause: {cause}",
        "causes": ["No Available Resources", "Incorrect Slice Mapping", "Policy Enforcement Failure"],
        "resolution": "Check slice resource allocation, correct policy configurations, and ensure dynamic slice adjustments."
    }
]

# Helper functions
def random_id():
    return hex(random.randint(0x1000, 0xFFFF))[2:].upper()

def random_time():
    start_time = datetime(2025, 1, 1)
    return (start_time + timedelta(seconds=random.randint(0, 31536000))).strftime("%Y-%m-%d %H:%M:%S")

def generate_log(issue, complexity="medium"):
    log_entry = issue["log_template"].format(
        cause=random.choice(issue["causes"]),
        ue_id=random_id(),
        cell_id=random.randint(1001, 1099),
        pci=random.randint(1, 512),
        sinr=random.randint(-20, 10),
        rsrp=random.randint(-130, -80),
        source_cell=random.randint(1001, 1099),
        target_cell=random.randint(1001, 1099),
        rsrp_source=random.randint(-130, -80),
        rsrp_target=random.randint(-130, -80),
        interference_level=random.randint(-120, -60),
        suspect_cell=random.randint(1001, 1099),
        upf_ip=f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}",
        packet_loss=random.randint(0, 100),
        before_drop=random.randint(1000, 5000),
        dropped_ues=random.randint(50, 1000),
        cpu_utilization=random.randint(50, 100),
        slice_type=random.choice(["eMBB", "URLLC", "mMTC"]),
        slice_id=random.randint(1, 1000)
    )
    
    
    return {
        "instruction": "Analyze the following 5G network log trace and provide the root cause and resolution steps.",
        "input": f"{random_time()} {log_entry}{additional_info}",
        "output": f"Issue: {issue['name']}\nRoot Cause: {log_entry.split('Cause: ')[1].split('\\n')[0]}\nResolution: {issue['resolution']}"
    }

# Generate datasets
DATASET_SIZE = 5000
medium_complexity_logs = [generate_log(random.choice(ISSUES), complexity="medium") for _ in range(DATASET_SIZE)]


# Save as JSONL files
output_dir = Path("./dataset")
output_dir.mkdir(parents=True, exist_ok=True)

medium_file = output_dir / "5g_medium_complexity.jsonl"
very_complex_file = output_dir / "5g_very_complex.jsonl"

with open(medium_file, "w") as f:
    for entry in medium_complexity_logs:
        f.write(json.dumps(entry) + "\n")

with open(very_complex_file, "w") as f:
    for entry in very_complex_logs:
        f.write(json.dumps(entry) + "\n")

print(f"Datasets successfully generated:\n - {medium_file} ({DATASET_SIZE} samples)\n - {very_complex_file} ({DATASET_SIZE + 10} samples)")
