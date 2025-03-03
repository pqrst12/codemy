import random
import json
from datetime import datetime, timedelta
from datasets import Dataset

# Define possible issues with log templates
ISSUES = [
    {
        "name": "RRC Connection Failure",
        "log_template": "[RRC] [ERROR] RRC Connection Setup Failure\nCause: {cause}\nUE ID: {ue_id}\nCell ID: {cell_id}\nPCI: {pci}\nSINR: {sinr} dB\nRSRP: {rsrp} dBm",
        "causes": ["Radio Link Failure (RLF)", "Network Congestion", "Authentication Failure"],
        "resolution": "Check signal strength, reduce congestion, and ensure proper authentication handling. If RLF occurs frequently, consider adjusting handover parameters and increasing transmission power. Authentication failures should be debugged by reviewing security credentials and verifying UE identity."
    },
    {
        "name": "Handover Failure",
        "log_template": "[HO] [ERROR] Handover Failure\nCause: {cause}\nSource Cell ID: {source_cell}\nTarget Cell ID: {target_cell}\nUE ID: {ue_id}\nSINR: {sinr} dB\nRSRP Source: {rsrp_source} dBm\nRSRP Target: {rsrp_target} dBm",
        "causes": ["Interference", "Timing Advance Issue", "Resource Unavailability"],
        "resolution": "Analyze interference levels, adjust timing advance, and ensure target cell has available resources. If resource unavailability is the issue, increase cell capacity or implement dynamic spectrum sharing to improve handover success rates."
    },
    {
        "name": "Interference Issue",
        "log_template": "[INTERFERENCE] [WARNING] High Interference Detected\nUE ID: {ue_id}\nCell ID: {cell_id}\nInterference Level: {interference_level} dBm",
        "causes": ["External Interference", "Hardware Failure", "Neighboring Cell Congestion"],
        "resolution": "Identify interference sources by conducting spectrum analysis. Adjust antenna parameters, modify frequency reuse patterns, and apply filtering techniques to mitigate external noise. If hardware issues are detected, perform immediate maintenance or replacement."
    },
    {
        "name": "Network Slicing Misconfiguration",
        "log_template": "[SLICING] [ERROR] Slice Configuration Mismatch\nSlice Type: {slice_type}\nSlice ID: {slice_id}\nUE ID: {ue_id}\nCell ID: {cell_id}",
        "causes": ["Misconfigured Slice Parameters", "Improper Resource Allocation", "UPF Routing Issue"],
        "resolution": "Verify slice parameters to match the network's QoS requirements. Ensure correct slice allocation per service type (eMBB, URLLC, mMTC) and confirm UPF routes traffic correctly through network elements."
    },
    {
        "name": "Transport Network Failure",
        "log_template": "[TRANSPORT] [ERROR] Packet Loss Detected\nUPF IP: {upf_ip}\nPacket Loss: {packet_loss}%\nUE ID: {ue_id}\nCell ID: {cell_id}",
        "causes": ["Link Failure", "High Latency", "Router Congestion"],
        "resolution": "Check transport link status, reduce congestion, and optimize routing paths. Use QoS prioritization for critical traffic, and deploy redundancy mechanisms like secondary transport paths to prevent failures."
    },
    {
        "name": "Massive UE Drops",
        "log_template": "[OVERLOAD] [CRITICAL] High UE Drop Rate\nDropped UEs: {dropped_ues}\nBefore Drop: {before_drop}\nCPU Utilization: {cpu_utilization}%\nCell ID: {cell_id}",
        "causes": ["CPU Overload", "High Traffic Volume", "Resource Exhaustion"],
        "resolution": "Optimize resource allocation, increase CPU capacity, and implement load balancing strategies. If the issue persists, consider dynamic scaling and network slicing to distribute load effectively."
    },
    {
        "name": "Synchronization Issue",
        "log_template": "[SYNC] [ERROR] Synchronization Failure Detected\nSuspect Cell: {suspect_cell}\nUE ID: {ue_id}\nCell ID: {cell_id}",
        "causes": ["Timing Offset", "GPS Signal Loss", "Synchronization Source Failure"],
        "resolution": "Verify synchronization source, check GPS signal, and recalibrate timing settings. Deploy alternative sync sources like IEEE 1588 PTP in case of GPS failures."
    },
    {
        "name": "Paging Failure",
        "log_template": "[PAGING] [ERROR] Paging Message Not Delivered\nUE ID: {ue_id}\nCell ID: {cell_id}\nPaging Cause: {cause}",
        "causes": ["UE Unreachable", "Paging Channel Congestion", "Core Network Issue"],
        "resolution": "Increase paging retries, optimize paging channel, and check core network connectivity. If congestion is persistent, consider increasing paging capacity or dynamically allocating resources based on network load."
    }
]

def random_time():
    return (datetime.now() - timedelta(minutes=random.randint(0, 1440))).strftime("%Y-%m-%d %H:%M:%S")

def random_id():
    return f"UE{random.randint(100000, 999999)}"

def generate_log_entry():
    issue = random.choice(ISSUES)
    log_entry = issue["log_template"].format(
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
        slice_id=random.randint(1, 1000),
        cause=random.choice(issue["causes"])
    )
    return {
        "instruction": "Analyze the following 5G network log trace and provide the root cause and resolution steps.",
        "input": f"{random_time()} {log_entry}",
        "output": f"Issue: {issue['name']}\nRoot Cause: {log_entry.split('Cause: ')[1].split('\\n')[0]}\nResolution: {issue['resolution']}"
    }

# Generate dataset
data = [generate_log_entry() for _ in range(5000)]

dataset = Dataset.from_list(data)
dataset.push_to_hub("5g_network_debugging_dataset")
