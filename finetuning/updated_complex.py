import random
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
        "name": "Slice Admission Failure and UE Registration Drop",
        "log_template": """
        [SLICE] [ERROR] Slice Admission Failure
        Slice Type: {slice_type}
        Slice ID: {slice_id}
        Cause: {slice_cause}

        [UE_REG] [ALERT] UE Registration Drop
        UE ID: {ue_id}
        Cause: {ue_drop_cause}
        """,
        "causes": ["Insufficient Slice Resources", "Incorrect Slice Mapping", "Slice Policy Violation"],
        "resolution": "Reconfigure slice policies and ensure resource allocation is optimized."
    },
    {
        "name": "Clock Sync Failure and Paging Congestion",
        "log_template": """
        [SYNC] [CRITICAL] Clock Sync Failure
        gNB ID: {gnb_id}
        Cause: {sync_cause}

        [PAGING] [WARNING] High Paging Load Detected
        Load: {paging_load}%
        Cause: {paging_cause}
        """,
        "causes": ["Clock Drift", "Backhaul Latency", "Sync Signal Loss"],
        "resolution": "Synchronize network clocks and optimize paging configurations."
    },
    {
        "name": "Signaling Storm and Control Plane Congestion",
        "log_template": """
        [CONTROL] [ERROR] Control Plane Congestion
        gNB ID: {gnb_id}
        Cause: {control_cause}

        [CORE] [CRITICAL] Excessive Signaling Messages
        Cause: {signaling_cause}
        """,
        "causes": ["Excessive UE Registrations", "Frequent Session Modifications", "Signaling Storm"],
        "resolution": "Mitigate excessive signaling by adjusting session timers and load balancing."
    },
    {
        "name": "Backhaul Packet Loss and Service Degradation",
        "log_template": """
        [BACKHAUL] [ERROR] High Packet Loss Detected
        Cause: {backhaul_cause}
        Packet Loss: {packet_loss}%
        
        [SERVICE] [CRITICAL] Service Degradation Detected
        UE ID: {ue_id}
        Slice ID: {slice_id}
        SINR: {sinr} dB
        """,
        "causes": ["Fluctuating Fiber Quality", "Routing Instability", "Packet Loss Burst"],
        "resolution": "Stabilize backhaul routing and implement quality-of-service monitoring."
    },
    {
        "name": "Core Network Overload and Call Setup Failure",
        "log_template": """
        [CORE] [ERROR] Core Network Overload
        Load: {cpu_utilization}%
        
        [CALL] [CRITICAL] Call Setup Failure
        Cause: {call_cause}
        """,
        "causes": ["Network Load", "Interference", "Resource Preemption"],
        "resolution": "Balance core network traffic and prioritize critical calls."
    },
    {
        "name": "Unstable Carrier Aggregation and UE Throughput Drop",
        "log_template": """
        [CA] [WARNING] Carrier Aggregation Instability
        Primary Cell: {cell_id}
        Secondary Cell: {source_cell}
        Cause: {ca_cause}

        [THROUGHPUT] [ALERT] UE Throughput Drop
        UE ID: {ue_id}
        RSRP: {rsrp} dBm
        SINR: {sinr} dB
        """,
        "causes": ["Poor Secondary Cell Quality", "Carrier Scheduling Conflict"],
        "resolution": "Optimize carrier aggregation parameters and improve scheduling algorithms."
    },
    {
        "name": "Frequent RLFs and Poor Mobility Performance",
        "log_template": """
        [RLF] [ERROR] Repeated Radio Link Failures
        UE ID: {ue_id}
        Cell ID: {cell_id}
        Cause: {rlf_cause}

        [MOBILITY] [WARNING] Poor Mobility Performance
        Handover Attempts: {ho_attempts}
        Successful Handovers: {ho_success}
        """,
        "causes": ["Coverage Gaps", "Interference", "Improper Power Control"],
        "resolution": "Enhance coverage planning and fine-tune handover parameters."
    }
]

# Generate log samples
def generate_samples(n=5000):
    samples = []
    for _ in range(n):
        issue = random.choice(COMPLEX_ISSUES)
        log = issue["log_template"].format(
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
            ho_attempts=random.randint(10, 50),
            ho_success=random.randint(5, 30),
            **{k: random.choice(v) for k, v in issue["causes"]}
        )
        samples.append({"log": log, "issue": issue["name"], "resolution": issue["resolution"]})
    return samples

# Generate dataset
dataset = Dataset.from_list(generate_samples(5000))
dataset.save_to_disk("5g_network_logs")
