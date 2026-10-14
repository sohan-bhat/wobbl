"""An optional cloud machine, rented through the user's own AWS CLI and credentials, for a CalculiX job too big for
the machines they have.

Nothing here runs unless asked for by name. Wobbl first lays out a plan with the instance, its cores and memory,
the predicted run time, the estimated cost at on demand prices, the hard stop after which the machine shuts
itself down and every AWS command it will run. It starts nothing until that exact plan is confirmed, by typing
yes at the prompt or by passing the plan's token back, and a token from any other plan is refused. The AWS CLI
signs its requests with the credentials it already has, and Wobbl never reads or stores them.

Once the machine is up it is one more SSH remote, so the job, its progress and its checkpoints work as they do
on a workstation. The instance is terminated when the results are back or the job is cancelled, and its own
shutdown timer ends it even if Wobbl never returns.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field

PRICES = "AWS on demand Linux rates in us-east-1 as of 2025"
# instance, vCPUs, physical cores, memory GB, USD per hour
INSTANCES = [
    ("c7i.2xlarge", 8, 4, 16, 0.357),
    ("c7i.4xlarge", 16, 8, 32, 0.714),
    ("c7i.8xlarge", 32, 16, 64, 1.428),
    ("c7i.16xlarge", 64, 32, 128, 2.856),
    ("m7i.8xlarge", 32, 16, 128, 1.6128),
    ("r7i.8xlarge", 32, 16, 256, 2.1168),
    ("r7i.16xlarge", 64, 32, 512, 4.2336),
]
SETUP_HOURS = 0.25        # booting, installing CalculiX from conda-forge and sending the input
GB_MONTH = 0.08           # gp3 volume, USD per GB month
IMAGE = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
MICROMAMBA = "https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64"


class CloudError(RuntimeError):
    pass


def machine(row):
    from ..estimate import Machine

    name, _, cores, gb, price = row
    return Machine(f"an AWS {name}", cores, cores, gb * 1e9, float("inf"), price_per_hour=price)


def cheapest(cost, limit_s: float) -> dict | None:
    """The cheapest instance that fits the run in memory and finishes inside the limit. CalculiX runs only."""
    from ..estimate import predict

    if cost.solver != "calculix":
        return None
    best = None
    for row in INSTANCES:
        e = predict(cost, machine(row), limit_s / 3600.0, advise_options=False)
        if not e.fits or e.seconds > limit_s or e.peak_bytes > 0.7 * row[3] * 1e9:
            continue
        hours = e.seconds / 3600.0 + SETUP_HOURS
        disk_gb = max(30.0, 2.0 * e.disk_bytes / 1e9)
        usd = hours * row[4] + disk_gb * GB_MONTH * hours / 730.0
        if best is None or usd < best["cost_usd"]:
            best = {"instance": row[0], "cores": row[2], "ram_gb": row[3], "price_per_hour": row[4],
                    "time_s": round(e.seconds), "hours": round(hours, 2), "disk_gb": round(disk_gb),
                    "cost_usd": round(usd, 2), "prices": PRICES}
    return best


@dataclass
class Plan:
    instance: str
    region: str
    key_name: str
    security_group: str
    cores: int
    ram_gb: float
    price_per_hour: float
    time_s: float
    hours: float
    max_hours: float
    disk_gb: int
    cost_usd: float
    max_cost_usd: float
    job: str
    commands: list[str] = field(default_factory=list)
    token: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def plan(cost, instance: str, region: str, key_name: str, security_group: str, job: str,
         max_hours: float | None = None, price: float | None = None) -> Plan:
    """What renting ``instance`` for this run would mean, with nothing started."""
    from ..estimate import predict

    row = next((r for r in INSTANCES if r[0] == instance), None)
    if row is None:
        if price is None:
            raise CloudError(f"Wobbl has no price for '{instance}'. Pass --price-per-hour, or pick one of "
                             + ", ".join(r[0] for r in INSTANCES))
        raise CloudError(f"Wobbl does not know the cores and memory of '{instance}'. Pick one of "
                         + ", ".join(r[0] for r in INSTANCES))
    if not key_name or not security_group:
        raise CloudError("a cloud run needs your EC2 key pair name and a security group that lets SSH in, given "
                         "as --key-name and --security-group")
    if price is not None:
        row = (*row[:4], float(price))
    e = predict(cost, machine(row), None, advise_options=False)
    if not e.fits:
        raise CloudError(f"the run does not fit an {instance} either. " + " ".join(e.problems))
    hours = e.seconds / 3600.0 + SETUP_HOURS
    stop = float(max_hours) if max_hours else round(2.0 * hours + 1.0, 1)
    disk_gb = int(max(30, round(2.0 * e.disk_bytes / 1e9)))
    p = Plan(instance, region, key_name, security_group, row[2], row[3], row[4], round(e.seconds), round(hours, 2),
             stop, disk_gb, round(hours * row[4], 2), round(stop * row[4], 2), job)
    p.commands = [
        f"aws ssm get-parameter --region {region} --name {IMAGE}",
        f"aws ec2 run-instances --region {region} --instance-type {instance} --key-name {key_name} "
        f"--security-group-ids {security_group} --instance-initiated-shutdown-behavior terminate "
        f"--block-device-mappings DeviceName=/dev/sda1,Ebs={{VolumeSize={disk_gb},DeleteOnTermination=true}} "
        f"--user-data (installs CalculiX and runs shutdown -h +{int(stop * 60)})",
        f"aws ec2 wait instance-running --region {region}",
        f"aws ec2 describe-instances --region {region}",
        f"aws ec2 terminate-instances --region {region} (when the results are back or the job is cancelled)",
    ]
    p.token = hashlib.sha256(json.dumps({k: v for k, v in asdict(p).items() if k != "token"},
                                        sort_keys=True).encode()).hexdigest()[:12]
    return p


def bootstrap(p: Plan) -> str:
    """User data for the instance. The shutdown timer comes first, so the machine ends itself whatever else
    happens, and with the terminate shutdown behavior a shutdown ends the rental."""
    return f"""#!/bin/sh
shutdown -h +{int(p.max_hours * 60)}
cd /home/ubuntu
curl -fsSL {MICROMAMBA} -o micromamba && chmod +x micromamba
./micromamba create -y -q -r /home/ubuntu/.wobbl/mamba -p /home/ubuntu/.wobbl/env -c conda-forge calculix
chown -R ubuntu:ubuntu /home/ubuntu/.wobbl
touch /home/ubuntu/.wobbl-ready
"""


def _aws(args: list[str], run=subprocess.run) -> str:
    r = run(["aws", *args, "--output", "json"], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise CloudError(f"aws {' '.join(args[:2])} failed ({(r.stderr or r.stdout).strip()[:300]})")
    return r.stdout


def launch(p: Plan, confirm: str, user: str = "ubuntu", run=subprocess.run) -> dict:
    """Rent the machine in the plan, but only for the plan's own token. Returns its instance id and SSH host."""
    if not confirm or confirm != p.token:
        raise CloudError("nothing was started, because this plan has not been confirmed. Show it to the user and "
                         f"pass confirm={p.token} only once they agree to its cost")
    if run is subprocess.run and not shutil.which("aws"):
        raise CloudError("the AWS CLI is not installed. Wobbl only rents machines through your own AWS CLI and "
                         "its credentials")
    image = json.loads(_aws(["ssm", "get-parameter", "--region", p.region, "--name", IMAGE], run))
    ami = image["Parameter"]["Value"]
    out = json.loads(_aws([
        "ec2", "run-instances", "--region", p.region, "--image-id", ami, "--instance-type", p.instance,
        "--key-name", p.key_name, "--security-group-ids", p.security_group, "--count", "1",
        "--instance-initiated-shutdown-behavior", "terminate", "--user-data", bootstrap(p),
        "--block-device-mappings",
        f"DeviceName=/dev/sda1,Ebs={{VolumeSize={p.disk_gb},DeleteOnTermination=true}}",
        "--tag-specifications", f"ResourceType=instance,Tags=[{{Key=wobbl-job,Value={p.job}}}]"], run))
    iid = out["Instances"][0]["InstanceId"]
    try:
        _aws(["ec2", "wait", "instance-running", "--region", p.region, "--instance-ids", iid], run)
        desc = json.loads(_aws(["ec2", "describe-instances", "--region", p.region, "--instance-ids", iid], run))
        dns = desc["Reservations"][0]["Instances"][0].get("PublicDnsName") or \
            desc["Reservations"][0]["Instances"][0].get("PublicIpAddress")
    except Exception:
        terminate(iid, p.region, run)
        raise
    return {"instance_id": iid, "region": p.region, "host": f"{user}@{dns}" if user else dns,
            "started": time.time(), "max_hours": p.max_hours, "price_per_hour": p.price_per_hour}


def terminate(instance_id: str, region: str, run=subprocess.run) -> bool:
    try:
        _aws(["ec2", "terminate-instances", "--region", region, "--instance-ids", instance_id], run)
        return True
    except CloudError:
        return False
