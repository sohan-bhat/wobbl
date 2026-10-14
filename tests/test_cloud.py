"""The optional cloud runner prices a run before anything starts, refuses to start without the plan's own token,
and rents through the user's AWS CLI with a machine that shuts itself down. No real AWS call is made here: a
recording stand in answers for the CLI."""
from __future__ import annotations

import json
import subprocess

import pytest

from wobbl import estimate as E
from wobbl.jobs import cloud


class FakeAWS:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        sub = " ".join(argv[1:3])
        out = {"ssm get-parameter": {"Parameter": {"Value": "ami-123"}},
               "ec2 run-instances": {"Instances": [{"InstanceId": "i-abc"}]},
               "ec2 describe-instances": {"Reservations": [{"Instances": [{"PublicDnsName": "ec2-1.example"}]}]}
               }.get(sub, {})
        return subprocess.CompletedProcess(argv, 0, json.dumps(out), "")


def big():
    return E.calculix_cost(1_000_000)


def test_cheapest_fits_memory_and_time():
    best = cloud.cheapest(big(), 24 * 3600)
    assert best is not None and best["ram_gb"] * 0.7 >= 60 and best["time_s"] < 24 * 3600
    for row in cloud.INSTANCES:
        if row[3] * 0.7e9 >= E.predict(big(), cloud.machine(row), advise_options=False).peak_bytes:
            e = E.predict(big(), cloud.machine(row), advise_options=False)
            if e.seconds <= 24 * 3600:
                assert best["cost_usd"] <= (e.seconds / 3600 + cloud.SETUP_HOURS) * row[4] + 1.0
    assert cloud.cheapest(E.explicit_cost(1000, 1000, 0.01), 3600) is None


def test_plan_starts_nothing_and_binds_a_token():
    with pytest.raises(cloud.CloudError, match="key"):
        cloud.plan(big(), "m7i.8xlarge", "us-east-1", "", "", "j1")
    p = cloud.plan(big(), "m7i.8xlarge", "us-east-1", "mykey", "sg-1", "j1")
    assert p.max_hours == pytest.approx(round(2 * p.hours + 1, 1)) and p.cost_usd > 0
    assert p.max_cost_usd == pytest.approx(p.max_hours * p.price_per_hour, rel=0.01)
    assert any("instance-initiated-shutdown-behavior terminate" in c for c in p.commands)
    again = cloud.plan(big(), "m7i.8xlarge", "us-east-1", "mykey", "sg-1", "j1")
    other = cloud.plan(big(), "r7i.8xlarge", "us-east-1", "mykey", "sg-1", "j1")
    assert p.token == again.token != other.token
    with pytest.raises(cloud.CloudError, match="price"):
        cloud.plan(big(), "x9.huge", "us-east-1", "mykey", "sg-1", "j1")


def test_launch_needs_this_plans_token():
    p = cloud.plan(big(), "m7i.8xlarge", "us-east-1", "mykey", "sg-1", "j1")
    other = cloud.plan(big(), "r7i.8xlarge", "us-east-1", "mykey", "sg-1", "j1")
    aws = FakeAWS()
    for token in ("", "yes", other.token):
        with pytest.raises(cloud.CloudError, match="not been confirmed"):
            cloud.launch(p, token, run=aws)
    assert aws.calls == []
    got = cloud.launch(p, p.token, run=aws)
    assert got["instance_id"] == "i-abc" and got["host"] == "ubuntu@ec2-1.example"
    run = next(c for c in aws.calls if c[1:3] == ["ec2", "run-instances"])
    assert run[run.index("--instance-initiated-shutdown-behavior") + 1] == "terminate"
    user_data = run[run.index("--user-data") + 1]
    assert user_data.splitlines()[1] == f"shutdown -h +{int(p.max_hours * 60)}" and "calculix" in user_data
    assert all(c[0] == "aws" for c in aws.calls)
    assert cloud.terminate("i-abc", "us-east-1", run=aws)
    assert aws.calls[-1][1:3] == ["ec2", "terminate-instances"]
