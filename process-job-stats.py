#!/usr/bin/env python3

import argparse
from datetime import datetime
import sys
import csv

JOB_FIELDS = [
    "job_id",
    "job_name",
    "username",
    "account",
    "partition",
    "elapsed",
    "nodes",
    "cpus",
    "tres",
    "submit_time",
    "start_time",
    "nodelist",
    "gpus",
    "wait_time_hours",
    "cpu_hours",
    "gpu_hours",
]


def calculate_wait_time_hours(iso1: str, iso2: str) -> float:
    dt1 = datetime.fromisoformat(iso1)
    dt2 = datetime.fromisoformat(iso2)
    delta = dt2 - dt1
    return abs(delta.total_seconds() / 60 / 60)


def test_calculate_wait_time_hours():
    assert (
        calculate_wait_time_hours("2025-02-03T23:38:00", "2025-02-03T23:53:00") == 0.25
    )


def calculate_gpus_from_tres(tres: str) -> int:
    if "gres/gpu=" not in tres:
        return 0
    for part in tres.split(","):
        if "gres/gpu=" in part:
            return int(part.split("=")[1])
    raise ValueError(f"error parsing tres: {tres}")


def test_calculate_gpus_from_tres():
    notres = "billing=8,cpu=8,mem=64G,node=1"
    tres = "billing=8,cpu=8,mem=64G,node=1,gres/gpu=4"
    assert calculate_gpus_from_tres(notres) == 0
    assert calculate_gpus_from_tres(tres) == 4


def parse_elapsed_to_seconds(elapsed: str) -> int:
    """
    elapsed could be in two forms:
        1-00:00:00 -> days-hours:minutes:seconds
        00:00:00   -> hours:minutes:seconds
    """
    e_days = 0
    if "-" in elapsed:
        p = elapsed.split("-")
        e_days = int(p[0])
        elapsed = p[1]
    h, m, s = [int(p) for p in elapsed.split(":")]
    return e_days * 86400 + h * 60 * 60 + m * 60 + s


def test_parse_elapsed_to_seconds():
    assert parse_elapsed_to_seconds("00:00:01") == 1
    assert parse_elapsed_to_seconds("1-00:00:01") == 86401
    assert parse_elapsed_to_seconds("00:01:01") == 61
    assert parse_elapsed_to_seconds("11:11:11") == 40271


def calculate_compute_hours(number: int, elapsed: str) -> float:
    elapsed_seconds = parse_elapsed_to_seconds(elapsed)
    return int(number) * elapsed_seconds / 60 / 60


def test_calculate_compute_hours():
    assert calculate_compute_hours(1, "01:00:00") == 1.0
    assert calculate_compute_hours(1, "00:30:00") == 0.5
    assert calculate_compute_hours(1, "2-14:11:33") == 62.1925


def parse_line(line: str) -> dict:
    """
    29148459_925|ld_stats_array|akapoor|kernlab|kern|00:07:23|1|8|billing=8,cpu=8,mem=64G,node=1|2025-02-03T23:38:14|2025-02-03T23:53:21|n0335
    job_id|job_name|username|account|partition|elapsed|nodes|cpus|tres|submit_time|start_time|nodelist
    """
    job = {}
    p = [e.strip() for e in line.split("|") if e.strip()]
    job["job_id"] = p[0]
    job["job_name"] = p[1]
    job["username"] = p[2]
    job["account"] = p[3]
    job["partition"] = p[4]
    job["elapsed"] = p[5]
    job["nodes"] = int(p[6])
    job["cpus"] = int(p[7])
    job["tres"] = p[8]
    job["submit_time"] = p[9]
    job["start_time"] = p[10]
    job["nodelist"] = p[11]
    job["gpus"] = calculate_gpus_from_tres(job["tres"])
    job["wait_time_hours"] = calculate_wait_time_hours(
        job["submit_time"], job["start_time"]
    )
    job["cpu_hours"] = calculate_compute_hours(job["cpus"], job["elapsed"])
    job["gpu_hours"] = calculate_compute_hours(job["gpus"], job["elapsed"])
    return job


def test_parse_line():
    job1 = parse_line(
        "29148459_925|ld_stats_array|akapoor|kernlab|kern|00:07:23|1|8|billing=8,cpu=8,mem=64G,node=1|2025-02-03T23:38:14|2025-02-03T23:53:21|n0335"
    )
    assert job1["job_id"] == "29148459_925"
    assert job1["job_name"] == "ld_stats_array"
    assert job1["username"] == "akapoor"
    assert job1["account"] == "kernlab"
    assert job1["partition"] == "kern"
    assert job1["elapsed"] == "00:07:23"
    assert job1["nodes"] == 1
    assert job1["cpus"] == 8
    assert job1["tres"] == "billing=8,cpu=8,mem=64G,node=1"
    assert job1["submit_time"] == "2025-02-03T23:38:14"
    assert job1["start_time"] == "2025-02-03T23:53:21"
    assert job1["nodelist"] == "n0335"
    assert job1["gpus"] == 0
    assert job1["wait_time_hours"] == 0.25194444444444447
    assert job1["cpu_hours"] == 0.9844444444444445
    assert job1["gpu_hours"] == 0.0
    job2 = parse_line(
        "29148459_925|ld_stats_array|akapoor|kernlab|kern|00:07:23|1|8|billing=8,cpu=8,mem=64G,node=1,gres/gpu=4|2025-02-03T23:38:14|2025-02-03T23:53:21|n0335"
    )
    assert job2["job_id"] == "29148459_925"
    assert job2["job_name"] == "ld_stats_array"
    assert job2["username"] == "akapoor"
    assert job2["account"] == "kernlab"
    assert job2["partition"] == "kern"
    assert job2["elapsed"] == "00:07:23"
    assert job2["nodes"] == 1
    assert job2["cpus"] == 8
    assert job2["tres"] == "billing=8,cpu=8,mem=64G,node=1,gres/gpu=4"
    assert job2["submit_time"] == "2025-02-03T23:38:14"
    assert job2["start_time"] == "2025-02-03T23:53:21"
    assert job2["nodelist"] == "n0335"
    assert job2["gpus"] == 4
    assert job2["wait_time_hours"] == 0.25194444444444447
    assert job2["cpu_hours"] == 0.9844444444444445
    assert job2["gpu_hours"] == 0.4922222222222222


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=False, help="path to output file")
    parser.add_argument(
        "-n", "--noheader", required=False, help="don't show header row"
    )
    args = parser.parse_args()

    try:
        out = sys.stdout
        if args.output:
            out = open(args.output, "w")
        csvwriter = csv.writer(out)
    except:
        out.close()
        raise

    try:
        if not args.noheader:
            print(",".join(JOB_FIELDS), file=out)
        for line in sys.stdin:
            job = parse_line(line)
            csvwriter.writerow(job.values())
    except:
        out.close()
        raise


if __name__ == "__main__":
    main()
