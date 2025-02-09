#!/usr/bin/env python3

import argparse
from datetime import datetime
import sys
import csv
from enum import Enum

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
    "end_time",
    "nodelist",
    "category",
    "gpus",
    "wait_time_hours",
    "cpu_hours",
    "gpu_hours",
    "date",
]

OPEN_USE_PARTITIONS = [
    "compute",
    "compute_intel",
    "computelong",
    "computelong_intel",
    "gpu",
    "gpulong",
    "interactive",
    "interactivegpu",
    "memory",
    "memorylong",
]


class JobCategory(Enum):
    OPEN = "open_use"
    DONATED = "donated"
    CONDO = "condo"


def calculate_wait_time_hours(iso1: str, iso2: str) -> float:
    dt1 = datetime.fromisoformat(iso1)
    dt2 = datetime.fromisoformat(iso2)
    delta = dt2 - dt1
    return abs(delta.total_seconds() / 60 / 60)


def calculate_gpus_from_tres(tres: str) -> int:
    if "gres/gpu=" not in tres:
        return 0
    for part in tres.split(","):
        if "gres/gpu=" in part:
            return int(part.split("=")[1])
    raise ValueError(f"error parsing tres: {tres}")


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


def get_day_date_from_iso(iso: str) -> str:
    return iso.split("T")[0]


def calculate_compute_hours(number: int, elapsed: str) -> float:
    elapsed_seconds = parse_elapsed_to_seconds(elapsed)
    return int(number) * elapsed_seconds / 60 / 60


def categorize_job(partition: str) -> JobCategory:
    if partition in OPEN_USE_PARTITIONS:
        return JobCategory.OPEN
    if partition == "preempt":
        return JobCategory.DONATED
    return JobCategory.CONDO


def parse_line(line: str) -> dict | None:
    """
    29148459_925|ld_stats_array|akapoor|kernlab|kern|00:07:23|1|8|billing=8,cpu=8,mem=64G,node=1|2025-02-03T23:38:14|2025-02-03T23:53:21|n0335
    job_id|job_name|username|account|partition|elapsed|nodes|cpus|tres|submit_time|start_time|nodelist
    """
    if line.startswith("JobID"):
        return None
    job = {}
    p = [e.strip() for e in line.split("|") if e.strip()]
    job["job_id"] = p[0]
    job["job_name"] = p[1]
    job["username"] = p[2]
    job["account"] = p[3]
    job["partition"] = p[4]
    job["elapsed"] = p[5]
    if parse_elapsed_to_seconds(job["elapsed"]) == 0:
        return None
    job["nodes"] = p[6]
    job["cpus"] = p[7]
    job["tres"] = p[8]
    job["submit_time"] = p[9]
    job["start_time"] = p[10]
    job["end_time"] = p[11]
    job["nodelist"] = p[12]
    job["category"] = categorize_job(job["partition"]).value
    job["gpus"] = str(calculate_gpus_from_tres(job["tres"]))
    job["wait_time_hours"] = str(
        calculate_wait_time_hours(job["submit_time"], job["start_time"])
    )
    job["cpu_hours"] = str(calculate_compute_hours(int(job["cpus"]), job["elapsed"]))
    job["gpu_hours"] = str(calculate_compute_hours(int(job["gpus"]), job["elapsed"]))
    job["date"] = get_day_date_from_iso(job["end_time"])
    return job


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
            if job:
                csvwriter.writerow(job.values())
    except:
        out.close()
        raise


if __name__ == "__main__":
    main()
