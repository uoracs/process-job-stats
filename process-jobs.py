#!/usr/bin/env python3

from datetime import datetime


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


def calculate_cpu_hours(cpus: int, elapsed: str) -> float:
    elapsed_seconds = parse_elapsed_to_seconds(elapsed)
    print(f"elapsed seconds: {elapsed_seconds}")
    return cpus * elapsed_seconds / 60 / 60


def test_calculate_cpu_hours():
    assert calculate_cpu_hours(1, "01:00:00") == 1.0
    assert calculate_cpu_hours(1, "00:30:00") == 0.5
    assert calculate_cpu_hours(1, "1-11:22:19") == 35.3725


def parse_line(line: str) -> dict:
    """
    29148459_925|ld_stats_array|akapoor|kernlab|kern|00:07:23|1|8|billing=8,cpu=8,mem=64G,node=1|2025-02-03T23:38:14|2025-02-03T23:53:21|n0335
    job_id|job_name|username|account|partition|elapsed|nodes|cpus|tres|submit_time|start_time|nodelist
    """
    job = {}
    p = line.split("|")
    job["job_id"] = p[0]
    job["job_name"] = p[1]
    job["username"] = p[2]
    job["account"] = p[3]
    job["partition"] = p[4]
    job["elapsed"] = p[5]
    job["nodes"] = p[6]
    job["cpus"] = p[7]
    job["tres"] = p[8]
    job["submit_time"] = p[9]
    job["start_time"] = p[10]
    job["nodelist"] = p[11]
    job["gpus"] = calculate_gpus_from_tres(job["tres"])
    job["wait_time_hours"] = calculate_wait_time_hours(
        job["submit_time"], job["start_time"]
    )
    return job


def test_parse_line():
    job = parse_line(
        "29148459_925|ld_stats_array|akapoor|kernlab|kern|00:07:23|1|8|billing=8,cpu=8,mem=64G,node=1|2025-02-03T23:38:14|2025-02-03T23:53:21|n0335"
    )
    assert job["job_id"] == "29148459_925"
    assert job["job_name"] == "ld_stats_array"
    assert job["username"] == "akapoor"
    assert job["account"] == "kernlab"
    assert job["partition"] == "kern"
    assert job["elapsed"] == "00:07:23"
    assert job["nodes"] == "1"
    assert job["cpus"] == "8"
    assert job["tres"] == "billing=8,cpu=8,mem=64G,node=1"
    assert job["submit_time"] == "2025-02-03T23:38:14"
    assert job["start_time"] == "2025-02-03T23:53:21"
    assert job["nodelist"] == "n0335"


def main():
    print("hi")
    print(parse_elapsed_to_seconds("01:00:00"))


if __name__ == "__main__":
    main()
