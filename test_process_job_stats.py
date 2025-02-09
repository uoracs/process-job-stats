from process_job_stats import *


def test_calculate_wait_time_hours():
    assert (
        calculate_wait_time_hours("2025-02-03T23:38:00", "2025-02-03T23:53:00") == 0.25
    )


def test_calculate_gpus_from_tres():
    notres = "billing=8,cpu=8,mem=64G,node=1"
    tres = "billing=8,cpu=8,mem=64G,node=1,gres/gpu=4"
    assert calculate_gpus_from_tres(notres) == 0
    assert calculate_gpus_from_tres(tres) == 4


def test_parse_elapsed_to_seconds():
    assert parse_elapsed_to_seconds("00:00:01") == 1
    assert parse_elapsed_to_seconds("1-00:00:01") == 86401
    assert parse_elapsed_to_seconds("00:01:01") == 61
    assert parse_elapsed_to_seconds("11:11:11") == 40271


def test_calculate_compute_hours():
    assert calculate_compute_hours(1, "01:00:00") == 1.0
    assert calculate_compute_hours(1, "00:30:00") == 0.5
    assert calculate_compute_hours(1, "2-14:11:33") == 62.1925


def test_parse_line():
    job1 = parse_line(
        "29148459_925|ld_stats_array|akapoor|kernlab|kern|00:07:23|1|8|billing=8,cpu=8,mem=64G,node=1|2025-02-03T23:38:14|2025-02-03T23:53:21|2025-02-03T23:53:21|n0335"
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
    assert job1["end_time"] == "2025-02-03T23:53:21"
    assert job1["nodelist"] == "n0335"
    assert job1["gpus"] == 0
    assert job1["wait_time_hours"] == 0.25194444444444447
    assert job1["cpu_hours"] == 0.9844444444444445
    assert job1["gpu_hours"] == 0.0
    assert job1["date"] == "2025-02-03"
    job2 = parse_line(
        "29148459_925|ld_stats_array|akapoor|kernlab|kern|00:07:23|1|8|billing=8,cpu=8,mem=64G,node=1,gres/gpu=4|2025-02-03T23:38:14|2025-02-03T23:53:21|2025-02-03T23:53:21|n0335"
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
    assert job1["end_time"] == "2025-02-03T23:53:21"
    assert job2["nodelist"] == "n0335"
    assert job2["gpus"] == 4
    assert job2["wait_time_hours"] == 0.25194444444444447
    assert job2["cpu_hours"] == 0.9844444444444445
    assert job2["gpu_hours"] == 0.4922222222222222
    assert job2["date"] == "2025-02-03"
