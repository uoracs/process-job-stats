#!/usr/bin/env python3

import argparse
import csv
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pwd import getpwuid
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

YESTERDAY_DATE = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

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

SLURM_BIN_DIR = "/gpfs/t2/slurm/apps/current/bin"
GPFS_BIN_DIR = "/usr/lpp/mmfs/bin"


class RawJobData:
    def __init__(self):
        logger.debug("  Starting: Getting jobs from sacct")
        try:
            cmd = f"sacct -X -P -n --starttime='{YESTERDAY_DATE}T00:00:00' --endtime='{YESTERDAY_DATE}T23:59:59' --state=F,CD --format=JobID,JobName,User,Account,Partition,Elapsed,NNodes,NCPUS,AllocTRES,Submit,Start,End,Nodelist"
            self.jobs = [
                line.strip()
                for line in subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
                .stdout.decode()
                .split("\n")
                if line.strip()
            ]
        except Exception as e:
            print(f"Failed to get job data stdout from SLURM: {e}")
            exit(1)
        logger.debug("  Finished: Getting jobs from sacct")


class NodePartitions:
    def __init__(self):
        logger.debug("  Starting: Getting Node -> Partition associations")
        try:
            s = subprocess.run(
                f"{SLURM_BIN_DIR}/sinfo -h -o '%n,%P'",
                shell=True,
                stdout=subprocess.PIPE,
            )
            entries = [
                line.strip() for line in s.stdout.decode().split("\n") if line.strip()
            ]
            self.node_partitions = {
                item.split(",")[0]: item.split(",")[1] for item in entries
            }
        except Exception as e:
            print(f"Failed to parse node partition data: {e}")
            exit(1)
        logger.debug("  Finished: Getting Node -> Partition associations")

    def get_partition(self, node: str) -> str:
        try:
            return self.node_partitions[node]
        except Exception:
            return ""


class AccountStorages:
    def __init__(self):
        logger.debug("  Starting: Getting Account -> StorageGB")
        try:
            cmd = f"{GPFS_BIN_DIR}/mmrepquota -j fs1 --block-size g | awk '/FILESET/ {{print $1\",\"$4}}'"
            s = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
            entries = [
                line.strip() for line in s.stdout.decode().split("\n") if line.strip()
            ]
            self.account_storage = {
                item.split(",")[0]: int(item.split(",")[1]) for item in entries
            }
        except Exception as e:
            print(f"Failed to parse GPFS storage data: {e}")
            exit(1)
        logger.debug("  Finished: Getting Account -> StorageGB")

    def get_storage(self, account: str) -> int:
        try:
            return self.account_storage[account]
        except Exception:
            return 0


class AccountPIs:
    def __init__(self):
        logger.debug("  Starting: Getting Account -> PI associations")
        out = {}
        try:
            s = subprocess.run(
                "find /gpfs/projects/* -maxdepth 0", shell=True, stdout=subprocess.PIPE
            )
            proj_fdirs = [
                line.strip() for line in s.stdout.decode().split("\n") if line.strip()
            ]
            for d in proj_fdirs:
                account = d.split("/")[-1]
                pi = getpwuid(os.stat(d).st_uid).pw_name
                out[account] = pi
        except Exception as e:
            print(f"Failed to parse account PI data: {e}")
            exit(1)
        self.account_pi = out
        logger.debug("  Finished: Getting Account -> PI associations")

    def get_pi(self, account: str) -> str:
        try:
            return self.account_pi[account]
        except Exception:
            return ""


class JobCategory(Enum):
    OPEN = "open_use"
    DONATED = "donated"
    CONDO = "condo"


@dataclass
class Job:
    """Represents a complete job, both SLURM data and derived data"""

    # Information from SLURM
    job_id: str
    job_name: str
    username: str
    account: str
    partition: str
    elapsed: str
    nodes: str
    cpus: int
    tres: str
    submit_time: str
    start_time: str
    end_time: str
    nodelist: str

    # Generated Fields
    pi: str
    account_storage_gb: int
    category: JobCategory
    openuse_weight: float
    condo_weight: float
    gpus: int
    cpu_hours: float
    gpu_hours: float
    wait_time_hours: float
    run_time_hours: float
    date: str

    def dict(self) -> Dict[str, Any]:
        return {k: str(v) for k, v in asdict(self).items()}

    def keys(self) -> List[str]:
        return list(self.dict().keys())

    def values(self) -> List[str]:
        return list(self.dict().values())

    def __init__(
        self,
        slurm_job_line: str,
        node_partitions: NodePartitions,
        account_storages: AccountStorages,
        account_pis: AccountPIs,
    ):
        """
        29148459_925|ld_stats_array|akapoor|kernlab|kern|00:07:23|1|8|billing=8,cpu=8,mem=64G,node=1|2025-02-03T23:38:14|2025-02-03T23:53:21|n0335
        job_id|job_name|username|account|partition|elapsed|nodes|cpus|tres|submit_time|start_time|nodelist
        """
        if slurm_job_line.startswith("JobID"):
            return None
        p = [e.strip() for e in slurm_job_line.split("|") if e.strip()]
        self.job_id = p[0]
        self.job_name = p[1]
        self.username = p[2]
        self.account = p[3]
        self.partition = p[4]
        self.elapsed = p[5]
        if self.parse_elapsed_to_seconds(self.elapsed) == 0:
            return None
        self.nodes = p[6]
        self.cpus = int(p[7])
        self.tres = p[8]
        self.submit_time = p[9]
        self.start_time = p[10]
        self.end_time = p[11]
        self.nodelist = p[12]

        self.pi = account_pis.get_pi(self.account)
        self.account_storage_gb = account_storages.get_storage(self.account)
        self.category = self.categorize_job(self.partition)
        self.openuse_weight = self.calculate_weight(JobCategory.OPEN, node_partitions)
        self.condo_weight = self.calculate_weight(JobCategory.CONDO, node_partitions)
        self.gpus = self.calculate_gpus_from_tres(self.tres)
        self.wait_time_hours = self.calculate_wait_time_hours(
            self.submit_time, self.start_time
        )
        self.run_time_hours = self.calculate_run_time_hours(self.elapsed)
        self.cpu_hours = self.calculate_compute_hours(self.cpus, self.elapsed)
        self.gpu_hours = self.calculate_compute_hours(self.gpus, self.elapsed)
        self.date = self.get_day_date_from_iso(self.end_time)

    def calculate_wait_time_hours(self, iso1: str, iso2: str) -> float:
        dt1 = datetime.fromisoformat(iso1)
        dt2 = datetime.fromisoformat(iso2)
        delta = dt2 - dt1
        return abs(delta.total_seconds() / 60 / 60)

    def calculate_gpus_from_tres(self, tres: str) -> int:
        if "gres/gpu=" not in tres:
            return 0
        for part in tres.split(","):
            if "gres/gpu=" in part:
                return int(part.split("=")[1])
        raise ValueError(f"error parsing tres: {tres}")

    def parse_elapsed_to_seconds(self, elapsed: str) -> int:
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

    def get_day_date_from_iso(self, iso: str) -> str:
        return iso.split("T")[0]

    def calculate_compute_hours(self, number: int, elapsed: str) -> float:
        elapsed_seconds = self.parse_elapsed_to_seconds(elapsed)
        return float(number) * elapsed_seconds / 60 / 60

    def calculate_run_time_hours(self, elapsed: str) -> float:
        return self.parse_elapsed_to_seconds(elapsed) / 60 / 60

    def categorize_job(self, partition: str) -> JobCategory:
        if partition in OPEN_USE_PARTITIONS:
            return JobCategory.OPEN
        if partition == "preempt":
            return JobCategory.DONATED
        return JobCategory.CONDO

    def calculate_weight(
        self, category: JobCategory, node_partitions: NodePartitions
    ) -> float:
        """
        This represents the weight of the nodes that ran in open-use nodes.
        Used to multiply by things like CPU Hours, etc, to properly weight jobs.
        """
        if category not in [JobCategory.OPEN, JobCategory.CONDO]:
            raise ValueError("category must be OPEN or CONDO")
        c = 0
        nodes = [n.strip() for n in self.nodelist.split(",") if n.strip()]
        nl = len(nodes)
        for n in nodes:
            try:
                np = node_partitions.get_partition(n)
            except Exception:
                nl -= 1
                continue
            is_openuse = np in OPEN_USE_PARTITIONS
            if category == JobCategory.OPEN and is_openuse:
                c += 1
            if category == JobCategory.CONDO and not is_openuse:
                c += 1
        if nl <= 0:
            return 0.0
        return c / nl

    def to_sql_insert(self) -> str:
        """Returns a SQL INSERT statement for inserting this job into a database"""
        return ""


def write_job_to_sqlite(job: Job):
    statement = job.to_sql_insert()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=False, help="path to output file")
    parser.add_argument(
        "-n", "--noheader", required=False, help="don't show header row"
    )
    parser.add_argument(
        "-d", "--debug", required=False, action="store_true", help="show debug logging"
    )
    args = parser.parse_args()

    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)

    logger.debug("Started Processing")
    out = sys.stdout
    try:
        if args.output:
            out = open(args.output, "w")
    except:
        out.close()
        raise
    logger.debug(f"Output file: {out}")
    writer = csv.writer(out)

    nps = NodePartitions()
    ass = AccountStorages()
    aps = AccountPIs()

    try:
        for i, line in enumerate(sys.stdin):
            logger.debug(f"Processing job {i}")
            job = Job(line, node_partitions=nps, account_storages=ass, account_pis=aps)
            if not args.noheader and i == 0:
                print(",".join(job.keys()), file=out)
            if job:
                writer.writerow(job.values())
    except:
        out.close()
        raise


if __name__ == "__main__":
    main()
