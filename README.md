# process-job-stats

Python script to take specific SLURM job parsable output, derive statistics, and output lines with extra information to be used with something like `pandas`.

## Usage

This script assumes you're using the SLURM format string:

`JobID|JobName|User|Account|Partition|Elapsed|NNodes|NCPUS|AllocTRES|Submit|Start|End|NodeList`

and you're using the `parsable2` flag, `-P`, omitting job steps, `-X`, and not printing the header `-n`.

You also probably want to report on Failed (`F`) and Completed (`CD`) states.

Simply pipe the `sacct` command to this script:

```bash
sacct -n -X -P \
  --starttime='2025-02-04T00:00:00' \
  --endtime='2025-02-04T23:59:59' \
  --state=F,CD \
  --format=JobID,JobName,User,Account,Partition,Elapsed,NNodes,NCPUS,AllocTRES,Submit,Start,Nodelist \
  | process-job-stats.py -o processed.csv
```
