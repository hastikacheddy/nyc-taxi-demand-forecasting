"""
CLI entrypoint executed inside an isolated container by the KubernetesPodOperator
(or DockerOperator):  python -m src.pipelines.cli <task> --as-of <iso8601>

Keeps the heavy compute out of the Airflow worker — Airflow only launches this.
"""
import argparse
import logging
import sys

from src.pipelines.tasks import run_task


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Taxi-demand pipeline task runner")
    parser.add_argument("task", help="task name (e.g. daily-inference, train-daily, monitor)")
    parser.add_argument("--as-of", default=None, help="execution window end (ISO-8601)")
    args = parser.parse_args(argv)

    result = run_task(args.task, as_of=args.as_of)
    logging.info("[cli] %s complete: %s", args.task, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
