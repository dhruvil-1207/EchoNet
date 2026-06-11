import csv
import os
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(
    LOG_DIR,
    "performance_log.csv"
)

os.makedirs(LOG_DIR, exist_ok=True)


def log_performance(
    function_name,
    duration_seconds,
    **metrics
):
    file_exists = os.path.exists(LOG_FILE)

    row = {
        "timestamp": str(datetime.now()),
        "function": function_name,
        "duration_seconds": round(
            duration_seconds,
            4
        )
    }

    row.update(metrics)

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)