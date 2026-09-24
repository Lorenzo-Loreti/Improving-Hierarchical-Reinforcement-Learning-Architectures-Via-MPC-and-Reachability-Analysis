"""A local, append-only copy of everything a training loop logs to W&B.

Without it a run's learning curve existed only on W&B (when --track was on)
or as free text on stdout, where the evaluation and solved-check lines do not
even carry the global step they belong to. Every flat-PPO and hPPO run now
writes, next to its checkpoints:

- `config.json`: the run's parsed arguments, written once at start-up;
- `metrics.jsonl`: one JSON object per `log()` call, i.e. per W&B
  `wandb.log` call, `{"global_step": ..., <the same keys W&B gets>}`.

Writing a file touches no random number generator, so a run with this log is
bit-identical to one without it. The scripts that read these back to plot
learning curves are `algorithms/study.py` and its per-scenario wrappers
(`scenarios/*/scripts/study_*.py`).

Imported by bare name, like `common`, once algorithms/ is on sys.path.
"""

import json
import os
import numbers


CONFIG_FILE = "config.json"
METRICS_FILE = "metrics.jsonl"


class MetricsLog:
    """Appends one JSON line per call; flushed every time, so an interrupted
    run keeps everything it logged up to the interruption."""

    def __init__(self, run_dir, config=None):
        os.makedirs(run_dir, exist_ok=True)
        if config is not None:
            with open(os.path.join(run_dir, CONFIG_FILE), "w") as f:
                json.dump(config, f, indent=2, default=str)
        self._file = open(os.path.join(run_dir, METRICS_FILE), "a")

    def log(self, metrics, step):
        # numpy scalars (np.float32 in particular) are not JSON-serialisable;
        # every value the loops log is a number, so cast them all.
        record = {"global_step": int(step)}
        for key, value in metrics.items():
            record[key] = float(value) if isinstance(value, numbers.Number) else value
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def read_config(run_dir):
    with open(os.path.join(run_dir, CONFIG_FILE)) as f:
        return json.load(f)


def read_metrics(run_dir):
    """Every record in `run_dir`'s metrics.jsonl, in logging order."""
    with open(os.path.join(run_dir, METRICS_FILE)) as f:
        return [json.loads(line) for line in f if line.strip()]


def series(records, key):
    """(global_steps, values) of every record that carries `key`."""
    pairs = [(r["global_step"], r[key]) for r in records if key in r]
    return [p[0] for p in pairs], [p[1] for p in pairs]
