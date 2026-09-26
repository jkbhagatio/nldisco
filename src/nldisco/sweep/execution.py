"""Local process and Submitit execution for trials and W&B agents."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from beartype import beartype
from dotenv import load_dotenv
from omegaconf import OmegaConf

from nldisco.sweep.search import apply_parameters, validate_config
from nldisco.sweep.training import run_trial


def _wandb_options(config):
    settings = config["wandb"]
    return {key: settings[key] for key in ("project", "entity", "group", "mode")}


def _wandb_agent(task):
    import wandb

    config = task["config"]
    results = []
    output = Path(task["output_dir"])
    output.mkdir(parents=True, exist_ok=False)

    def train():
        try:
            with wandb.init(**_wandb_options(config), dir=str(output)) as run:
                trial = apply_parameters(config, dict(run.config))
                trial["wandb"]["enabled"] = True
                validate_config(OmegaConf.create(trial))
                run.config.update({"nldisco": trial})
                result = run_trial(trial, str(output / run.id), task["device"])
                results.append(result)
        except Exception as error:
            results.append({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            raise

    wandb.agent(
        task["sweep_id"],
        function=train,
        count=task["count"],
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
    )
    result = {
        "status": "failed" if any(r["status"] == "failed" for r in results) else "completed",
        "runs": results,
        "requested_runs": task["count"],
    }
    (output / "agent.json").write_text(json.dumps(result, indent=2))
    return result


def _execute_task(task):
    config = task["config"]
    if config["env_file"]:
        load_dotenv(config["env_file"], override=False)
    try:
        if task["kind"] == "agent":
            return _wandb_agent(task)
        if config["wandb"]["enabled"]:
            import wandb

            # The parent already exists, but run_trial reserves the leaf directory.
            with wandb.init(
                **_wandb_options(config),
                config={"nldisco": config},
                dir=str(Path(task["output_dir"]).parent),
            ):
                return run_trial(config, task["output_dir"], task["device"])
        return run_trial(config, task["output_dir"], task["device"])
    except Exception as error:
        result = {
            "status": "failed",
            "output_dir": task["output_dir"],
            "error": f"{type(error).__name__}: {error}",
        }
        # Include failures that happened before run_trial could reserve its directory.
        Path(task["output_dir"] + ".failure.json").write_text(json.dumps(result, indent=2))
        return result


def _execute_lane(tasks):
    return [_execute_task(task) for task in tasks]


@beartype
def execute(
    tasks: list[dict[str, Any]], config: dict[str, Any], output_dir: str
) -> dict[str, Any]:
    """Run bounded local workers or submit a bounded Slurm array.

    Slurm workers use their allocation's logical device 0. Local GPU lanes keep
    a fixed device for their entire lifetime, preventing accidental GPU sharing.
    """
    output = Path(output_dir)
    if not tasks:
        raise ValueError("At least one task is required")
    settings = config["execution"]
    parallel = min(settings["max_parallel"], len(tasks))
    if settings["backend"] == "slurm":
        import submitit

        executor = submitit.AutoExecutor(folder=str(output / "slurm" / "%j"), cluster="slurm")
        slurm = settings["slurm"]
        parameters = {
            "timeout_min": slurm["timeout_min"],
            "cpus_per_task": slurm["cpus_per_task"],
            "mem_gb": slurm["mem_gb"],
            "gpus_per_node": slurm["gpus_per_node"],
            "tasks_per_node": 1,
            "nodes": 1,
            "slurm_array_parallelism": parallel,
        }
        parameters.update(
            {
                f"slurm_{key}": slurm[key]
                for key in ("partition", "account", "constraint")
                if slurm[key]
            }
        )
        executor.update_parameters(**parameters)
        for task in tasks:
            task["device"] = "cuda:0" if slurm["gpus_per_node"] else "cpu"
        jobs = executor.map_array(_execute_task, tasks)
        manifest = {"status": "submitted", "job_ids": [str(job.job_id) for job in jobs]}
        (output / "jobs.json").write_text(json.dumps(manifest, indent=2))
        if not settings["wait"]:
            return manifest
        results = []
        for job in jobs:
            try:
                results.append(job.result())
            except Exception as error:
                results.append(
                    {"status": "failed", "job_id": str(job.job_id), "error": str(error)}
                )
    else:
        lanes = [[] for _ in range(parallel)]
        for index, task in enumerate(tasks):
            lane = index % parallel
            task["device"] = settings["devices"][lane % len(settings["devices"])]
            lanes[lane].append(task)
        if parallel == 1:
            results = _execute_lane(lanes[0])
        else:
            with ProcessPoolExecutor(
                max_workers=parallel, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                futures = [pool.submit(_execute_lane, lane) for lane in lanes]
                results = []
                for future in futures:
                    try:
                        results.extend(future.result())
                    except Exception as error:
                        results.append({"status": "failed", "error": str(error)})
    summary = {
        "status": "failed" if any(r["status"] == "failed" for r in results) else "completed",
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    if summary["status"] == "failed":
        raise RuntimeError(f"One or more workers failed; see {output / 'summary.json'}")
    return summary
