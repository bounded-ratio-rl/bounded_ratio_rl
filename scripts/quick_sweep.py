import argparse
import re
import subprocess


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, text=True, capture_output=True)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{out}")
    return out


def create_sweep(yaml_path: str, project: str, entity: str | None) -> str:
    cmd = ["wandb", "sweep", yaml_path, "-p", project]
    if entity:
        cmd += ["-e", entity]

    out = run(cmd)

    # W&B prints something like:
    # "wandb: Run sweep agent with: wandb agent entity/project/sweepid"
    m = re.search(r"wandb agent\s+([^\s]+)", out)
    if m:
        return m.group(1)

    # Fallback: sometimes it prints "Create sweep with ID: <id>"
    m = re.search(r"Create sweep with ID:\s*([A-Za-z0-9]+)", out)
    if m:
        sweep_id = m.group(1)
        if not entity:
            raise RuntimeError(
                "Parsed sweep id but entity was not provided; can't form entity/project/sweepid reliably.\n"
                "Provide --entity."
            )
        return f"{entity}/{project}/{sweep_id}"

    raise RuntimeError(f"Could not parse sweep path from wandb output:\n\n{out}")


def main():
    ap = argparse.ArgumentParser(description="Create a W&B sweep and immediately run an agent.")
    ap.add_argument("sweep_yaml", help="Path to sweep yaml (e.g. sweeps/atari/ppo_seed.yaml)")
    ap.add_argument("-p", "--project", required=True, help="W&B project (e.g. BPO)")
    ap.add_argument("-e", "--entity", default=None, help="W&B entity (user or team), e.g. {your-entity-name}")
    ap.add_argument("--count", type=int, default=None, help="Runs for this agent (optional)")
    args = ap.parse_args()

    sweep_path = create_sweep(args.sweep_yaml, args.project, args.entity)
    print(f"[quick_submit] sweep: {sweep_path}", flush=True)

    agent_cmd = ["wandb", "agent", sweep_path]
    if args.count is not None:
        agent_cmd[1:1] = ["agent", "--count", str(args.count)]
        # Equivalent: ["wandb","agent","--count",...,"entity/project/sweepid"]

    # Run agent attached to this job (streams logs to Slurm output)
    subprocess.run(agent_cmd, check=True)


if __name__ == "__main__":
    main()

