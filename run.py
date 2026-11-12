import argparse
import json
from pathlib import Path
import subprocess
import sys

from genevolve_repro.common import ROOT, WORLD_SIZE, required_env
from genevolve_repro.sft_export import export_files, sft_manifest


def distributed(subcommand, work, *arguments):
    subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}", str(ROOT / "run.py"),
        "--work", str(work), subcommand, *arguments,
    ], cwd=ROOT, check=True)


def evaluation_arguments(args):
    result = ["--split", args.split]
    if args.policy is not None:
        result.extend(["--policy", str(args.policy.resolve())])
    if args.max_cases is not None:
        result.extend(["--max-cases", str(args.max_cases)])
    return result


def eval_flags(parser):
    parser.add_argument("--split", choices=["bench", "rl-val", "wise"], default="bench")
    parser.add_argument("--policy", type=Path, help="HF export (default: WORK/grpo/final)")
    parser.add_argument("--max-cases", type=int, help="Limit a live evaluation for a pilot")


def execute(args, work):
    if args.command == "grpo":
        export_files(args.sft_checkpoint)
        if args.steps is not None and args.steps < 1:
            raise ValueError("--steps must be positive.")
    if args.command in {"grpo", "eval"}:
        keys = ("WISE_API_KEY",) if getattr(args, "split", None) == "wise" else ("JUDGE_API_KEY",)
        for key in ("SERPER_API_KEY", *keys):
            required_env(key)
    if args.command == "prepare":
        from genevolve_repro.data import prepare
        prepare(work)
    if args.command == "grpo":
        flags = ["--sft-checkpoint", str(args.sft_checkpoint.resolve())]
        if args.steps is not None:
            flags.extend(["--steps", str(args.steps)])
        distributed("_grpo", work, *flags)
    if args.command == "eval":
        distributed("_eval", work, *evaluation_arguments(args))


def sft_flag(parser):
    parser.add_argument("--sft-checkpoint", required=True, type=Path,
                        help="existing Qwen3-VL-2B SFT Hugging Face export")


def main():
    parser = argparse.ArgumentParser(description="GenEvolve 2B stage 2: GRPO only on four GPUs")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="download pinned RL and benchmark splits")
    sft_flag(commands.add_parser("check-sft", help="check and fingerprint an existing SFT export"))
    for name in ("grpo", "_grpo"):
        stage = commands.add_parser(name)
        sft_flag(stage)
        stage.add_argument("--steps", type=int, help="RL updates (default: one pass over the train split)")
    for name in ("eval", "_eval"):
        eval_flags(commands.add_parser(name))
    args = parser.parse_args()
    work = args.work.resolve()
    if args.command == "check-sft":
        try:
            print(json.dumps(sft_manifest(args.sft_checkpoint.resolve()), indent=2))
        except ValueError as error:
            parser.error(str(error))
    elif args.command == "_grpo":
        from genevolve_repro.evolve import worker
        worker(work, args.sft_checkpoint.resolve(), args.steps)
    elif args.command == "_eval":
        from genevolve_repro.evaluate import worker
        policy = args.policy.resolve() if args.policy else work / "grpo" / "final"
        worker(work, policy, args.split, args.max_cases)
    else:
        from filelock import FileLock, Timeout
        work.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(work / ".run.lock", timeout=0):
                execute(args, work)
        except Timeout as error:
            raise SystemExit(f"Another run is using {work}.") from error
        except ValueError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
