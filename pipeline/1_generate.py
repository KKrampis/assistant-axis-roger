#!/usr/bin/env python3
"""
Generate model responses using vLLM batch inference.

Supports two modes:
- roger (default): Combined role+trait instructions, standalone traits/roles, and default.
  Uses goal_roles_and_traits.json to select which roles/traits to combine.
  Flags --no_traits, --traits_only, --roles_only control which subset to generate.
  Flags --combos_only_r / --combos_only_t restrict to one half of the
  combination grid (r_ = goal-role x non-goal-trait; t_ = non-goal-role x
  goal-trait).  Both write into the same combinations/ directory.
- christina: Standalone roles (or traits) from --roles_dir. Run separately per
  entity type to avoid name collisions.

Supports automatic multi-worker parallelization when total GPUs > tensor_parallel_size.

Usage:
    # Roger mode (combined + standalone traits + default)
    uv run 1_generate.py --mode roger --model Qwen/Qwen3-32B \\
        --output_dir outputs/roger/responses \\
        --goal_count 30 --non_goal_count 30

    # Christina mode (roles only -- rerun with --roles_dir for traits)
    uv run 1_generate.py --mode christina --model Qwen/Qwen3-32B \\
        --output_dir outputs/roles/responses

    # With explicit tensor parallelism
    uv run 1_generate.py --model Qwen/Qwen3-32B --tensor_parallel_size 2 ...
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.generation import RoleResponseGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Work-item collection
# ---------------------------------------------------------------------------

def collect_work_items(args) -> List[Dict]:
    """Build work items based on mode.

    Returns list of dicts.  Standalone items have keys:
        output_name, type="standalone", file_path
    Combined items have keys:
        output_name, type="combined", role_file, trait_file,
        role_name, trait_name, goal_source
    """
    items: List[Dict] = []
    roles_dir = Path(args.roles_dir)

    if args.mode == "roger":
        traits_dir = Path(args.traits_dir)

        if args.roles_only:
            # Standalone roles (all files in roles_dir, filtered by --roles if given)
            for fp in sorted(roles_dir.glob("*.json")):
                if args.roles and fp.stem not in args.roles:
                    continue
                items.append({
                    "output_name": fp.stem,
                    "type": "standalone",
                    "file_path": str(fp),
                })
            logger.info(f"Roger mode (roles_only): {len(items)} standalone roles")

        else:
            if not args.traits_only:
                with open(args.goal_file) as f:
                    goal_data = json.load(f)

                goal_roles = goal_data["roles"]["goal"]
                non_goal_roles = goal_data["roles"]["non_goal"]
                goal_traits = goal_data["traits"]["goal"]
                non_goal_traits = goal_data["traits"]["non_goal"]

                gc, ngc = args.goal_count, args.non_goal_count
                if gc > len(goal_roles) and gc > len(goal_traits):
                    logger.error(
                        f"--goal_count {gc} exceeds both roles.goal "
                        f"({len(goal_roles)}) and traits.goal ({len(goal_traits)})")
                    sys.exit(1)
                if ngc > len(non_goal_roles) and ngc > len(non_goal_traits):
                    logger.error(
                        f"--non_goal_count {ngc} exceeds both roles.non_goal "
                        f"({len(non_goal_roles)}) and traits.non_goal ({len(non_goal_traits)})")
                    sys.exit(1)

                use_goal_roles = goal_roles[:min(gc, len(goal_roles))]
                use_non_goal_roles = non_goal_roles[:min(ngc, len(non_goal_roles))]
                use_goal_traits = goal_traits[:min(gc, len(goal_traits))]
                use_non_goal_traits = non_goal_traits[:min(ngc, len(non_goal_traits))]

                # --combos_only_r / --combos_only_t restrict to one half
                # of the combination grid (used by run_pipeline.sh's
                # r_combinations / t_combinations subset types).  Both halves
                # write into the same combinations/ directory, so the file
                # layout downstream tooling expects is preserved.
                emit_r = not args.combos_only_t
                emit_t = not args.combos_only_r

                # r_ combos: goal role x non-goal trait  (goal from role)
                for role_name in (use_goal_roles if emit_r else []):
                    role_file = roles_dir / f"{role_name}.json"
                    if not role_file.exists():
                        logger.warning(f"Role file missing: {role_file}")
                        continue
                    for trait_name in use_non_goal_traits:
                        trait_file = traits_dir / f"{trait_name}.json"
                        if not trait_file.exists():
                            logger.warning(f"Trait file missing: {trait_file}")
                            continue
                        items.append({
                            "output_name": f"r_{role_name}__{trait_name}",
                            "type": "combined",
                            "role_name": role_name,
                            "trait_name": trait_name,
                            "role_file": str(role_file),
                            "trait_file": str(trait_file),
                            "goal_source": "role",
                        })

                # t_ combos: non-goal role x goal trait  (goal from trait)
                for role_name in (use_non_goal_roles if emit_t else []):
                    role_file = roles_dir / f"{role_name}.json"
                    if not role_file.exists():
                        logger.warning(f"Role file missing: {role_file}")
                        continue
                    for trait_name in use_goal_traits:
                        trait_file = traits_dir / f"{trait_name}.json"
                        if not trait_file.exists():
                            logger.warning(f"Trait file missing: {trait_file}")
                            continue
                        items.append({
                            "output_name": f"t_{role_name}__{trait_name}",
                            "type": "combined",
                            "role_name": role_name,
                            "trait_name": trait_name,
                            "role_file": str(role_file),
                            "trait_file": str(trait_file),
                            "goal_source": "trait",
                        })

                # Default role (always needed for axis computation)
                default_file = roles_dir / "default.json"
                if default_file.exists():
                    items.append({
                        "output_name": "default",
                        "type": "standalone",
                        "file_path": str(default_file),
                    })
                else:
                    logger.warning(f"Default role not found: {default_file}")

            if not args.no_traits:
                # Standalone traits (all files in traits_dir)
                for fp in sorted(traits_dir.glob("*.json")):
                    items.append({
                        "output_name": fp.stem,
                        "type": "standalone",
                        "file_path": str(fp),
                    })

            n_combined = sum(1 for i in items if i["type"] == "combined")
            n_standalone = sum(1 for i in items if i["type"] == "standalone")
            logger.info(f"Roger mode: {n_combined} combined + {n_standalone} standalone "
                        f"= {len(items)} total work items")

    else:  # christina — scan roles_dir (unchanged behaviour)
        for fp in sorted(roles_dir.glob("*.json")):
            name = fp.stem
            if args.roles and name not in args.roles:
                continue
            items.append({
                "output_name": name,
                "type": "standalone",
                "file_path": str(fp),
            })
        logger.info(f"Christina mode: {len(items)} items from {roles_dir}")

    return items


# ---------------------------------------------------------------------------
# Worker / multi-worker
# ---------------------------------------------------------------------------

def _process_item(generator: RoleResponseGenerator, item: Dict, worker_logger):
    """Process a single work item, returning True on success."""
    output_name = item["output_name"]
    try:
        if item["type"] == "standalone":
            data = generator.load_role(Path(item["file_path"]))
            if "instruction" not in data:
                worker_logger.warning(f"Skipping {output_name}: missing 'instruction'")
                return False
            responses = generator.generate_role_responses(output_name, data)
        elif item["type"] == "combined":
            role_data = generator.load_role(Path(item["role_file"]))
            trait_data = generator.load_role(Path(item["trait_file"]))
            responses = generator.generate_combined_responses(
                role_data, trait_data,
                item["role_name"], item["trait_name"],
                item["goal_source"],
            )
        else:
            worker_logger.warning(f"Unknown item type: {item['type']}")
            return False

        if responses:
            generator.save_responses(output_name, responses)
            return True
        worker_logger.warning(f"No responses for '{output_name}'")
        return False

    except Exception as e:
        worker_logger.error(f"Error processing {output_name}: {e}")
        return False


def process_items_on_worker(
    worker_id: int, gpu_ids: List[int], work_items: List[Dict], args,
):
    """Process work items on a single GPU worker."""
    gpu_ids_str = ",".join(map(str, gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    wlog = logging.getLogger(f"Worker-{worker_id}")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        f"%(asctime)s - Worker-{worker_id}[GPUs:{gpu_ids_str}] - "
        f"%(levelname)s - %(message)s"))
    wlog.addHandler(handler)
    wlog.setLevel(logging.INFO)

    wlog.info(f"Starting with GPUs {gpu_ids}, {len(work_items)} items")

    try:
        generator = RoleResponseGenerator(
            model_name=args.model,
            roles_dir=args.roles_dir,
            output_dir=args.output_dir,
            questions_file=args.questions_file,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            question_count=args.question_count,
            reduce_questions=args.reduce_questions,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
        generator.generator.load()

        completed = failed = 0
        from tqdm import tqdm
        for item in tqdm(work_items, desc=f"Worker-{worker_id}", position=worker_id):
            ok = _process_item(generator, item, wlog)
            if ok:
                completed += 1
            else:
                failed += 1

        wlog.info(f"Done: {completed} successful, {failed} failed")

    except Exception as e:
        wlog.error(f"Fatal error: {e}")
    finally:
        wlog.info("Cleanup completed")


def run_multi_worker(work_items: List[Dict], args) -> int:
    """Distribute work items across multiple GPU workers."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = [int(x.strip())
                   for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                   if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    total_gpus = len(gpu_ids)
    if total_gpus == 0:
        logger.error("No GPUs available.")
        return 1

    tp = args.tensor_parallel_size
    if tp > total_gpus:
        logger.error(f"tensor_parallel_size ({tp}) > available GPUs ({total_gpus})")
        return 1

    num_workers = total_gpus // tp
    if total_gpus % tp != 0:
        logger.warning(
            f"GPUs ({total_gpus}) not divisible by TP ({tp}). "
            f"{num_workers} workers, {total_gpus % tp} GPU(s) unused.")

    logger.info(f"GPUs: {gpu_ids}, TP: {tp}, Workers: {num_workers}")

    gpu_chunks = [gpu_ids[i * tp:(i + 1) * tp] for i in range(num_workers)]

    # Round-robin distribution
    item_chunks: List[List[Dict]] = [[] for _ in range(num_workers)]
    for idx, item in enumerate(work_items):
        item_chunks[idx % num_workers].append(item)

    for i in range(num_workers):
        logger.info(f"Worker {i} (GPUs {gpu_chunks[i]}): {len(item_chunks[i])} items")

    mp.set_start_method("spawn", force=True)

    processes = []
    for wid in range(num_workers):
        if item_chunks[wid]:
            p = mp.Process(
                target=process_items_on_worker,
                args=(wid, gpu_chunks[wid], item_chunks[wid], args),
            )
            p.start()
            processes.append(p)

    logger.info(f"Launched {len(processes)} worker processes")
    for p in processes:
        p.join()

    logger.info("Multi-worker processing completed!")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate responses using vLLM batch inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    parser.add_argument("--mode", type=str, default="roger",
                        choices=["roger", "christina"],
                        help="Pipeline mode (default: roger)")

    # Roger-mode parameters
    parser.add_argument("--goal_count", type=int, default=30,
                        help="Top-N goal items per list (roger mode)")
    parser.add_argument("--non_goal_count", type=int, default=30,
                        help="Top-N non-goal items per list (roger mode)")
    parser.add_argument("--traits_dir", type=str,
                        default="../data/traits/instructions",
                        help="Trait instruction JSON directory")
    parser.add_argument("--goal_file", type=str,
                        default="../data/goal_roles_and_traits.json",
                        help="Goal roles/traits JSON (roger mode)")
    parser.add_argument("--no_traits", action="store_true",
                        help="Roger mode: skip standalone traits (combos + default only)")
    parser.add_argument("--traits_only", action="store_true",
                        help="Roger mode: standalone traits only (skip combos + default)")
    parser.add_argument("--roles_only", action="store_true",
                        help="Roger mode: standalone roles only (skip combos + traits)")
    parser.add_argument("--combos_only_r", action="store_true",
                        help="Roger mode: only r_ combos (goal role x non-goal "
                             "trait); written to the same dir as full combos")
    parser.add_argument("--combos_only_t", action="store_true",
                        help="Roger mode: only t_ combos (non-goal role x goal "
                             "trait); written to the same dir as full combos")

    # Shared parameters
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--roles_dir", type=str,
                        default="../data/roles/instructions",
                        help="Role instruction JSON directory")
    parser.add_argument("--questions_file", type=str,
                        default="../data/extraction_questions.jsonl",
                        help="Path to questions JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for JSONL files")
    parser.add_argument("--max_model_len", type=int, default=2048,
                        help="Maximum model context length")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="GPUs per worker (auto-detect if None)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95,
                        help="GPU memory utilization")
    parser.add_argument("--question_count", type=int, default=None,
                        help="Number of questions per entity (default: 300 roger, 240 christina)")
    parser.add_argument("--reduce_questions", type=int, default=None,
                        help="Take every Nth question (default: 3 roger, 1 christina)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--roles", nargs="+",
                        help="Specific roles to process (christina mode)")

    args = parser.parse_args()

    exclusive_flags = sum([args.no_traits, args.traits_only, args.roles_only])
    if exclusive_flags > 1:
        parser.error("--no_traits, --traits_only, and --roles_only are mutually exclusive")
    if args.combos_only_r and args.combos_only_t:
        parser.error("--combos_only_r and --combos_only_t are mutually exclusive")
    if (args.combos_only_r or args.combos_only_t) and (
            args.traits_only or args.roles_only):
        parser.error("--combos_only_r / --combos_only_t cannot be combined with "
                     "--traits_only or --roles_only")
    roger_flags = (args.no_traits or args.traits_only or args.roles_only
                   or args.combos_only_r or args.combos_only_t)
    if args.mode != "roger" and roger_flags:
        parser.error("--no_traits / --traits_only / --roles_only / "
                     "--combos_only_r / --combos_only_t are only valid in roger mode")

    if args.question_count is None:
        args.question_count = 300 if args.mode == "roger" else 240
    if args.reduce_questions is None:
        args.reduce_questions = 3 if args.mode == "roger" else 1

    # Collect and filter work items
    all_items = collect_work_items(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for item in all_items:
        if (output_dir / f"{item['output_name']}.jsonl").exists():
            logger.info(f"Skipping '{item['output_name']}' (already exists)")
        else:
            pending.append(item)

    if not pending:
        logger.info("Nothing to process — all items already exist")
        return

    logger.info(f"{len(pending)} items to process "
                f"({len(all_items) - len(pending)} skipped)")

    # GPU detection
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        total_gpus = len([x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                          if x.strip()])
    else:
        total_gpus = torch.cuda.device_count()

    tp = args.tensor_parallel_size if args.tensor_parallel_size else total_gpus
    use_multi = total_gpus > 1 and tp > 0 and total_gpus > tp

    if use_multi:
        logger.info(f"Multi-worker: {total_gpus} GPUs, TP={tp}")
        args.tensor_parallel_size = tp
        rc = run_multi_worker(pending, args)
        if rc != 0:
            sys.exit(rc)
    else:
        logger.info(f"Single-worker: {tp} GPU(s)")

        generator = RoleResponseGenerator(
            model_name=args.model,
            roles_dir=args.roles_dir,
            output_dir=args.output_dir,
            questions_file=args.questions_file,
            max_model_len=args.max_model_len,
            tensor_parallel_size=tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
            question_count=args.question_count,
            reduce_questions=args.reduce_questions,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
        generator.generator.load()

        from tqdm import tqdm
        for item in tqdm(pending, desc="Processing"):
            _process_item(generator, item, logger)

    logger.info("Done!")


if __name__ == "__main__":
    main()
