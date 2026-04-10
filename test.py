import argparse
import os, sys
import pickle
import sys
import time
import warnings
from collections import OrderedDict
from typing import Union
import torch
import torch.serialization
import torch.nn as nn

from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import gather_by_index, unbatchify
from tqdm.auto import tqdm

from utils import get_dataloader
from envs import MTVRPEnv
from models import RouteFinderBase
from models import RouteFinderPolicy, LoRAPolicy, MultiLoRAPolicy, CadaPolicy, CadaLoRAPolicy, CadaMultiLoRAPolicy
import torchrl.data as torchrl_data
import torchrl.data.tensor_specs as torchrl_tensor_specs

CompositeSpec = getattr(torchrl_data, "CompositeSpec", getattr(torchrl_data, "Composite"))
# Backward compatibility for checkpoints referencing old torchrl symbol path.
if not hasattr(torchrl_tensor_specs, "CompositeSpec") and hasattr(
    torchrl_tensor_specs, "Composite"
):
    setattr(torchrl_tensor_specs, "CompositeSpec", torchrl_tensor_specs.Composite)

# Required for loading old checkpoints under PyTorch 2.6+ weights-only behavior.
torch.serialization.add_safe_globals([CompositeSpec])


# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")


class Logger(object):
    def __init__(self, file_name, stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, 'ab', buffering=0)
        self.log_disable = False
    def write(self, message):
        self.terminal.write(str(message))
        if not self.log_disable:
            self.log.write(str(message).encode("utf-8"))
    def flush(self):
        self.terminal.flush()
        if not self.log_disable:
            self.log.flush()
    def close(self):
        self.log.close()
    def disable_log(self):
        self.log_disable = True
        self.close()



def test(
        policy,
        td,
        env,
        num_augment=8,
        augment_fn="dihedral8",  # or symmetric. Default is dihedral8 for reported eval
        num_starts=None,
    decode_type="multistart_greedy",
    temperature=1.0,
        device: Union[str, torch.device] = "cuda",
):
    costs_bks = td.get("costs_bks", None)

    with torch.inference_mode():
        with (
            torch.autocast("cuda")
            if "cuda" in str(device)
            else torch.inference_mode()
        ):  # Use mixed precision if supported
            n_start = env.get_num_starts(td) if num_starts is None else num_starts

            if num_augment > 1:
                td = StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)(td)

            # Evaluate policy
            out = policy(
                td,
                env,
                phase="test",
                num_starts=n_start,
                decode_type=decode_type,
                temperature=temperature,
                return_actions=True,
            )

            # Unbatchify reward to [batch_size, num_augment, num_starts].
            reward = unbatchify(out["reward"], (num_augment, n_start))

            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (num_augment, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if num_augment > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                # If costs_bks is available, we calculate the gap to BKS
                if costs_bks is not None:
                    # note: torch.abs is here as a temporary fix, since we forgot to
                    # convert rewards to costs. Does not affect the results.
                    gap_to_bks = (
                            100
                            * (-max_aug_reward - torch.abs(costs_bks))
                            / torch.abs(costs_bks)
                    )
                    out.update({"gap_to_bks": gap_to_bks})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            if out.get("gap_to_bks", None) is None:
                out.update({"gap_to_bks": 69420})  # Dummy value

            return out



def load_model_weights(policy, path, device, strict=True):
    _policy_weights = torch.load(path, map_location=torch.device("cpu"), weights_only=False)['state_dict']
    policy_weights = {}
    for name, weight in _policy_weights.items():
        assert name.split('.')[0] == 'policy'
        policy_weights[name.lstrip('policy.')] = weight
    policy.load_state_dict(policy_weights, strict=strict)        
    policy = policy.to(device).eval()
    return policy




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        help="Problem name: cvrp, vrptw, etc. or all",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Problem size: 50, 100, for automatic loading",
    )
    parser.add_argument(
        "--datasets",
        help="Filename of the dataset(s) to evaluate. Defaults to all under data/{problem}/ dir",
        default=None,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='data'
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default='logs'
    )
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--remove-mixed-backhaul",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove mixed backhaul instances. Use --no-remove-mixed-backhaul to keep them.",
    )
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save results to results/main/{size}/{checkpoint}",
    )

    parser.add_argument('--model_name', type=str, default='rf_base')
    parser.add_argument('--lora_rank', type=int, nargs='+')
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--lora_use_gate', type=int, default=1)
    parser.add_argument('--lora_act_func', type=str, default='sigmoid')

    parser.add_argument('--lora_n_experts', type=int, default=4)
    parser.add_argument('--lora_top_k', type=int, default=4)
    parser.add_argument('--lora_temperature', type=float, default=1.0)
    parser.add_argument('--lora_use_trainable_layer', type=int, default=1)
    parser.add_argument('--lora_use_dynamic_topK', type=int, default=0)
    parser.add_argument('--lora_use_basis_variants', type=int, default=0)
    parser.add_argument('--lora_use_basis_variants_as_input', type=int, default=0)
    parser.add_argument('--lora_use_linear', type=int, default=0)

    parser.add_argument(
        '--decode_type',
        type=str,
        default='multistart_greedy',
        choices=['greedy', 'sampling', 'multistart_greedy', 'multistart_sampling'],
        help='Decoding strategy. Use multistart_sampling for multinomial sampling on feasible actions.',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Softmax temperature used by decoding strategy.',
    )
    parser.add_argument(
        '--rollout_times',
        type=int,
        default=1,
        help='Number of independent decoding rollouts per batch (use >1 with sampling).',
    )
    parser.add_argument(
        '--report_rollout_steps',
        type=int,
        nargs='*',
        default=None,
        help='Milestone rollout indices to report in a single run, e.g. --report_rollout_steps 20 75 150 300',
    )



    # Use load_from_checkpoint with map_location, which is handled internally by Lightning
    # Suppress FutureWarnings related to torch.load and weights_only
    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

    opts = parser.parse_args()
    opts.lora_use_gate = bool(opts.lora_use_gate)
    opts.lora_use_trainable_layer = bool(opts.lora_use_trainable_layer)
    opts.lora_use_dynamic_topK = bool(opts.lora_use_dynamic_topK)
    opts.lora_use_basis_variants = bool(opts.lora_use_basis_variants)
    opts.lora_use_basis_variants_as_input = bool(opts.lora_use_basis_variants_as_input)
    opts.lora_use_linear = bool(opts.lora_use_linear)
    assert opts.rollout_times >= 1, "rollout_times must be >= 1"
    assert opts.temperature > 0, "temperature must be > 0"
    report_steps = opts.report_rollout_steps if opts.report_rollout_steps is not None else [opts.rollout_times]
    report_steps = sorted(set(int(x) for x in report_steps if x > 0 and x <= opts.rollout_times))
    if len(report_steps) == 0:
        report_steps = [opts.rollout_times]

    log_file_name = f"test_{opts.size}_{opts.model_name}_{time.strftime('%Y%m%d-%H%M%S', time.localtime())}.txt"
    os.makedirs(opts.log_path, exist_ok=True)
    sys.stdout = Logger(os.path.join(opts.log_path, log_file_name), sys.stdout)
    sys.stderr = Logger(os.path.join(opts.log_path, log_file_name), sys.stderr)

    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if opts.datasets is not None:
        data_paths = opts.datasets.split(",")
    else:
        # list recursively all npz files in data/
        data_paths = []
        for root, _, files in os.walk(opts.dataset_path):
            for file in files:
                if "test" not in root:
                    continue
                if file.endswith(".npz"):
                    if opts.remove_mixed_backhaul and "m" in root and opts.size <= 100:
                        continue
                    if str(opts.size) in file:
                        if file == f"{opts.size}.npz":
                            data_paths.append(os.path.join(root, file))

        assert len(data_paths) > 0, "No datasets found. Check the data directory."
        data_paths = sorted(sorted(data_paths), key=lambda x: len(x))
        print(f"Found {len(data_paths)} datasets on the following paths: {data_paths}")

        ordered_tasks = [
            "cvrp", "vrptw", "ovrp", "vrpl",
            "vrpb", "ovrptw", "vrpbl", "vrpbltw",
            "vrpbtw", "vrpltw", "ovrpb", "ovrpbl",
            "ovrpbltw", "ovrpbtw", "ovrpl", "ovrpltw",
        ]
        ordered_paths = [f"{opts.dataset_path}/{task}/test/{opts.size}.npz" for task in ordered_tasks]
        data_paths = [ordered_p for ordered_p in ordered_paths if ordered_p in data_paths]


    # Load model
    print("Loading checkpoint from ", opts.checkpoint)
    if opts.model_name == 'rf_base':
        policy = RouteFinderPolicy(
            normalization='rms',
            encoder_use_prenorm=True,
            encoder_use_post_layers_norm=True,
            parallel_gated_kwargs={
                "mlp_activation": "silu"
            }
        )
    elif opts.model_name == 'cada_base':
        policy = CadaPolicy(
            normalization="rms",
            encoder_use_prenorm=False,
            encoder_use_post_layers_norm=False,
            parallel_gated_kwargs={
                "mlp_activation": "silu"
            },
            attn_sparse_ratio=0.5,
            sparse_applied_to_score=True,
        )
    elif opts.model_name == 'rf_lora':
        policy = LoRAPolicy(
            normalization='rms',
            encoder_use_prenorm=True,
            encoder_use_post_layers_norm=True,
            parallel_gated_kwargs={
                "mlp_activation": "silu"
            },
            lora_rank=opts.lora_rank,
            lora_alpha=opts.lora_alpha,
            lora_use_gate=opts.lora_use_gate,
            lora_act_func=opts.lora_act_func,
        )
    elif opts.model_name == 'cada_lora':
        policy = CadaLoRAPolicy(
            normalization="rms",
            encoder_use_prenorm=False,
            encoder_use_post_layers_norm=False,
            parallel_gated_kwargs={
                "mlp_activation": "silu"
            },
            attn_sparse_ratio=0.5,
            sparse_applied_to_score=True,
            lora_rank=opts.lora_rank,
            lora_alpha=opts.lora_alpha,
            lora_use_gate=opts.lora_use_gate,
            lora_act_func=opts.lora_act_func,
        )
    elif opts.model_name == 'rf_multilora':
        assert opts.lora_act_func in ['softmax','softplus','sigmoid']
        policy = MultiLoRAPolicy(
            normalization='rms',
            encoder_use_prenorm=True,
            encoder_use_post_layers_norm=True,
            parallel_gated_kwargs={
                "mlp_activation": "silu"
            },
            lora_rank=opts.lora_rank,
            lora_alpha=opts.lora_alpha,
            lora_act_func=opts.lora_act_func,
            lora_n_experts=opts.lora_n_experts,
            lora_top_k=opts.lora_top_k,
            lora_temperature=opts.lora_temperature,
            lora_use_trainable_layer=opts.lora_use_trainable_layer,
            lora_use_dynamic_topK=opts.lora_use_dynamic_topK,
            lora_use_basis_variants=opts.lora_use_basis_variants,
            lora_use_basis_variants_as_input=opts.lora_use_basis_variants_as_input,
            lora_use_linear=opts.lora_use_linear,
        )
    elif opts.model_name == 'cada_multilora':
        assert opts.lora_act_func in ['softmax','softplus','sigmoid']
        policy = CadaMultiLoRAPolicy(
            normalization="rms",
            encoder_use_prenorm=False,
            encoder_use_post_layers_norm=False,
            parallel_gated_kwargs={
                "mlp_activation": "silu"
            },
            attn_sparse_ratio=0.5,
            sparse_applied_to_score=True,
            lora_rank=opts.lora_rank,
            lora_alpha=opts.lora_alpha,
            lora_act_func=opts.lora_act_func,
            lora_n_experts=opts.lora_n_experts,
            lora_top_k=opts.lora_top_k,
            lora_temperature=opts.lora_temperature,
            lora_use_trainable_layer=opts.lora_use_trainable_layer,
            lora_use_dynamic_topK=opts.lora_use_dynamic_topK,
            lora_use_basis_variants=opts.lora_use_basis_variants,
            lora_use_basis_variants_as_input=opts.lora_use_basis_variants_as_input,
            lora_use_linear=opts.lora_use_linear,
        )
    else:
        raise NotImplementedError

    policy = load_model_weights(policy, opts.checkpoint, device, strict=False)

    env = MTVRPEnv()
    results = {}
    for dataset in data_paths:
        print("\n")
        dataset_name = dataset.split("/")[-3].split(".")[0].upper()

        td_test = env.load_data(dataset)  # this also adds the bks cost
        dataloader = get_dataloader(td_test, batch_size=opts.batch_size)

        start = time.time()
        best_reward = None
        best_gap = None
        milestone_results = {}
        for rollout_idx in range(1, opts.rollout_times + 1):
            res = []
            for batch in dataloader:
                td_test = env.reset(batch).to(device)
                o = test(
                    policy,
                    td_test,
                    env,
                    decode_type=opts.decode_type,
                    temperature=opts.temperature,
                    device=device,
                )
                res.append(o)

            current_reward = torch.cat([o["max_aug_reward"] for o in res])
            current_gap = torch.cat([o["gap_to_bks"] for o in res])

            if best_reward is None:
                best_reward = current_reward
                best_gap = current_gap
            else:
                assert best_gap is not None
                improved_mask = current_reward > best_reward
                best_reward = torch.where(improved_mask, current_reward, best_reward)
                best_gap = torch.where(improved_mask, current_gap, best_gap)

            if rollout_idx in report_steps:
                milestone_cost = -best_reward.mean().item()
                milestone_gap = best_gap.mean().item()
                milestone_results[rollout_idx] = {
                    "cost": milestone_cost,
                    "gap": milestone_gap,
                }
                print(
                    f"{dataset_name} | Rollout@{rollout_idx} | Cost: {milestone_cost:.3f} | Gap: {milestone_gap:.3f}%"
                )

        assert best_reward is not None and best_gap is not None
        out = {
            "max_aug_reward": best_reward,
            "gap_to_bks": best_gap,
        }

        inference_time = time.time() - start
        print(
            f"{dataset_name} | Cost: {-out['max_aug_reward'].mean().item():.3f} | Gap: {out['gap_to_bks'].mean().item():.3f}% | Inference time: {inference_time:.3f} s"
        )

        if results.get(dataset_name, None) is None:
            results[dataset_name] = {}
        results[dataset_name]["cost"] = -out["max_aug_reward"].mean().item()
        results[dataset_name]["gap"] = out["gap_to_bks"].mean().item()
        results[dataset_name]["inference_time"] = inference_time
        results[dataset_name]["rollout_snapshots"] = milestone_results


    if opts.save_results:
        # Save results with checkpoint name under results/main/
        checkpoint_name = opts.checkpoint.split("/")[-1].split(".")[0]
        savedir = f"results/main/{opts.size}/"
        os.makedirs(savedir, exist_ok=True)
        pickle.dump(results, open(savedir + checkpoint_name + ".pkl", "wb"))


