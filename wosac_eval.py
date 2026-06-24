import json
import pickle
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import torch
import torch.multiprocessing as mp
from google.protobuf import text_format
from tqdm import tqdm
from waymo_open_dataset.protos import sim_agents_metrics_pb2

from wosac_fast_eval_tool.fast_sim_agents_metrics import metrics as sim_agents_metric_api
from wosac_fast_eval_tool.scenario_gt_converter import gt_scenario_to_device


BASE_METRIC_NAMES = [
    'metametric',
    'average_displacement_error',
    'min_average_displacement_error',
    'linear_speed_likelihood',
    'linear_acceleration_likelihood',
    'angular_speed_likelihood',
    'angular_acceleration_likelihood',
    'distance_to_nearest_object_likelihood',
    'collision_indication_likelihood',
    'time_to_collision_likelihood',
    'distance_to_road_edge_likelihood',
    'offroad_indication_likelihood',
    'simulated_collision_rate',
    'simulated_offroad_rate',
]

METRIC_NAMES_2025 = [
    'traffic_light_violation_likelihood',
    'simulated_traffic_light_violation_rate',
]


@dataclass
class EvalResult:
    scenario_id: str
    rollout_file: str
    gt_file: str
    metrics: dict[str, float] | None = None
    error: str | None = None


def get_metric_names(version: str) -> list[str]:
    metric_names = list(BASE_METRIC_NAMES)
    if version == '2025':
        metric_names.extend(METRIC_NAMES_2025)
    return metric_names


def load_eval_config(version: str) -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    if version == '2024':
        proto_path = Path('wosac_fast_eval_tool/fast_sim_agents_metrics/challenge_2024_config.textproto')
    elif version == '2025':
        proto_path = Path('wosac_fast_eval_tool/fast_sim_agents_metrics/challenge_2025_sim_agents_config.textproto')
    else:
        raise ValueError(f'Unsupported WOSAC version: {version}')

    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    with open(proto_path, 'r', encoding='utf-8') as handle:
        text_format.Parse(handle.read(), config)
    return config


def load_pickle(path: Path) -> Any:
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def infer_scenario_id_from_name(file_name: str) -> str:
    suffix = '_closedloop_wosac.pkl'
    if file_name.endswith(suffix):
        scenario_id = file_name[:-len(suffix)]
    elif file_name.endswith('.pkl'):
        scenario_id = file_name[:-4]
    else:
        scenario_id = Path(file_name).stem

    if scenario_id.startswith('scenario_'):
        scenario_id = scenario_id[len('scenario_'):]
    return scenario_id


def normalize_prediction(
    predict: dict[str, Any],
    device: str,
    rollout_key: str,
) -> dict[str, torch.Tensor]:
    if 'agent_id' in predict and 'simulated_states' in predict:
        return {
            'agent_id': torch.as_tensor(predict['agent_id'], device=device),
            'simulated_states': torch.as_tensor(predict['simulated_states'], device=device),
        }

    if 'agents_id' in predict and 'model_rollouts' in predict:
        if rollout_key not in predict['model_rollouts']:
            available = ', '.join(sorted(predict['model_rollouts'].keys()))
            raise KeyError(
                f"Requested rollout key '{rollout_key}' not found. Available keys: [{available}]"
            )
        rollout_branch = predict['model_rollouts'][rollout_key]
        if 'rollouts' not in rollout_branch:
            keys = ', '.join(sorted(rollout_branch.keys()))
            raise KeyError(
                f"CATK rollout branch '{rollout_key}' must contain 'rollouts'. Found keys: [{keys}]"
            )
        agent_id = torch.as_tensor(predict['agents_id'], device=device)
        simulated_states = torch.as_tensor(rollout_branch['rollouts'], device=device)
        sim_agent_mask = predict.get('sim_agent_mask')
        if sim_agent_mask is not None:
            valid_agent_mask = torch.as_tensor(sim_agent_mask, device=device, dtype=torch.bool)
            agent_id = agent_id[valid_agent_mask]
            simulated_states = simulated_states[:, valid_agent_mask]
        return {
            'agent_id': agent_id,
            'simulated_states': simulated_states,
        }

    keys = ', '.join(sorted(predict.keys()))
    raise KeyError(
        "Rollout pickle must be either standard format with 'agent_id'/'simulated_states' "
        "or CATK format with 'agents_id'/'model_rollouts'. "
        f'Found keys: [{keys}]'
    )


def fill_missing_agents_with_gt(
    predict: dict[str, torch.Tensor],
    gt_scenario: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    pred_agent_ids = predict['agent_id'].int()
    simulated_states = predict['simulated_states']
    sim_agent_ids = gt_scenario['sim_agent_ids'].int()

    object_ids = gt_scenario['object_ids'].int()
    existing_pos, existing_gt_indices = torch.where(pred_agent_ids.unsqueeze(1) == object_ids.unsqueeze(0))
    if existing_gt_indices.numel() > 0:
        order = torch.argsort(existing_pos)
        existing_pos = existing_pos[order]
        existing_gt_indices = existing_gt_indices[order]
        num_steps = simulated_states.shape[2]
        gt_z = gt_scenario['tracks'][existing_gt_indices, 11 : 11 + num_steps, 2].to(simulated_states.dtype)
        simulated_states = simulated_states.clone()
        simulated_states[:, existing_pos, :, 2] = gt_z.unsqueeze(0).expand(simulated_states.shape[0], -1, -1)

    missing_agent_ids = sim_agent_ids[~torch.isin(sim_agent_ids, pred_agent_ids)]
    if missing_agent_ids.numel() == 0:
        return {
            'agent_id': pred_agent_ids,
            'simulated_states': simulated_states,
        }

    missing_pos, gt_indices = torch.where(missing_agent_ids.unsqueeze(1) == object_ids.unsqueeze(0))
    if gt_indices.numel() != missing_agent_ids.numel():
        found_agent_ids = object_ids[gt_indices]
        not_found = missing_agent_ids[~torch.isin(missing_agent_ids, found_agent_ids)]
        raise KeyError(f'GT tracks missing sim agent ids needed for fill: {not_found.detach().cpu().tolist()}')

    order = torch.argsort(missing_pos)
    gt_indices = gt_indices[order]
    missing_agent_ids = missing_agent_ids[missing_pos[order]]

    num_rollouts, _, num_steps, state_dim = simulated_states.shape
    gt_future = gt_scenario['tracks'][gt_indices, 11 : 11 + num_steps]
    if gt_future.shape[1] < num_steps:
        raise ValueError(
            f'GT future has only {gt_future.shape[1]} steps, cannot fill {num_steps} simulated steps'
        )

    gt_simulated_states = gt_future[..., [0, 1, 2, 6]].to(simulated_states.dtype)
    if state_dim != 4:
        raise ValueError(f'Expected simulated states with 4 dims (x,y,z,yaw), got {state_dim}')
    gt_simulated_states = gt_simulated_states.unsqueeze(0).repeat(num_rollouts, 1, 1, 1)

    return {
        'agent_id': torch.cat([pred_agent_ids, missing_agent_ids.to(pred_agent_ids.dtype)], dim=0),
        'simulated_states': torch.cat([simulated_states, gt_simulated_states], dim=1),
    }


def evaluate_one_file(
    scenario_id: str,
    rollout_path: Path,
    gt_path: Path,
    eval_config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    version: str,
    device: str,
    rollout_key: str,
    fill_missing_agents_with_gt_flag: bool,
) -> EvalResult:
    try:
        gt_scenario = load_pickle(gt_path)
        predict = load_pickle(rollout_path)
        gt_scenario = gt_scenario_to_device(gt_scenario, device=device)
        predict = normalize_prediction(predict, device=device, rollout_key=rollout_key)
        if fill_missing_agents_with_gt_flag:
            predict = fill_missing_agents_with_gt(predict, gt_scenario)
        metrics = sim_agents_metric_api.compute_scenario_metrics_for_bundle(
            eval_config,
            gt_scenario,
            predict,
            version,
        )
        return EvalResult(
            scenario_id=scenario_id,
            rollout_file=rollout_path.name,
            gt_file=gt_path.name,
            metrics=dict(metrics),
        )
    except Exception as exc:
        return EvalResult(
            scenario_id=scenario_id,
            rollout_file=rollout_path.name,
            gt_file=gt_path.name,
            error=str(exc),
        )


def worker(
    device: str,
    eval_config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    file_queue: Any,
    result_queue: Any,
    version: str,
    rollout_key: str,
    fill_missing_agents_with_gt_flag: bool,
) -> None:
    if device.startswith('cuda'):
        torch.cuda.set_device(device)

    while True:
        try:
            item = file_queue.get_nowait()
        except Empty:
            break

        result = evaluate_one_file(
            scenario_id=item['scenario_id'],
            rollout_path=Path(item['rollout_path']),
            gt_path=Path(item['gt_path']),
            eval_config=eval_config,
            version=version,
            device=device,
            rollout_key=rollout_key,
            fill_missing_agents_with_gt_flag=fill_missing_agents_with_gt_flag,
        )
        result_queue.put(result.__dict__)


def aggregate_metrics(results: list[EvalResult], metric_names: list[str]) -> dict[str, float]:
    successful = [result for result in results if result.metrics is not None]
    if not successful:
        return {}

    aggregate = {metric_name: 0.0 for metric_name in metric_names}
    for result in successful:
        assert result.metrics is not None
        for metric_name in metric_names:
            aggregate[metric_name] += float(result.metrics[metric_name])

    n_success = len(successful)
    for metric_name in metric_names:
        aggregate[metric_name] /= n_success
    return aggregate


def build_report(
    *,
    version: str,
    rollout_dir: Path,
    gt_dir: Path,
    matched_files: list[dict[str, str]],
    missing_gt_files: list[str],
    missing_rollout_files: list[str],
    results: list[EvalResult],
    metric_names: list[str],
    rollout_key: str,
    fill_missing_agents_with_gt_flag: bool,
) -> dict[str, Any]:
    successful = [result for result in results if result.metrics is not None]
    failed = [result for result in results if result.error is not None]

    return {
        'version': version,
        'rollout_dir': str(rollout_dir.resolve()),
        'gt_dir': str(gt_dir.resolve()),
        'rollout_key': rollout_key,
        'fill_missing_agents_with_gt': fill_missing_agents_with_gt_flag,
        'summary': {
            'matched_files': len(matched_files),
            'successful_files': len(successful),
            'failed_files': len(failed),
            'missing_gt_files': len(missing_gt_files),
            'missing_rollout_files': len(missing_rollout_files),
        },
        'metrics': aggregate_metrics(results, metric_names),
        'failed_files': [
            {
                'scenario_id': result.scenario_id,
                'rollout_file': result.rollout_file,
                'gt_file': result.gt_file,
                'error': result.error,
            }
            for result in failed
        ],
        'missing_gt_files': missing_gt_files,
        'missing_rollout_files': missing_rollout_files,
    }


def resolve_files(rollout_dir: Path, gt_dir: Path) -> tuple[list[dict[str, str]], list[str], list[str]]:
    rollout_paths = sorted(rollout_dir.glob('*.pkl'))
    gt_paths = sorted(gt_dir.glob('*.pkl'))

    rollout_map = {}
    for path in rollout_paths:
        scenario_id = infer_scenario_id_from_name(path.name)
        if scenario_id not in rollout_map:
            rollout_map[scenario_id] = path

    gt_map = {path.stem: path for path in gt_paths}

    rollout_ids = set(rollout_map.keys())
    gt_ids = set(gt_map.keys())
    matched_ids = sorted(rollout_ids & gt_ids)
    missing_gt_files = sorted(rollout_ids - gt_ids)
    missing_rollout_files = sorted(gt_ids - rollout_ids)

    matched_files = [
        {
            'scenario_id': scenario_id,
            'rollout_path': str(rollout_map[scenario_id]),
            'gt_path': str(gt_map[scenario_id]),
        }
        for scenario_id in matched_ids
    ]
    return matched_files, missing_gt_files, missing_rollout_files


def run_eval(
    *,
    rollout_dir: Path,
    gt_dir: Path,
    version: str,
    num_gpus: int | None,
    force_device: str | None,
    debug: bool,
    rollout_key: str,
    fill_missing_agents_with_gt_flag: bool,
) -> dict[str, Any]:
    eval_config = load_eval_config(version)
    metric_names = get_metric_names(version)
    matched_files, missing_gt_files, missing_rollout_files = resolve_files(rollout_dir, gt_dir)

    if not matched_files:
        raise RuntimeError(
            'No matched rollout/GT pickle files were found. Expected scenario_id alignment between both directories.'
        )

    results: list[EvalResult] = []
    available_cuda = torch.cuda.device_count()
    use_cpu = force_device == 'cpu' or available_cuda == 0 or num_gpus == 0

    if debug or use_cpu:
        device = force_device or ('cuda:0' if available_cuda > 0 and not use_cpu else 'cpu')
        for item in tqdm(matched_files, desc='Evaluating WOSAC'):
            results.append(
                evaluate_one_file(
                    scenario_id=item['scenario_id'],
                    rollout_path=Path(item['rollout_path']),
                    gt_path=Path(item['gt_path']),
                    eval_config=eval_config,
                    version=version,
                    device=device,
                    rollout_key=rollout_key,
                    fill_missing_agents_with_gt_flag=fill_missing_agents_with_gt_flag,
                )
            )
    else:
        manager = mp.Manager()
        file_queue = manager.Queue()
        result_queue = manager.Queue()
        for item in matched_files:
            file_queue.put(item)

        worker_count = available_cuda if num_gpus is None else min(num_gpus, available_cuda)
        processes = []
        for worker_idx in range(worker_count):
            process = mp.Process(
                target=worker,
                args=(
                    f'cuda:{worker_idx}',
                    eval_config,
                    file_queue,
                    result_queue,
                    version,
                    rollout_key,
                    fill_missing_agents_with_gt_flag,
                ),
            )
            process.start()
            processes.append(process)

        with tqdm(total=len(matched_files), desc='Evaluating WOSAC') as progress_bar:
            while len(results) < len(matched_files):
                result = EvalResult(**result_queue.get())
                results.append(result)
                progress_bar.update(1)

        for process in processes:
            process.join()

    return build_report(
        version=version,
        rollout_dir=rollout_dir,
        gt_dir=gt_dir,
        matched_files=matched_files,
        missing_gt_files=missing_gt_files,
        missing_rollout_files=missing_rollout_files,
        results=results,
        metric_names=metric_names,
        rollout_key=rollout_key,
        fill_missing_agents_with_gt_flag=fill_missing_agents_with_gt_flag,
    )


def main() -> None:
    mp.set_start_method('spawn', force=True)
    parser = ArgumentParser(description='Offline WOSAC evaluator for rollout pickle files.')
    parser.add_argument('rollout_dir', type=Path, help='Directory containing rollout pickle files.')
    parser.add_argument(
        '--gt-dir',
        type=Path,
        default=Path('data/waymo_processed/validation_gt'),
        help='Directory containing GT scenario pickle files.',
    )
    parser.add_argument('--version', type=str, choices=['2024', '2025'], default='2025', help='WOSAC metric version.')
    parser.add_argument(
        '--rollout-key',
        type=str,
        choices=['baseline', 'controlnet', 'vbd_ttg_sparse_gt'],
        default='baseline',
        help='CATK rollout branch to evaluate when using CATK closed-loop rollout pickles.',
    )
    parser.add_argument(
        '--report-path',
        type=Path,
        default=None,
        help='Optional JSON output path. Defaults to <rollout_dir>/wosac_metrics_report.json.',
    )
    parser.add_argument('--num-gpus', type=int, default=None, help='Number of GPUs to use. Set 0 to force CPU.')
    parser.add_argument('--device', type=str, choices=['cpu'], default=None, help='Force CPU execution.')
    parser.add_argument('--debug', action='store_true', help='Run in a single process for easier debugging.')
    parser.add_argument(
        '--fill-missing-agents-with-gt',
        action='store_true',
        help='Append GT future trajectories for WOSAC sim agents missing from the rollout prediction.',
    )
    parser.add_argument('--fail-on-error', action='store_true', help='Exit non-zero if any matched scenario fails to evaluate.')
    args = parser.parse_args()

    rollout_dir = args.rollout_dir.resolve()
    gt_dir = args.gt_dir.resolve()
    report_path = args.report_path.resolve() if args.report_path is not None else rollout_dir / 'wosac_metrics_report.json'

    if not rollout_dir.is_dir():
        raise NotADirectoryError(f'Rollout directory does not exist: {rollout_dir}')
    if not gt_dir.is_dir():
        raise NotADirectoryError(f'GT directory does not exist: {gt_dir}')

    report = run_eval(
        rollout_dir=rollout_dir,
        gt_dir=gt_dir,
        version=args.version,
        num_gpus=args.num_gpus,
        force_device=args.device,
        debug=args.debug,
        rollout_key=args.rollout_key,
        fill_missing_agents_with_gt_flag=args.fill_missing_agents_with_gt,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print(json.dumps(report['summary'], indent=2, sort_keys=True))
    print('metrics:')
    print(json.dumps(report['metrics'], indent=2, sort_keys=True))
    print(f'report saved to {report_path}')

    if args.fail_on_error and report['summary']['failed_files'] > 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
