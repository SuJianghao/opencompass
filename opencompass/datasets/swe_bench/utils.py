import platform
import traceback
from pathlib import Path, PurePosixPath
import re

from opencompass.utils import get_logger

logger = get_logger()


def get_remote_docker_image_from_id(instance_id: str) -> str:
    """Image name format as found on DockerHub since swebench v3.0"""
    # NOTE: The swebench library contains this logic within `make_test_spec`,
    # but this module would require significant refactoring to use it.
    updated_instance_id = instance_id.replace('__', '_1776_')
    if platform.machine() in {'aarch64', 'arm64'}:
        from swebench.harness.constants import USE_X86  # type: ignore

        # use arm64 unless explicitly specified
        arch = 'arm64' if instance_id not in USE_X86 else 'x86_64'
    else:
        arch = 'x86_64'
    return f'swebench/sweb.eval.{arch}.{updated_instance_id}:latest'


GIT_APPLY_CMDS = [
    'git apply --verbose',
    'git apply --verbose --reject',
    'patch --batch --fuzz=5 -p1 -i',
]


def eval_instance(instance: dict, pred: str, timeout: int = 1800, log_dir: str = 'outputs'):
    from docker.client import DockerClient
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL,
        APPLY_PATCH_PASS,
        DOCKER_PATCH,
        DOCKER_USER,
        DOCKER_WORKDIR,
        KEY_INSTANCE_ID,
        KEY_MODEL,
        KEY_PREDICTION,
        LOG_TEST_OUTPUT,
        UTF8,
    )
    from swebench.harness.docker_utils import cleanup_container, copy_to_container, exec_run_with_timeout
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

    from .build_images import build_container

    # Build + start instance container (instance image should already be built)
    container = None
    eval_completed = False
    report = {}
    try:
        test_spec: TestSpec = make_test_spec(instance, namespace='swebench')
        instance_id = test_spec.instance_id
        pred = {
            KEY_PREDICTION: pred,
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: '',
        }
        log_dir = Path(log_dir) / 'swebench_log' / instance_id

        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Starting evaluation for {instance_id} in log dir {log_dir}...')

        client = DockerClient.from_env()
        container = build_container(test_spec, client=client)
        container.start()
        logger.info(f'Container for {instance_id} started: {container.id}')

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / 'patch.diff')
        patch_file.write_text(pred[KEY_PREDICTION] or '')
        logger.info(f'Intermediate patch for {instance_id} written to {patch_file}, now applying to container...')
        copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))

        # Attempt to apply patch to container
        applied_patch = False
        for git_apply_cmd in GIT_APPLY_CMDS:
            val = container.exec_run(
                f'{git_apply_cmd} {DOCKER_PATCH}',
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            if val.exit_code == 0:
                logger.info(f'{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8)}')
                applied_patch = True
                break
            else:
                logger.info(f'Failed to apply patch to container: {git_apply_cmd}')
        if not applied_patch:
            logger.info(f'{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}')
            raise Exception(f'{instance_id} {APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}')

        # Get git diff before running eval script
        git_diff_output_before = (
            container.exec_run('git -c core.fileMode=false diff', workdir=DOCKER_WORKDIR).output.decode(UTF8).strip()
        )
        logger.info(f'Git diff before:\n{git_diff_output_before}')

        eval_file = Path(log_dir / 'eval.sh')
        eval_file.write_text(test_spec.eval_script)
        logger.info(f'Eval script for {instance_id} written to {eval_file}; copying to container...')
        copy_to_container(container, eval_file, PurePosixPath('/eval.sh'))

        # Run eval script, write output to logs
        test_output, timed_out, total_runtime = exec_run_with_timeout(container, '/bin/bash /eval.sh', timeout)
        test_output_path = log_dir / LOG_TEST_OUTPUT
        logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
        with open(test_output_path, 'w') as f:
            f.write(test_output)
            logger.info(f'Test output for {instance_id} written to {test_output_path}')
            if timed_out:
                f.write(f'\n\nTimeout error: {timeout} seconds exceeded.')
                raise TimeoutError(f'{instance_id} Test timed out after {timeout} seconds.')

        # Get git diff after running eval script (ignore permission changes)
        git_diff_output_after = (
            container.exec_run('git -c core.fileMode=false diff', workdir=DOCKER_WORKDIR).output.decode(UTF8).strip()
        )

        # Check if git diff changed after running eval script
        logger.info(f'Git diff after:\n{git_diff_output_after}')
        if git_diff_output_after != git_diff_output_before:
            logger.info('Git diff changed after running eval script')

        # Get report from test output
        logger.info(f'Grading answer for {instance_id}...')
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(f'report: {report}\n'
                    f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}")
        eval_completed = True
    except Exception as e:
        error_msg = (f'Error in evaluating model for {instance_id}: {e}\n'
                     f'{traceback.format_exc()}')
        logger.error(error_msg)
        report['error'] = error_msg
    finally:
        # Remove instance container
        cleanup_container(client, container, logger)
        return {
            'completed': eval_completed,
            'resolved': report.get(instance_id, {}).get('resolved', False),
            'report': report,
        }
    

def extract_diff(response):
    """
    Extracts the diff from a response formatted in different ways
    """
    if response is None:
        return None
    diff_matches = []
    other_matches = []
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    return response.split("</s>")[0]


def run_in_threads_with_progress(
    items: Sequence[T],
    worker: Callable[[T], R],
    *,
    desc: str,
    max_workers: int,
    log_interval: Optional[int] = None,
    on_result: Optional[Callable[[T, R], None]] = None,
    on_error: Optional[Callable[[T, Exception], None]] = None,
    filter_none_results: bool = False,
) -> List[R]:
    """
    Execute a collection of tasks concurrently with a ThreadPoolExecutor while
    displaying a tqdm progress bar and emitting periodic heartbeat logs.

    Key behaviors:
    - Concurrency: Uses up to `min(len(items), max_workers)` threads.
    - Progress: A tqdm bar advances when each task finishes (success or failure).
    - Heartbeat: If no tasks finish within `heartbeat_sec`, a status line is logged.
    - Ordering: Results are appended in completion order (not the original order).
    - Error handling:
        * If `on_error` is provided, it is called for each failed item; execution continues
          unless `on_error` itself raises.
        * If `on_error` is None, the first exception is raised immediately and stops processing.
    - Callbacks:
        * `on_result(item, result)` is called after a successful result is obtained.
        * Both callbacks run in the main thread (not worker threads).

    Args:
        items: A sequence of items (inputs) to process. Converted to a list internally.
        worker: A callable executed in threads to process a single item and return a result.
        desc: A short text shown as the tqdm progress bar description.
        max_workers: Upper bound on the number of concurrent threads.
        heartbeat_sec: Interval (in seconds) to wait before emitting a heartbeat log if
            no tasks complete in that window.
        on_result: Optional callback invoked as on_result(item, result) after success.
        on_error: Optional callback invoked as on_error(item, exception) on failure. If omitted,
            the exception is propagated and the function terminates early.

    Returns:
        A list of results collected as tasks complete (completion order).
        If some tasks fail and `on_error` is provided (and does not re-raise), those failures
        are skipped and not included in the returned results.

    Raises:
        Exception: Propagates the first task exception if `on_error` is not provided, or if
        `on_error` re-raises.

    Notes:
        - The function is blocking until all tasks complete or an exception is propagated.
        - Use `on_error` to implement "best-effort" processing where failures are logged
          and the rest continue.
    """
    # Defensive copy to avoid consuming a generator multiple times and to compute pool size.
    pending_items: List[T] = list(items)
    if not pending_items:
        return []

    # Include indices to ensure results are returned in input order
    indexed_items = list(enumerate(items))
    results: List[Optional[R]] = [None] * len(items)  # Preallocate results list

    # Bound the pool by actual workload size for efficiency.
    with ThreadPoolExecutor(max_workers=min(len(indexed_items), max_workers)) as executor:
        # Submit all tasks up-front and map futures back to their originating item.
        future_to_index = {executor.submit(worker, item): index for index, item in indexed_items}

        # Progress bar reflects total number of submitted tasks; updated per finished future.
        with tqdm(
            total=len(indexed_items),
            desc=desc,
            mininterval=1,
            dynamic_ncols=True,
            logger=logger,
            log_interval=log_interval
        ) as pbar:
            # Track unfinished futures and poll with a timeout to enable heartbeat logs.
            pending = set(future_to_index.keys())
            while pending:
                # Wait with timeout to detect stalls and emit heartbeats proactively.
                done, not_done = wait(pending, timeout=1)
                if not done:
                    # Heartbeat when nothing has completed within the window.
                    pbar.check_log()
                    continue

                # Consume completed futures.
                for future in done:
                    index = future_to_index[future]
                    try:
                        res = future.result()
                        results[index] = res  # Store result at the correct index
                        # Invoke success callback in caller thread (not in worker).
                        if on_result is not None:
                            on_result(items[index], res)
                    except Exception as exc:
                        # Delegate failure handling to on_error if provided; otherwise bubble up.
                        if on_error is not None:
                            on_error(items[index], exc)
                        else:
                            raise
                    finally:
                        # Always advance progress for completed futures (success or failure).
                        pbar.update(1)
                        pbar.refresh()

                # Continue polling remaining futures.
                pending = not_done

    # Return results, which are now guaranteed to be in input order
    if filter_none_results:
        # Filter out None results if on_error was used and some tasks failed
        results = [res for res in results if res is not None]
    return results