from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
import re
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from hide.client.hide_client import CreateProjectRequest, Repository, Project, TaskResult
from hide.devcontainer.model import ImageDevContainer
import yaml
from ghapi.all import GhApi
from git import Repo
from simple_parsing.helpers.serialization.serializable import FrozenSerializable
from swebench import MAP_VERSION_TO_INSTALL, get_environment_yml, get_requirements

import docker
import docker.errors
import docker.models.containers
from hide import HideClient
from sweagent import REPO_ROOT
from sweagent.environment.constants import DEFAULT_PYTHON_IMAGE, DEFAULT_PYTHON_VERSION, PYTHON_IMAGES
from sweagent.environment.utils import (
    PROCESS_DONE_MARKER_END,
    PROCESS_DONE_MARKER_START,
    InvalidGithubURL,
    copy_anything_to_container,
    format_trajectory_markdown,
    get_container,
    get_gh_issue_data,
    image_exists,
    parse_gh_issue_url,
    read_with_timeout,
    read_with_timeout_experimental,
)
from sweagent.utils.config import keys_config
from sweagent.utils.log import default_logger, get_logger

LONG_TIMEOUT = 500
PATH_TO_REQS = "/root/requirements.txt"
PATH_TO_ENV_YML = "/root/environment.yml"


@dataclass(frozen=True)
class EnvironmentArguments(FrozenSerializable):
    """Configure data sources and setup instructions for the environment in which we solve the tasks."""

    # Source of issue statement/problem statement. To run over a batch of issues: Path to a data file
    # (`json`, `jsonl`) or directory. To run over single issue: github issue url or path to markdown file
    # with problem statement or problem statement as text prefixed with `text://`.
    data_path: str
    # Name of the docker image to use for the environment. Defaults to sweagent/swe-agent:latest
    image_name: str = "sweagent/swe-agent:latest"
    # When running over SWE-bench issues: Specify the split to use.
    split: str = "dev"
    # Specify a branch name or a commit hash to checkout before running the task.
    # Only used when running over a single problem statement/issue.
    base_commit: str | None = None
    # Use a persistent container with this name. After every task, the container will be paused, but not removed.
    # This is useful for speedup when running multiple tasks from the same repositories in a row, as the repositories
    # will have already been cloned and the conda environments will have been installed.
    container_name: str | None = None
    # Try to install the environment before running the task.
    install_environment: bool = True
    # No effect, kept for backwards compatibility.
    timeout: int | None = None
    # Enable environment logger.
    verbose: bool = False
    # Do not use attempt to use a repository mirror from https://github.com/swe-bench.
    no_mirror: bool = False
    # Cache task images to speed up task initialization. This means that the environment will be saved as a
    # docker image for every repository, base commit, and setup combination. This uses quite a bit of disk space
    # but speeds up task initialization significantly when running over multiple issues from the same repository
    # (or using different models for the same issues).
    cache_task_images: bool = False
    # Custom environment setup. Currently only used when data_path points to a single issue.
    # This needs to be either a string pointing to a yaml file (with yaml, yml file extension)
    # or a shell script (with sh extension).
    # See https://princeton-nlp.github.io/SWE-agent/usage/cl_tutorial#environment-setup
    environment_setup: str | None = None
    # Only used when running on single issue. Path to local repository or github repository.
    repo_path: str = ""

    def __post_init__(self):
        if self.timeout is not None:
            default_logger.warning("The 'timeout' argument is deprecated and has no effect.")
        if self.cache_task_images and self.container_name:
            msg = (
                "Not allowed to use persistent container with caching task images "
                "(probably doesn't make sense and takes excessive space)."
            )
            raise ValueError(msg)
        if self.container_name is not None and self.container_name.strip() == "":
            msg = "Set container_name to None if you don't want to use a persistent container."
            raise ValueError(msg)


class EnvHook:
    """Hook to be used in `SWEEnv`.

    Subclass this class, add functionality and add it with `SWEEEnv.add_hook(hook)`.
    This allows to inject custom functionality at different stages of the environment
    lifecycle, in particular to connect SWE-agent to a new interface (like a GUI).
    """

    def on_init(self) -> None:
        """Gets called when the hook is added"""

    def on_copy_repo_started(self, *, repo_type: str, repo_path: str) -> None:
        """Gets called when the repository is being cloned to the container

        Args:
            repo_type: Type of repository. Either 'local' or 'github'
            repo_path: Path to the repository
        """

    def on_install_env_started(self) -> None:
        """Called when we start installing the environment"""

    def on_close(self):
        """Called when the environment is closed"""


class SWEEnvService:
    """Provisioner for SWE-agent environment"""

    def __init__(self, args: EnvironmentArguments, hide: HideClient):
        self.args = args
        self.hide = hide
        self.logger = get_logger("SWEEnvProvisioner")

    async def create_env(self, issue: dict) -> SWEEnv:
        """Create environment"""
        install_configs = self._get_install_configs(issue)
        if not install_configs:
            # TODO: handle missing install configs
            raise ValueError("No install configs found")

        # image = self._get_image(install_configs)
        image = self.args.image_name
        on_create_command = self._get_on_create_command(install_configs)
        on_create_command["flake8"] = "pip install flake8"

        container_config = ImageDevContainer(
            name="swe-agent-dev",
            image=image, 
            containerEnv=self._get_default_env_variables(),
            onCreateCommand=on_create_command,
        )

        repository = Repository(
            url=self._get_repository_url(issue),
            commit=issue["base_commit"],
        )

        request = CreateProjectRequest(
            repository=repository,
            devcontainer=container_config,
        )

        # async?
        project = self.hide.create_project(request)
        return SWEEnv(args=self.args, hide=self.hide, project=project, record=issue)

    async def delete_env(self, env: SWEEnv):
        """Delete environment"""
        self.hide.delete_project(env.project)

    def _get_repository_url(self, issue: dict) -> str:
        """Get repository URL from issue data"""
        # TODO: Using the token in the URL is a security risk
        github_token = keys_config.get("GITHUB_TOKEN", "")  # type: ignore
        token_prefix = f"{github_token}@" if github_token else ""
        # fixme: This if statement is brittle and should probably be replaced with better logic
        if not self.args.no_mirror and issue["problem_statement_source"] == "swe-bench":
            repo_name = issue["repo"].replace("/", "__")
            self.logger.info(f"{token_prefix} not found in container, cloning...")
            return f"https://{token_prefix}github.com/swe-bench/{repo_name}.git"
        else:
            self.logger.info("Trying to clone from non-mirror...")
            return f"https://github.com/{issue['repo']}.git"

    def _get_default_env_variables(self) -> dict[str, str]:
        """Get default environment variables"""
        return {
            "CURRENT_FILE": "",
            "CURRENT_LINE": "0",
            "SEARCH_RESULTS": "()",
            "SEARCH_FILES": "()",
            "SEARCH_INDEX": "0",
        }

    def _get_install_configs(self, issue: dict) -> dict | None:
        """Return config for environment setup"""
        if (
            issue["problem_statement_source"] != "swe-bench" or issue["repo_type"] == "local"
        ) and self.args.environment_setup is None:
            self.logger.warning(
                "install_environment is set to True, but the data path is a GitHub URL "
                "without an environment config file (environment_config key/flag). "
                "Skipping conda environment installation.",
            )
            return None
        if self.args.environment_setup is not None:
            assert isinstance(self.args.environment_setup, (str, os.PathLike))
            if Path(self.args.environment_setup).suffix in [".yml", ".yaml"]:
                try:
                    return yaml.safe_load(Path(self.args.environment_setup).read_text())
                except Exception as e:
                    msg = "Environment config file needs to be a yaml file"
                    raise ValueError(msg) from e
            elif Path(self.args.environment_setup).suffix == ".sh":
                return {
                    "shell_script_path": self.args.environment_setup,
                }
            else:
                msg = "Environment config file needs to be a yaml file or shell script"
                raise ValueError(msg)
        else:
            try:
                return MAP_VERSION_TO_INSTALL[issue["repo"]][str(issue["version"])]
            except KeyError as e:
                msg = (
                    "Tried to look up install configs in swe-bench, but failed. "
                    "You can set a custom environment config with the environment_config key/flag."
                )
                raise ValueError(msg) from e

    def _get_image(self, install_configs: dict[str, str]) -> str:
        python_version = install_configs.get("python", DEFAULT_PYTHON_VERSION)
        return PYTHON_IMAGES.get(python_version, DEFAULT_PYTHON_IMAGE)

    def _get_on_create_command(self, install_configs: dict[str, str]) -> dict[str, str]:
        commands = {}

        if "packages" in install_configs:
            packages = install_configs["packages"]

            if packages == "requirements.txt":
                commands["packages"] = "pip install -r requirements.txt"
            elif packages == "environment.yml":
                # TODO: add support for conda environment
                raise NotImplementedError("Conda environment is not supported yet.")
            else:
                commands["packages"] = f"pip install {packages}"

        if "pip_packages" in install_configs:
            pip_packages = install_configs["pip_packages"]
            commands["pip_packages"] = f"pip install {' '.join(pip_packages)}"

        if "pre_install" in install_configs:
            pre_install = install_configs["pre_install"]
            commands["pre_install"] = pre_install

        if "install" in install_configs:
            install = install_configs["install"]
            commands["install"] = install

        if "post_install" in install_configs:
            post_install = install_configs["post_install"]
            commands["post_install"] = post_install

        return commands

class SWEEnv(gym.Env):
    """Gym environment for SWE-bench. This class should handle all communication with the docker container."""

    name = "swe_main"
    # This prefix will be prepended to the image name when caching task images
    cached_image_prefix = "swe-agent-task-env-"

    def __init__(self, args: EnvironmentArguments, hide: HideClient, project: Project, record: dict):
        super().__init__()
        t0 = time.perf_counter()
        self.args = args
        self.base_commit: str | None = None
        self.communicate_output: str | None = None
        self.container_name: str | None = args.container_name
        self.install_environment = args.install_environment
        self.logger = get_logger("SWEEnv")
        self.persistent = args.container_name is not None
        self.returncode: None | int = None
        if not self.args.verbose:
            # fixme: This creates problems if we have multiple instances of this class
            self.logger.disabled = True

        self.hide = hide
        self.project = project

        #: The commit hash of the swe-agent repository
        self.commit_sha = None
        try:
            repo = Repo(REPO_ROOT, search_parent_directories=True)
            self.commit_sha = repo.head.object.hexsha
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.exception("Failed to get commit hash for this repo: %s", str(e))

        self._github_token: str = keys_config.get("GITHUB_TOKEN", "")  # type: ignore

        self.record = record
        self.clean_multi_line_functions = lambda x: x
        self.hooks: list[EnvHook] = []

        # TODO: do it when creating the container
        self._init_scripts()

        self.logger.debug("Environment initialization took %.2f seconds", time.perf_counter() - t0)

    def step(self, action: str) -> tuple[str | None, int, bool, dict]:
        """
        Runs given action in environment and returns corresponding output

        Args:
            action: command to run in bash shell

        Returns:
            observation:  output from container
            reward: value between 0 and 1 quantifying correctness of output + environment state
            done: whether task is over
            info: additional information (e.g. debugging information)
        """
        info = {}

        observation = ""
        # Handle special actions
        if action.strip() == "skip":
            observation = "Skipped"
            info["exit_status"] = "skipped"
            return observation, 0, True, info
        if action in {"exit_context", "exit_cost", "exit_error", "exit_format", "exit_api"}:
            try:
                output = self.communicate(input="submit")
                observation = output.stdOut if output.stdOut else output.stdErr
                submission = self.get_submission(observation)
                assert submission is not None and submission.strip() != "", AssertionError("No submission found.")
                self.logger.info(f"Found submission: {submission}")
                info["exit_status"] = f"submitted ({action})"
                info["submission"] = submission
                observation = "Exited (autosubmitted)"
                self.logger.info("Exiting with autosubmission")
                return observation, 0, True, info
            except KeyboardInterrupt:
                raise
            except:
                observation = "Exited"
                info["exit_status"] = action
                return observation, 0, True, info

        # Attempt to run action in container
        observation = ""
        try:
            output = self.communicate(input=action, timeout_duration=25)
            observation = output.stdOut if output.stdOut else output.stdErr
        except TimeoutError:
            try:
                self.interrupt()
                observation += "\nEXECUTION TIMED OUT"
            except RuntimeError as e:
                observation += "\nEXECUTION TIMED OUT AND INTERRUPT FAILED. RESTARTING PROCESS."
                info["exit_status"] = "early_exit"
                self.logger.warning(f"Failed to interrupt container: {e}\nRESTARTING PROCESS.")
                return observation, 0, True, info
        except RuntimeError as e:
            observation += "\nCOMMAND FAILED TO EXECUTE. RESTARTING PROCESS."
            info["exit_status"] = "early_exit"
            self.logger.warning(f"Failed to execute command: {e}\nRESTARTING PROCESS.")
            return observation, 0, True, info
        except BrokenPipeError as e:
            observation += "\nBROKEN PIPE ERROR. RESTARTING PROCESS."
            info["exit_status"] = "early_exit"
            self.logger.error(f"Broken pipe error: {e}\nRESTARTING PROCESS.")
            return observation, 0, True, info
        except Exception:
            observation += "\nEXECUTION FAILED OR COMMAND MALFORMED"
            self.logger.exception("Unknown exception")

        # Record submission and end episode if `submit` keyword found
        submission = self.get_submission(observation)
        if submission is not None:
            self.logger.info(f"Found submission: {submission}")
            info["exit_status"] = "submitted"
            info["submission"] = submission if submission.strip() != "" else None
            observation = submission if submission.strip() != "" else None
            return observation, 0, True, info
        return observation, 0, False, info

    def close(self) -> None:
        """
        Handle environment shutdown
        """
        self.logger.info("Beginning environment shutdown...")

        try:
            self.hide.delete_project(self.project)
        except KeyboardInterrupt:
            raise
        except:
            self.logger.warning("Errors when exiting container", exc_info=True)

        for hook in self.hooks:
            hook.on_close()

    # MARK: Helper functions #

    def _reset_container(self) -> None:
        # if self.container is not None:
        #     try:
        #         self.container.terminate()
        #     except KeyboardInterrupt:
        #         raise
        #     except:
        #         self.logger.warning("Failed to terminate container", exc_info=True)
        #     else:
        #         self.logger.debug("Terminated container")
        # self._init_container()
        self._init_scripts()

    # @staticmethod
    # def _get_container_name(image_name: str) -> str:
    #     """Return name of container"""
    #     process_id = str(os.getpid())
    #     current_time = str(datetime.datetime.now())
    #     unique_string = current_time + process_id
    #     hash_object = hashlib.sha256(unique_string.encode())
    #     image_name_sanitized = image_name.replace("/", "-")
    #     image_name_sanitized = image_name_sanitized.replace(":", "-")
    #     return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    # def _init_container(self, cached_image: str | None = None) -> None:
    #     """
    #     Handles container initialization. Defines container name and creates it.
    #     If cached_image is provided, it will use that image name instead of the default.
    #     """
    #     image_name = self.image_name
    #     if cached_image is not None:
    #         image_name = cached_image
    #         self.logger.info(f"Using cached image: {image_name}")
    #     if self.persistent:
    #         assert self.container_name is not None
    #     else:
    #         # Make sure that we get a new container name just in case removing didn't work.
    #         # Might be a fix for https://github.com/princeton-nlp/SWE-agent/issues/451
    #         self.container_name = self._get_container_name(image_name)
    #     self.container, self.parent_pids = get_container(self.container_name, image_name, persistent=self.persistent)
    #     try:
    #         client = docker.from_env(timeout=600)
    #     except docker.errors.DockerException as e:
    #         if "Error while fetching server API version" in str(e):
    #             msg = "Docker is not running. Please start Docker and try again."
    #         else:
    #             msg = "Unknown docker exception occurred. Are you sure docker is running?"
    #         raise RuntimeError(msg) from e
    #     t0 = time.time()
    #     self.container_obj = None
    #     while time.time() - t0 < 60:
    #         try:
    #             self.container_obj = client.containers.get(self.container_name)
    #         except docker.errors.NotFound:
    #             self.logger.debug("Couldn't find container. Let's wait and retry.")
    #             time.sleep(1)
    #         else:
    #             break
    #     else:
    #         print(f"{self.persistent=}")
    #         available_containers = client.containers.list(all=True)
    #         available_containers_info = json.dumps([str(c.attrs) for c in available_containers], indent=2)
    #         print(available_containers_info)
    #         msg = "Failed to get container object."
    #         raise RuntimeError(msg)
    #     self.logger.info("🌱 Environment Initialized")

    def _init_scripts(self):
        """
        Initialize custom commands within container
        """
        self.communicate_with_handling(
            "source /root/.bashrc",
            error_msg="Failed to source .bashrc",
        )
        self.communicate_with_handling(
            "mkdir -p /root/commands",
            error_msg="Failed to create commands directory",
        )
        self.communicate_with_handling(
            "touch /root/commands/__init__.py",
            error_msg="Failed to create __init__.py",
        )
        self.communicate_with_handling(
            "export PATH=$PATH:/root/commands",
            error_msg="Failed to add commands directory to PATH",
        )

    def _communicate(
        self,
        input: str,
        # timeout_duration=25,
    ) -> TaskResult:
        return self.hide.run_task(project_id=self.project.id, command=input)

    def _check_syntax(self, input: str) -> tuple[TaskResult, bool]:
        """
        Saves environment variables to file
        """
        output = self._communicate(f"/bin/bash -n <<'EOF'\n{input}\nEOF\n")
        return output, output.exitCode == 0

    def communicate(
        self,
        input: str,
        timeout_duration=25,
    ) -> TaskResult:
        """
        Sends input to container and returns output

        Args:
            input: input to send to container

        Returns:
            output: output from container
        """
        if input.strip() != "exit":
            output, valid = self._check_syntax(input)
            if not valid:
                return output  # shows syntax errors
            output = self._communicate(
                input,
            )
            return output
        else:
            self.hide.delete_project(self.project)
            return TaskResult(exitCode=0, stdOut="", stdErr="")

    def communicate_with_handling(self, input: str, error_msg: str, timeout_duration=25) -> str:
        """
        Wrapper for communicate function that raises error if return code is non-zero

        Args:
            input: input to send to container
            error_msg: error message to raise if return code is non-zero
            timeout_duration: duration to wait for output

        Returns:
            output: output from container
        """
        output = self.communicate(input, timeout_duration=timeout_duration)
        logs = f"stdout:\n{output.stdOut}\n\nstderr:\n{output.stdErr}"
        if output.exitCode != 0:
            self.logger.error(f"{error_msg}: {logs}")
            self.close()
            msg = f"{error_msg}: {logs}"
            raise RuntimeError(msg)
        return logs

    def get_available_actions(self) -> list[str]:
        """
        Returns list of available actions in current environment state

        Currently not in use.
        """
        return []

    def get_pids(self, all_pids=False) -> list[list[str]]:
        """
        Gets list of processes running inside docker container

        Args:
            all_pids: whether to return all pids, or whether to exclude ps
                and parent PIDs

        Returns:
            list of PIDs
        """
        pids = self.hide.run_task(project_id=self.project.id, command="ps -eo pid,comm --no-headers").stdOut.split("\n")
        pids = [x.split() for x in pids if x]
        if not all_pids:
            pids = [x for x in pids if x[1] != "ps" and x[0] not in self.parent_pids]
        return pids

    def get_submission(self, output: str) -> str:
        """
        Function for extracting diff patch submission at the end of an episode.

        Args:
            output: `submit` observation

        Returns:
            submission: diff patch submission
        """
        pattern = r"\<\<SUBMISSION\|\|(.*)\|\|SUBMISSION\>\>"
        match = re.search(pattern, output, re.DOTALL)
        if match is None:
            return None
        return match.group(1)

    def add_commands(self, commands: list[dict]) -> None:
        """
        Adds custom commands to container
        """
        for command in commands:
            name = command["name"]
            contents = command["contents"]
            self.copy_file_to_container(contents, f"/root/commands/{name}")
            if command["type"] == "source_file":
                self.communicate_with_handling(
                    f"source /root/commands/{name}",
                    error_msg=(
                        f"Failed to source {name}. If you meant to make a script,"
                        " start the file with a shebang (e.g. #!/usr/bin/env python)."
                    ),
                )
            elif command["type"] == "script":
                self.communicate_with_handling(
                    f"chmod +x /root/commands/{name}",
                    error_msg=f"Failed to chmod {name}",
                )
            elif command["type"] == "utility":
                # nothing to do for utility scripts
                pass
            else:
                msg = f"Invalid command type: {command['type']}"
                raise ValueError(msg)

    def interrupt(self):
        """
        Send interrupt signal to container and exhaust stdout buffer with a communicate call
        """
        pids = self.get_pids()
        for pid, cmd in pids:
            if pid not in self.parent_pids and cmd != "ps":
                self.hide.run_task(project_id=self.project.id, command=f"kill -9 {pid}")
        try:
            output = self.communicate(input="echo 'interrupted'", timeout_duration=5)
            assert output.stdOut.strip().endswith("interrupted"), "container health check failed"
        except TimeoutError:
            msg = "Failed to interrupt container"
            raise RuntimeError(msg)

    def open_pr(self, *, trajectory, _dry_run: bool = False):
        """Create PR to repository

        Args:
            trajectory: Trajectory of actions taken by the agent
            _dry_run: Whether to actually push anything or just simulate it
        """
        self.logger.info("Opening PR")
        # TODO: have better way of handling this
        # Adding random string suffix to avoid name conflicts if we had a previously failed run
        issue_url = self.args.data_path
        try:
            issue = get_gh_issue_data(issue_url, token=self._github_token)
        except InvalidGithubURL as e:
            msg = "Data path must be a github issue URL if --open_pr is set."
            raise ValueError(msg) from e
        branch_name = f"swe-agent-fix-#{issue.number}-" + str(random.random())[2:10]

        self.communicate_with_handling(
            input="rm -f model.patch",
            error_msg="Failed to remove model patch",
            timeout_duration=10,
        )
        self.communicate_with_handling(
            input=f"git checkout -b {branch_name}",
            error_msg="Failed to switch to new branch",
            timeout_duration=10,
        )
        self.communicate_with_handling(
            input="git add .",
            error_msg="Failed to add commits",
            timeout_duration=10,
        )
        dry_run_flag = "--allow-empty" if _dry_run else ""
        self.communicate_with_handling(
            input=f"git commit -m 'Fix: {issue.title}' -m 'Closes #{issue.number}' {dry_run_flag}",
            error_msg="Failed to commit changes",
            timeout_duration=10,
        )

        owner, repo, _ = parse_gh_issue_url(issue_url)
        # If `--repo_path` was specified with a different github URL, then the record will contain
        # the forking user
        assert self.record is not None
        if self.record["repo_type"] != "github":
            # We already validated that `--data_path` is a github issue URL
            # so this is the only case where we can reach here
            msg = "--repo_path must point to a github URL if --open_pr is set"
            raise ValueError(msg)
        forker, _ = self.record["repo"].split("/")
        head = branch_name
        remote = "origin"
        if forker != owner:
            head = f"{forker}:{branch_name}"
            token_prefix = ""
            if self._github_token:
                token_prefix = f"{self._github_token}@"
            fork_url = f"https://{token_prefix}github.com/{forker}/{repo}.git"
            self.logger.debug(f"Using fork: {fork_url}")
            self.communicate_with_handling(
                input=f"git remote add fork {fork_url}",
                error_msg="Failed to create new git remote",
                timeout_duration=10,
            )
            remote = "fork"
        dry_run_prefix = "echo " if _dry_run else ""
        self.communicate_with_handling(
            input=f"{dry_run_prefix} git push {remote} {branch_name}",
            error_msg=(
                "Failed to push branch to remote. Please check your token and permissions. "
                "You might want to push to a fork with the push_gh_repo_url option."
            ),
            timeout_duration=10,
        )
        body = (
            f"This is a PR opened by AI tool [SWE Agent](https://github.com/princeton-nlp/SWE-agent/) "
            f"to close [#{issue.number}]({issue_url}) ({issue.title}).\n\nCloses #{issue.number}."
        )
        body += "\n\n" + format_trajectory_markdown(trajectory)
        api = GhApi(token=self._github_token)
        if not _dry_run:
            pr_info = api.pulls.create(
                owner=owner,
                repo=repo,
                title=f"SWE-agent[bot] PR to fix: {issue.title}",
                head=head,
                base="main",
                body=body,
                draft=True,
            )
            self.logger.info(
                f"🎉 PR created as a draft at {pr_info.html_url}. Please review it carefully, push "
                "any required changes onto the branch and then click "
                "'Ready for Review' to bring it to the attention of the maintainers.",
            )

    # TODO: Why not just use copy_anything_to_container?
    def copy_file_to_container(self, contents: str, container_path: str) -> None:
        """
        Copies a given string into a Docker container at a specified path.

        Args:
            env: Development environment.
            contents: The string to copy into the container.
            container_path: The path inside the container where the string should be copied to.

        Returns:
            None
        """
        try:
            self.hide.create_file(self.project.id, container_path, contents)
        except Exception as e:
            self.logger.error(f"An error occurred while copying file to container: {e}")
            self.logger.error(traceback.format_exc())
