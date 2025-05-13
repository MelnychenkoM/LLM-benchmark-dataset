import os
import tempfile

from abc import ABC, abstractmethod
from typing import List, Union, TypedDict

import constants


class ArtifactDirFile(TypedDict):
    """
        prompt_name - same as defined in the prompts.templates.py, e.g. SYSTEM_MESSAGE
        prompt - the prompt string
        subfolders - the subfolders where the prompt will be saved into
        file_name - the txt file name (without extension) where the prompt will be saved into
    """
    prompt_name: str
    prompt: str
    subfolders: List[str]
    file_name: str


class PromptBuilder(ABC):
    ARTIFACT_TYPE = "prompts"
    CREATE_ARTIFACT_JOB_TYPE = "create_prompts"
    UPDATE_ARTIFACT_JOB_TYPE = "update_prompts"

    @property
    @abstractmethod
    def ARTIFACT_NAME(self) -> str:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def ARTIFACT_DIR(self) -> list[ArtifactDirFile]:
        raise NotImplementedError


    def __init__(
        self,
        project: str = None,
        version: str = None,
        automatically_update_version: bool = False,
        entity: str = None,
        prompt_artifact_description: str = '',
        prompt_artifact_tags: List[str] = None,
        prompt_artifact_metadata: dict = None
    ) -> None:
        """_summary_

        Args:
            project (str): wandb project name.
            version (str, optional): load and use prompts from w&b 
                artifact of the given version.
                Defaults to None - use the prompts from local file.
            automatically_update_version (bool): automatically update prompts artifact
                version with the current prompt values from local files.
                Ignored if version is not None.
            entity (str, optional): wandb entity name. 
                Defaults to None - WANDB_ENTITY from constants is used.
        """
        self.project = project
        self.version = version
        self.automatically_update_version = automatically_update_version
        self.entity = constants.WANDB_ENTITY if entity is None else entity
        self.prompt_artifact_description = prompt_artifact_description
        self.prompt_artifact_tags = prompt_artifact_tags if prompt_artifact_tags else []
        self.prompt_artifact_metadata = prompt_artifact_metadata if prompt_artifact_metadata else {}

        if self.project is not None:
            import wandb

        if version is None and automatically_update_version:
            self.update_artifact()

        if version is None:
            self.prompts = {prompt_config["prompt_name"]: prompt_config["prompt"]
                            for prompt_config in self.ARTIFACT_DIR}
        else:
            self.prompts = self._load_prompts_from_artifact(
                self.entity, project, version
            )

    def update_artifact(self, version: str = "latest") -> None:
        if self._check_if_artifact_exists(version):
            self._update_artifact(version)
        else:
            self._create_artifact()

    def build_prompt(self, prompt_name: str) -> str:
        return self.prompts[prompt_name]

    def _update_artifact(self, version) -> None:
        if self._are_local_and_remote_artifacts_different(version):
            print(f"PromptBuilder | Updating artifact {self.ARTIFACT_NAME}:{version}")

            config = {
                "project": self.project,
                "entity": self.entity,
                "job_type": self.UPDATE_ARTIFACT_JOB_TYPE,
                "name": f'Update artifact "{self.ARTIFACT_NAME}"',
                "tags": self.prompt_artifact_tags,
            }

            with wandb.init(**config) as run:
                artifact = run.use_artifact(
                    f'{self.ARTIFACT_NAME}:{version}',
                    type=self.ARTIFACT_TYPE
                )

                # for simplicity, to relog artifact dir let's
                # clear its content and readd the whole dir
                draft_artifact = artifact.new_draft()
                for f in artifact.files():
                    draft_artifact.remove(f.name)

                with tempfile.TemporaryDirectory() as artifact_dir:
                    self._fill_artifact_dir(artifact_dir)
                    draft_artifact.add_dir(artifact_dir)
                    run.log_artifact(draft_artifact)
                    draft_artifact.wait()

    def _check_if_artifact_exists(self, version: str = 'latest') -> bool:
        artifact_name = f"{self.entity}/{self.project}/{self.ARTIFACT_NAME}:{version}"

        try:
            wandb.Api().artifact(artifact_name)
            print(f"PromptBuilder | Artifact {artifact_name} already exists")

            return True
        except wandb.errors.CommError:
            print(f"PromptBuilder | Artifact {artifact_name} doesn't already exist")
            return False

    def _are_local_and_remote_artifacts_different(self, version) -> bool:
        with tempfile.TemporaryDirectory() as local_artifact_dir:
            self._fill_artifact_dir(local_artifact_dir)
            local_artifact = wandb.Artifact(
                name=self.ARTIFACT_NAME,
                type=self.ARTIFACT_TYPE
            )
            local_artifact.add_dir(local_artifact_dir)

            remote_artifact = wandb.Api().artifact(
                f'{self.entity}/{self.project}/{self.ARTIFACT_NAME}:{version}'
            )

            are_different = remote_artifact.digest != local_artifact.digest

            print(f"PromptBuilder | Local and remote artifacts are different: {are_different}")

        return are_different

    def _create_artifact(self) -> None:
        print(f"PromptBuilder | Creating artifact {self.ARTIFACT_NAME}")

        config = {
            "project": self.project,
            "entity": self.entity,
            "job_type": self.CREATE_ARTIFACT_JOB_TYPE,
            "name": f'Create artifact "{self.ARTIFACT_NAME}"',
            "tags": self.prompt_artifact_tags,
        }

        with wandb.init(**config) as run:
            artifact = wandb.Artifact(
                name=self.ARTIFACT_NAME,
                type=self.ARTIFACT_TYPE,
                description=self.prompt_artifact_description,
                metadata=self.prompt_artifact_metadata
            )

            with tempfile.TemporaryDirectory() as artifact_dir:
                self._fill_artifact_dir(artifact_dir)
                artifact.add_dir(artifact_dir)
                run.log_artifact(artifact)
                artifact.wait()

    def _fill_artifact_dir(self, artifact_dir: str) -> None:
        for item in self.ARTIFACT_DIR:
            self._save_prompt_to_file(
                item['prompt'],
                artifact_dir,
                item['file_name'],
                item['subfolders']
            )

    def _save_prompt_to_file(
        self,
        prompt: str,
        temp_dir: str,
        file_name: str,
        subfolders: list[str] = None
    ) -> None:
        subfolders = subfolders if subfolders else []
        path = os.path.join(temp_dir, *subfolders, f'{file_name}.txt')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            f.write(prompt)

    def _load_prompts_from_artifact(self, entity, project, version) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_name = f"{entity}/{project}/{self.ARTIFACT_NAME}:{version}"
            artifact = wandb.Api().artifact(artifact_name)
            artifact.download(temp_dir)

            prompts = {}
            for prompt_config in self.ARTIFACT_DIR:
                prompt_path = os.path.join(
                    temp_dir, *
                    prompt_config['subfolders'], f"{prompt_config['file_name']}.txt"
                )

                with open(prompt_path, "r") as f:
                    prompts[prompt_config['prompt_name']] = f.read()

        return prompts
