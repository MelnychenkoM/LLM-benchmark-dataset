from wandb.sdk.wandb_run import Run as WandbRun

from utils.wandb.tracer_callback import TracerCallback
from .prompt_buider import PromptBuilder


class PromptsUseArtifactCallback(TracerCallback):
    def __init__(self, prompts_builders: list[PromptBuilder]):
        self.prompts_builders = prompts_builders

    def on_finish(self, wandb_run: WandbRun) -> None:
        for prompts_builder in self.prompts_builders:
            if not self._should_ignore(prompts_builder):
                wandb_run.use_artifact(
                    self._artifact_name(prompts_builder),
                    type=self._artifact_type(prompts_builder)
                )

    def _artifact_name(self, prompts_builder: PromptBuilder) -> str:
        version = prompts_builder.version if prompts_builder.version else "latest"
        return f'{prompts_builder.ARTIFACT_NAME}:{version}'

    def _artifact_type(self, prompts_builder: PromptBuilder) -> str:
        return prompts_builder.ARTIFACT_TYPE

    def _should_ignore(self, prompts_builder: PromptBuilder) -> bool:
        return prompts_builder.version is None and prompts_builder.automatically_update_version == False
