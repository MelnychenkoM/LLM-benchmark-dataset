import os
import wandb
import pandas as pd
from typing import List


import constants


class BenchmarkDatasetArtifact:
    ARTIFACT_NAME = "benchmark_dataset"
    ARTIFACT_TYPE = "benchmark_dataset"
    CREATE_ARTIFACT_JOB_TYPE = "create_benchmark_dataset"
    UPDATE_ARTIFACT_JOB_TYPE = "update_benchmark_dataset"

    ARTIFACT_DIR = [
        {
            'name': 'articles',
            'type': 'dir',
            'path': 'articles'
        },
        {
            'name': 'amino_acids',
            'type': 'file',
            'path': 'amino_acids.xlsx'
        },
        {
            'name': 'pockets',
            'type': 'file',
            'path': 'pockets.xlsx'
        }
    ]

    def __init__(
        self,
        folder: str,
        project: str,
        version: str = None,
        automatically_update_version: bool = False,
        entity: str = None,
        artifact_description: str = '',
        artifact_tags: List[str] = None,
        artifact_metadata: dict = None
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
        self.folder = folder
        self.project = project
        self.version = version
        self.automatically_update_version = automatically_update_version
        self.entity = constants.WANDB_ENTITY if entity is None else entity
        self.artifact_description = artifact_description
        self.artifact_tags = artifact_tags if artifact_tags else []
        self.artifact_metadata = artifact_metadata if artifact_metadata else {}

        if version is None and automatically_update_version:
            self.update_artifact()

        if version is None:
            self.pockets_df, self.amino_acids_df = self._load_tables()
        else:
            self._download_artifact(version)
            self.pockets_df, self.amino_acids_df = self._load_tables()

    def prepare_article_for_evaluation(
        self,
        article: str,
        reader,
    ) -> dict:
        pockets_in_article_df = self.pockets_df.query(
            f'article_pdf_name == "{article}"'
        )
        text = reader(self._article_path(article))

        merged_df = pd.merge(pockets_in_article_df, self.amino_acids_df,
                             on=['article_pdf_name', 'target', 'pocket_id'])

        # df with columns 'target', 'pocket_id', 'pocket_description'
        # and 'amino_acid' (where amino acids are aggregated into a list)
        grouped = merged_df.groupby(
            ['target', 'pocket_id', 'pocket_description'],
            group_keys=True
        )['amino_acid'].apply(list).reset_index()

        # Convert the grouped DataFrame into a list of dictionaries
        annotations = {}
        if grouped.empty:
            # this should be the case for articles which do not describe binding pockets
            target = pockets_in_article_df['target'].iloc[0]
            print(f"WARNING! Article {article} has no pockets for target {target}!")
            annotations[target] = {
                # list of strings (chunks) with the article content to pass directly into the chain
                "text": text,
                # the dict with keys 'target', 'pocket_id', 'pocket_description' and 'amino_acids'
                "annotated_pockets": {},
                # subset of pockets_df with pockets for the target in the article
                "annotated_pockets_extra_data": {},
                # subset of amino_acids_df with amino acids for the target in the article
                "annotated_amino_acids_extra_data": {},
            }
        else:
            for target, target_group in grouped.groupby('target'):
                annotated_pockets = target_group.to_dict('records')
                for pocket in annotated_pockets:
                    # Rename 'amino_acid' key to 'amino_acids'
                    pocket['amino_acids'] = pocket.pop('amino_acid')

                pockets_extra_data = self._format_pockets_extra_data(
                    pockets_in_article_df, target
                )

                amino_acids_extra_data = self._format_amino_acids_extra_data(
                    pockets_in_article_df, self.amino_acids_df, target
                )

                annotations[target] = {
                    # list of strings (chunks) with the article content to pass directly into the chain
                    "text": text,
                    # the dict with keys 'target', 'pocket_id', 'pocket_description' and 'amino_acids'
                    "annotated_pockets": annotated_pockets,
                    # subset of pockets_df with pockets for the target in the article
                    "annotated_pockets_extra_data": pockets_extra_data,
                    # subset of amino_acids_df with amino acids for the target in the article
                    "annotated_amino_acids_extra_data": amino_acids_extra_data,
                }

        return annotations

    def _format_pockets_extra_data(
        self,
        pockets_in_article_df: pd.DataFrame,
        target: str
    ) -> dict[str, dict]:
        """
            Uses pockets_df to extract data for additional columns in the logged pockets table.
            Returns a dict with keys corresponding to pocket ids for `target` mentioned in the 
            article and values being dicts with extra data to add to the pockets table.
        """
        pockets_extra_data = {}
        grouped_by_pocket_id_dfs = pockets_in_article_df.query(f'target == "{target}"') \
            .drop(columns=['article_pdf_name', 'folder_name', 'pocket_description']) \
            .groupby('pocket_id')

        for pocket_id, pocket_df in grouped_by_pocket_id_dfs:
            pocket_extra_data = pocket_df.to_dict('records')
            if len(pocket_extra_data) > 1:
                raise ValueError(
                    f"More than one pocket with the same id: {pocket_id}"
                )

            pocket_extra_data = pocket_extra_data[0]
            pocket_extra_data.pop('pocket_id')
            pockets_extra_data[pocket_id] = pocket_extra_data

        return pockets_extra_data

    def _format_amino_acids_extra_data(
        self,
        pockets_in_article_df: pd.DataFrame,
        amino_acids_df: pd.DataFrame,
        target: str
    ) -> dict[str, dict]:
        # TODO
        amino_acids_extra_data = {}

        amino_acids_for_target_in_article_df = \
            pd.merge(
                pockets_in_article_df.query(f'target == "{target}"')[
                    ['article_pdf_name', 'target', 'pocket_id']
                ],
                amino_acids_df,
                on=['article_pdf_name', 'target', 'pocket_id']
            )

        grouped_by_amino_acid_dfs = amino_acids_for_target_in_article_df \
            .groupby(['pocket_id', 'amino_acid'])

        for (pocket_id, amino_acid_id), amino_acid_df in grouped_by_amino_acid_dfs:
            amino_acid_extra_data = amino_acid_df.to_dict('records')
            if len(amino_acid_extra_data) > 1:
                raise ValueError(
                    f"More than one amino acid with the same id: {amino_acid_id}"
                )

            amino_acid_extra_data = amino_acid_extra_data[0]
            amino_acid_extra_data.pop('amino_acid')
            amino_acid_extra_data.pop('article_pdf_name')
            pocket_id = amino_acid_extra_data.pop('pocket_id')
            amino_acids_extra_data[f'{pocket_id}___{amino_acid_id}'] = amino_acid_extra_data

        return amino_acids_extra_data

    def update_artifact(self, version: str = "latest") -> None:
        if self._check_if_artifact_exists(version):
            self._update_artifact(version)
        else:
            self._create_artifact()

    def _artifact_full_name(self, version: str) -> str:
        return f'{self.entity}/{self.project}/{self.ARTIFACT_NAME}:{version}'

    def _update_artifact(self, version) -> None:
        if self._are_local_and_remote_artifacts_different(version):
            print(
                f"{self.__class__.__name__} | Updating artifact {self.ARTIFACT_NAME}:{version}"
            )

            config = {
                "project": self.project,
                "entity": self.entity,
                "job_type": self.UPDATE_ARTIFACT_JOB_TYPE,
                "name": f'Update artifact "{self.ARTIFACT_NAME}"',
                "tags": self.artifact_tags,
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

                self._fill_artifact(draft_artifact)
                run.log_artifact(draft_artifact)
                draft_artifact.wait()

    def _check_if_artifact_exists(self, version: str = 'latest') -> bool:
        artifact_name = f"{self.entity}/{self.project}/{self.ARTIFACT_NAME}:{version}"

        try:
            wandb.Api().artifact(artifact_name)
            print(
                f"{self.__class__.__name__} | Artifact {artifact_name} already exists")

            return True
        except wandb.errors.CommError:
            print(
                f"{self.__class__.__name__} | Artifact {artifact_name} doesn't exist yet")
            return False

    def _are_local_and_remote_artifacts_different(self, version) -> bool:
        local_artifact = wandb.Artifact(
            name=self.ARTIFACT_NAME,
            type=self.ARTIFACT_TYPE
        )
        self._fill_artifact(local_artifact)

        remote_artifact = wandb.Api().artifact(self._artifact_full_name(version))

        are_different = remote_artifact.digest != local_artifact.digest

        print(
            f"{self.__class__.__name__} | Local and remote artifacts are different: {are_different}"
        )

        return are_different

    def _create_artifact(self) -> None:
        print(f"{self.__class__.__name__} | Creating artifact {self.ARTIFACT_NAME}")

        config = {
            "project": self.project,
            "entity": self.entity,
            "job_type": self.CREATE_ARTIFACT_JOB_TYPE,
            "name": f'Create artifact "{self.ARTIFACT_NAME}"',
            "tags": self.artifact_tags,
        }

        with wandb.init(**config) as run:
            artifact = wandb.Artifact(
                name=self.ARTIFACT_NAME,
                type=self.ARTIFACT_TYPE,
                description=self.artifact_description,
                metadata=self.artifact_metadata
            )

            self._fill_artifact(artifact)
            run.log_artifact(artifact)
            artifact.wait()

    def _fill_artifact(self, artifact: wandb.Artifact) -> None:
        for file_config in self.ARTIFACT_DIR:
            item_path = os.path.join(self.folder, file_config['path'])

            if file_config['type'] == 'dir':
                artifact.add_dir(item_path)
            else:
                artifact.add_file(item_path)

    def _download_artifact(self, version) -> None:

        should_download = True

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        elif len(os.listdir(self.folder)) > 0:
            if self._are_local_and_remote_artifacts_different(version):
                raise ValueError(
                    f"Folder {self.folder} is not empty! Can not load artifact there.")
            else:
                should_download = False

        if should_download:
            artifact = wandb.Api().artifact(self._artifact_full_name(version))
            artifact.download(self.folder)

    def _load_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        pocket_tables_file_config = [
            fc for fc in self.ARTIFACT_DIR if fc['name'] == 'pockets'
        ][0]
        pockets_df = pd.read_excel(
            os.path.join(self.folder, pocket_tables_file_config['path'])
        )

        amino_acids_file_config = [
            fc for fc in self.ARTIFACT_DIR if fc['name'] == 'amino_acids'
        ][0]
        amino_acids_df = pd.read_excel(
            os.path.join(self.folder, amino_acids_file_config['path'])
        )

        return pockets_df, amino_acids_df

    def _article_path(self, article: str) -> str:
        articles_root_folder = [
            item['path'] for item in self.ARTIFACT_DIR
            if item['name'] == 'articles'
        ][0]

        target_folder = self.pockets_df.query(
            f'article_pdf_name == "{article}"').iloc[0]['folder_name']

        return os.path.join(self.folder, articles_root_folder, target_folder, article)
