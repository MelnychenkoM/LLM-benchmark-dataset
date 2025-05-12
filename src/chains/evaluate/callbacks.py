import wandb
import json
import pandas as pd
import numpy as np
from typing import Union
from langchain_core.tracers.schemas import Run as LangchainRun
from wandb.sdk.wandb_run import Run as WandbRun

from utils.common import handle_zero_devision
from utils.wandb.tracer_callback import TracerCallback
from .metrics import (
    recall, 
    intersection_over_union, 
    percentage_of_false_positives, 
    calculate_pockets_accuracy,
    calculate_pocket_number_accuracy
)

class PocketTableCallback(TracerCallback):
    def __init__(self):
        self.columns: list[str] = None
        self.data: list[list] = None

    def on_log_trace_from_run_end(
        self,
        wandb_run: WandbRun,
        langchain_run: LangchainRun
    ) -> None:
        annotated_pockets = langchain_run.outputs["annotated_pockets"]
        extracted_pockets = langchain_run.outputs["extracted_pockets"]
        annotated2extracted_pocket_matches = \
            langchain_run.outputs["annotated2extracted_pocket_matches"]
        matched_amino_acids = langchain_run.outputs["matched_amino_acids"]
        relevance = langchain_run.outputs.get("relevance", [])

        constant_extra_columns = langchain_run.extra['metadata'].get('general')
        annotated_extra_data = langchain_run.extra['metadata'] \
            .get(self.__class__.__name__, {}) \
            .get('annotated_extra_data')
        
        if relevance:
            constant_extra_columns['is_relevant'] = relevance['is_relevant']
            constant_extra_columns['reasoning'] = relevance['reasoning']
        try:
            columns, rows = _prepare_pockets_wandb_table(
                annotated_pockets,
                extracted_pockets,
                annotated2extracted_pocket_matches,
                matched_amino_acids,
                extra_constant_columns=constant_extra_columns,
                annotated_extra_data=annotated_extra_data,
            )
        except Exception as e:
            raise ValueError(
                f"PocketTableCallback | Error preparing wandb table: {e}"
            )
        
        if self.columns is None:
            self.columns = columns
            self.data = rows
        else:
            if columns != self.columns:
                self.data = _merge_dataframes(self.data, self.columns, rows, columns)
            else:
                self.data.extend(rows)

    def on_finish(self, wandb_run: WandbRun) -> None:
        if self.columns and self.data:
            summary_metrics = self._summary_metrics()

            wandb_run.log({
                "pockets": wandb.Table(data=self.data, columns=self.columns),
                **summary_metrics
            }, commit=False)

    def _summary_metrics(self) -> dict:
        df = pd.DataFrame(self.data, columns=self.columns)
        
        if 'annotated_pocket_id' in df.columns:
            n_annotated_pockets = df.dropna(subset=['annotated_pocket_id']) \
                .groupby(['article_pdf_name', 'target', 'annotated_pocket_id']).size().reset_index().shape[0]
            n_matches = df.query("status == 'matched'") \
                .groupby(['annotated_pocket_id'])['status'].count().sum()
            n_matched_annotated_pockets = df.query("status == 'matched'") \
                .groupby(['article_pdf_name', 'target', 'annotated_pocket_id']).size().reset_index().shape[0]
        else:
            n_annotated_pockets = 0
            n_matches = 0
            n_matched_annotated_pockets = 0
            
        if 'extracted_pocket_id' in df.columns:
            n_extracted_pockets = df.dropna(subset=['extracted_pocket_id']) \
                .groupby(['article_pdf_name', 'target', 'extracted_pocket_id']).size().reset_index().shape[0]
            n_matched_extracted_pockets = df.query("status == 'matched'") \
                .groupby(['article_pdf_name', 'target', 'extracted_pocket_id']).size().reset_index().shape[0]
        else:
            n_extracted_pockets = 0
            n_matched_extracted_pockets = 0
        
        n_fake_pockets = df.query("status == 'fake'").shape[0]

        accuracy_pockets_match = calculate_pocket_number_accuracy(df)

        return {
            "accuracy_pockets_match": accuracy_pockets_match,
            "n_annotated_pockets": n_annotated_pockets,
            "n_extracted_pockets": n_extracted_pockets,
            "n_matches_pockets": n_matches,
            "n_matched_annotated_pockets": n_matched_annotated_pockets,
            "n_matched_extracted_pockets": n_matched_extracted_pockets,
            "recall_pockets": handle_zero_devision(n_matched_annotated_pockets, n_annotated_pockets),
            "mean_n_extracted_per_annotated_pockets": handle_zero_devision(n_matches, n_matched_annotated_pockets),
            "mean_n_annotated_per_extracted_pockets": handle_zero_devision(n_matches, n_matched_extracted_pockets),
            "n_fake_pockets": n_fake_pockets,
            "false_positives_rate_pockets": handle_zero_devision(n_fake_pockets, n_extracted_pockets),
            "true_negatives_rate_pockets": 1 - handle_zero_devision(n_fake_pockets, n_extracted_pockets)
        }


class AminoAcidsTableCallback(TracerCallback):
    def __init__(self):
        self.columns: list[str] = None
        self.data: list[list] = None

    def on_log_trace_from_run_end(
        self,
        wandb_run: WandbRun,
        langchain_run: LangchainRun
    ) -> None:
        annotated_pockets = langchain_run.outputs["annotated_pockets"]
        extracted_pockets = langchain_run.outputs["extracted_pockets"]
        annotated2extracted_pocket_matches = \
            langchain_run.outputs["annotated2extracted_pocket_matches"]
        matched_amino_acids = langchain_run.outputs["matched_amino_acids"]

        constant_extra_columns = langchain_run.extra['metadata'].get('general')
        annotated_extra_data = langchain_run.extra['metadata'] \
            .get(self.__class__.__name__, {}) \
            .get('annotated_extra_data')

        try:
            columns, rows = _prepare_amino_acids_wandb_table(
                annotated_pockets,
                extracted_pockets,
                annotated2extracted_pocket_matches,
                matched_amino_acids,
                constant_extra_columns=constant_extra_columns,
                annotated_extra_data=annotated_extra_data
            )
        except Exception as e:
            raise ValueError(
                f"AminoAcidsTableCallback | Error preparing wandb table: {e}"
            )

        if self.columns is None:
            self.columns = columns
            self.data = rows
        else:
            if columns != self.columns:
                self.data = _merge_dataframes(self.data, self.columns, rows, columns)
            else:
                self.data.extend(rows)

    def on_finish(self, wandb_run: WandbRun) -> None:
        if self.columns and self.data:
            summary_metrics = self._summary_metrics()
            wandb_run.log({
                "amino_acids": wandb.Table(data=self.data, columns=self.columns),
                **summary_metrics
            }, commit=False)

    def _summary_metrics(self) -> dict:
        df = pd.DataFrame(self.data, columns=self.columns)

        n_annotated_amino_acids = df.dropna(subset=['annotated_pocket_id', 'annotated_amino_acid']) \
            .groupby(['article_pdf_name', 'target', 'annotated_pocket_id', 'annotated_amino_acid']).size().reset_index().shape[0]
        n_matched_amino_acids = df.query("status == 'matched'").dropna(
            subset=['annotated_pocket_id', 'annotated_amino_acid']
        ).groupby(['article_pdf_name', 'target', 'annotated_pocket_id', 'annotated_amino_acid']).size().reset_index().shape[0]
        recall = handle_zero_devision(
            n_matched_amino_acids, n_annotated_amino_acids
        )

        df_amino_acid_counts_per_pocket = df.dropna(
            subset=['annotated_pocket_id', 'annotated_amino_acid']
        ).groupby(['article_pdf_name', 'target', 'annotated_pocket_id'])['annotated_amino_acid'].count()
        df_matched_amino_acid_counts_per_pocket = df.dropna(
            subset=['annotated_pocket_id', 'annotated_amino_acid']
        ).query("status == 'matched'").groupby(
            ['article_pdf_name', 'target', 'annotated_pocket_id']
        )['annotated_amino_acid'].count()

        df_pocket_coverage = pd.concat(
            [df_amino_acid_counts_per_pocket,
                df_matched_amino_acid_counts_per_pocket], axis=1
        )
        df_pocket_coverage.columns = [
            'n_amino_acids', 'n_extracted_amino_acids']
        df_pocket_coverage.fillna(0, inplace=True)
        df_pocket_coverage['extracted_amino_acids_percentage'] = \
            np.where(
                df_pocket_coverage['n_amino_acids'] == 0,
                0,
                df_pocket_coverage['n_extracted_amino_acids'] /
            df_pocket_coverage['n_amino_acids']
        )

        mean_extracted_amino_acids_percentage = \
            df_pocket_coverage['extracted_amino_acids_percentage'].mean()

        n_fake_amino_acids = df.query("status == 'fake'").groupby(
            ['article_pdf_name', 'target',
                'extracted_pocket_id', 'extracted_amino_acid']
        ).size().reset_index().shape[0]
        n_extracted_amino_acids = df.dropna(
            subset=['extracted_pocket_id', 'extracted_amino_acid']
        ).groupby(['article_pdf_name', 'target', 'extracted_pocket_id', 'extracted_amino_acid']).size().reset_index().shape[0]

        false_positives_rate = handle_zero_devision(
            n_fake_amino_acids, n_extracted_amino_acids
        )
        precision = 1 - false_positives_rate

        metrics_for_extracted_pockets = self._metrics_for_extracted_pockets(df)
        metrics_for_unique_by_id_amino_acids = self._metrics_for_unique_by_id_amino_acids(df)

        return {
            "n_annotated_amino_acids": n_annotated_amino_acids,
            "n_extracted_amino_acids": n_extracted_amino_acids,
            "n_matched_amino_acids": n_matched_amino_acids,
            "recall_amino_acids": recall,
            "precision_amino_acids": precision,
            "f1_score_amino_acids": handle_zero_devision(2 * precision * recall, precision + recall),
            "mean_extracted_amino_acids_percentage": mean_extracted_amino_acids_percentage,
            "n_fake_amino_acids": n_fake_amino_acids,
            "false_positives_rate_amino_acids": false_positives_rate,
            "annotated_pockets_amino_acid_coverage": wandb.Table(dataframe=df_pocket_coverage.reset_index()),
            "metrics_for_unique_by_id_amino_acids": wandb.Table(dataframe=metrics_for_unique_by_id_amino_acids),
            "mean_amino_acids_precion_per_extracted_pocket": metrics_for_extracted_pockets["precision"].mean(),
            "mean_percentage_of_fake_amino_acids_per_extracted_pocket": metrics_for_extracted_pockets["false_positive_rate"].mean(),
            "extracted_pockets_amino_acids_quality": wandb.Table(dataframe=metrics_for_extracted_pockets)
        }
        
    def _metrics_for_unique_by_id_amino_acids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Unlike in the `_summary_metrics`, here the uniqueness of amino acids is not determined on per pocket basis.
        It is determined by the uniqueness of the amino acid itself for a given article and target.
        Thus, in groupby we need to group only by the 'article_pdf_name', 'target', 'annotated_amino_acid'.
        """
        n_annotated_amino_acids = df.dropna(subset=['annotated_pocket_id', 'annotated_amino_acid']) \
            .groupby(['article_pdf_name', 'target', 'annotated_amino_acid']).size().reset_index().shape[0]
        n_matched_amino_acids = df.query("status == 'matched'").dropna(
            subset=['annotated_pocket_id', 'annotated_amino_acid']
        ).groupby(['article_pdf_name', 'target', 'annotated_amino_acid']).size().reset_index().shape[0]
        n_unmatched_amino_acids = df.query("status == 'unmatched'").dropna(
            subset=['annotated_pocket_id', 'annotated_amino_acid']
        ).groupby(['article_pdf_name', 'target', 'annotated_amino_acid']).size().reset_index().shape[0]   
        n_fake_amino_acids = df.query("status == 'fake'").groupby(
            ['article_pdf_name', 'target', 'extracted_amino_acid']
        ).size().reset_index().shape[0]
        n_extracted_amino_acids = df.dropna(
            subset=['extracted_pocket_id', 'extracted_amino_acid']
        ).groupby(['article_pdf_name', 'target', 'extracted_amino_acid']).size().reset_index().shape[0]

        recall = handle_zero_devision(
            n_matched_amino_acids, n_annotated_amino_acids
        )
        false_positives_rate = handle_zero_devision(
            n_fake_amino_acids, n_extracted_amino_acids
        )
        precision = 1 - false_positives_rate
        f1_score = handle_zero_devision(2 * precision * recall, precision + recall)
        
        return pd.DataFrame({
            "n_annotated": [n_annotated_amino_acids],
            "n_annotated_matched": [n_matched_amino_acids],
            "n_annotated_unmatched": [n_unmatched_amino_acids],
            "n_extracted": [n_extracted_amino_acids],
            "n_extracted_fake": [n_fake_amino_acids],
            "recall": [recall],
            "precision": [precision],
            "f1_score": [f1_score]
        })
        
        

    def _metrics_for_extracted_pockets(self, df: pd.DataFrame) -> dict:
        matched_extracted_pocket_ids = df.query("status == 'matched'")[
            'extracted_pocket_id'].unique().tolist()

        df_amino_acid_counts_per_extracted_pocket = df.query(
            f"extracted_pocket_id in {matched_extracted_pocket_ids}"
        ).groupby([
            'article_pdf_name', 'target', 'extracted_pocket_id'
        ])['extracted_amino_acid'].nunique()

        df_matched_amino_acid_counts_per_extracted_pocket = df.query(
            f"extracted_pocket_id in {matched_extracted_pocket_ids} and status == 'matched'"
        ).groupby([
            'article_pdf_name', 'target', 'extracted_pocket_id'
        ])['extracted_amino_acid'].nunique()

        df_fake_amino_acid_counts_per_extracted_pocket = df.query(
            f"extracted_pocket_id in {matched_extracted_pocket_ids} and status == 'fake'"
        ).groupby([
            'article_pdf_name', 'target', 'extracted_pocket_id'
        ])['extracted_amino_acid'].nunique()

        df_extracted_pockets_stats = pd.concat(
            [df_amino_acid_counts_per_extracted_pocket,
             df_matched_amino_acid_counts_per_extracted_pocket,
             df_fake_amino_acid_counts_per_extracted_pocket],
            axis=1
        )
        df_extracted_pockets_stats.columns = [
            'n_amino_acids', 'n_matched_amino_acids', 'n_fake_amino_acids']
        df_extracted_pockets_stats.fillna(0, inplace=True)
        df_extracted_pockets_stats['precision'] = \
            np.where(
                df_extracted_pockets_stats['n_amino_acids'] == 0,
                0,
                df_extracted_pockets_stats['n_matched_amino_acids'] /
            df_extracted_pockets_stats['n_amino_acids']
        )
        df_extracted_pockets_stats['false_positive_rate'] = \
            np.where(
                df_extracted_pockets_stats['n_amino_acids'] == 0,
                0,
                df_extracted_pockets_stats['n_fake_amino_acids'] /
            df_extracted_pockets_stats['n_amino_acids']
        )
        df_extracted_pockets_stats.reset_index(inplace=True)

        return df_extracted_pockets_stats


def _prepare_pockets_wandb_table(
    annotated_pockets: list[dict],
    extracted_pockets: list[dict],
    annotated2extracted_pocket_matches: list[dict],
    matched_amino_acids: list[dict],
    extra_constant_columns: dict = None,
    annotated_extra_data: dict[str, dict] = None
) -> tuple[list[str], list[list]]:
    """ 
    Prepare a wandb table with the pockets and the metrics. 
    The expected minimal set of columns:
    - annotated_pocket_id
    - annotated_pocket_name
    - annotated_pocket_amino_acids
    - extracted_pocket_id
    - extracted_pocket_name
    - extracted_pocket_amino_acids
    - recall
    - iou
    - false_positives
    """
    extra_constant_columns = extra_constant_columns if extra_constant_columns else {}
    annotated_pockets_columns = list(annotated_pockets[0].keys()) if len(annotated_pockets) > 0 else []
    extracted_pockets_columns = list(extracted_pockets[0].keys()) if len(extracted_pockets) > 0 else []
    annotated_extra_data_columns = list(list(annotated_extra_data.values())[
                                        0].keys()) if annotated_extra_data else []
    metrics_columns = ['matched_amino_acids',
                       'recall', 'iou', 'false_positives']

    columns = \
        [f'annotated_{col}' for col in annotated_pockets_columns] + \
        [f'extracted_{col}' for col in extracted_pockets_columns] + \
        ["status"] + \
        metrics_columns
    rows = []

    rows.extend(
        _create_rows_for_matched_pockets(
            annotated2extracted_pocket_matches,
            matched_amino_acids,
            columns,
            list(extra_constant_columns.values()),
            annotated_extra_data
        )
    )
    rows.extend(
        _create_rows_for_umatched_annotated_pockets(
            annotated_pockets,
            annotated2extracted_pocket_matches,
            annotated_pockets_columns,
            extracted_pockets_columns,
            metrics_columns,
            list(extra_constant_columns.values()),
            annotated_extra_data
        )
    )
    rows.extend(
        _create_rows_for_umatched_extracted_pockets(
            extracted_pockets,
            annotated2extracted_pocket_matches,
            annotated_pockets_columns,
            extracted_pockets_columns,
            metrics_columns,
            list(extra_constant_columns.values()),
            annotated_extra_data_columns,
            annotated_extra_data
        )
    )

    columns = list(extra_constant_columns.keys()) + \
        columns + annotated_extra_data_columns

    return columns, rows


def _prefixise_keys(d: dict, prefix: str) -> dict:
    """
    Add a prefix to the keys of the dictionary.
    """
    return {f"{prefix}_{k}": v for k, v in d.items()}


def _create_rows_for_matched_pockets(
    annotated2extracted_pocket_matches: list[dict],
    matched_amino_acids: list[dict],
    columns: list[str],
    extra_constant_columns_values: list = None,
    annotated_extra_data: dict[str, dict] = None
):
    extra_constant_columns_values = extra_constant_columns_values if extra_constant_columns_values else []
    parsed_pocket_matches = _parse_pocket_matches(
        annotated2extracted_pocket_matches
    )

    matched_pockets_rows = [
        {
            **_prefixise_keys(pockets["annotated_pocket"], "annotated"),
            **_prefixise_keys(pockets["extracted_pocket"], "extracted"),
            **_calculate_amino_acid_metrics_per_pocket(
                pockets["annotated_pocket"],
                pockets["extracted_pocket"],
                amino_acids
            ),
            "matched_amino_acids": json.dumps(amino_acids),
            "status": "matched"
        }
        for amino_acids, pockets in zip(
            matched_amino_acids, parsed_pocket_matches
        )
    ]

    matched_pockets_rows = [
        _stringify_amino_acids(
            r, ["annotated_amino_acids", "extracted_amino_acids"]
        )
        for r in matched_pockets_rows
    ]

    rows = []

    for pocket_row in matched_pockets_rows:
        row = extra_constant_columns_values + \
            [pocket_row[col] for col in columns]
        if annotated_extra_data:
            row += list(
                annotated_extra_data[pocket_row["annotated_pocket_id"]].values(
                )
            )
        rows.append(row)

    return rows


def _create_rows_for_umatched_annotated_pockets(
    annotated_pockets: list[dict],
    annotated2extracted_pocket_matches: list[dict],
    annotated_pockets_columns: list[str],
    extracted_pockets_columns: list[str],
    metrics_columns: list[str],
    extra_constant_columns_values: list = None,
    annotated_extra_data: dict[str, dict] = None
):
    """
    Create rows for the unmatched annotated pockets.
    """
    extra_constant_columns_values = extra_constant_columns_values if extra_constant_columns_values else []
    parsed_pocket_matches = _parse_pocket_matches(
        annotated2extracted_pocket_matches
    )
    annotated_pocket_ids = [p["pocket_id"] for p in annotated_pockets]
    matched_pocket_ids = [
        p["annotated_pocket"]["pocket_id"]
        for p in parsed_pocket_matches
    ]

    unmatched_annotated_pocket_ids = \
        set(annotated_pocket_ids) - set(matched_pocket_ids)

    annotated_pockets = [
        _stringify_amino_acids(p, "amino_acids") for p in annotated_pockets
    ]

    rows = []
    for pocket in annotated_pockets:
        if pocket["pocket_id"] in unmatched_annotated_pocket_ids:

            row = extra_constant_columns_values + \
                [pocket[col] for col in annotated_pockets_columns] + \
                [None for _ in extracted_pockets_columns] + \
                ["unmatched"] + \
                [None for _ in metrics_columns]

            if annotated_extra_data:
                row += list(annotated_extra_data[pocket["pocket_id"]].values())

            rows.append(row)

    return rows


def _create_rows_for_umatched_extracted_pockets(
    extracted_pockets: list[dict],
    annotated2extracted_pocket_matches: list[dict],
    annotated_pockets_columns: list[str],
    extracted_pockets_columns: list[str],
    metrics_columns: list[str],
    extra_constant_columns_values: list = None,
    annotated_extra_data_columns: list = None,
    annotated_extra_data: dict[str, dict] = None
):
    """
    Create rows for the unmatched extracted pockets.
    """
    extra_constant_columns_values = extra_constant_columns_values if extra_constant_columns_values else []
    annotated_extra_data_columns = annotated_extra_data_columns if annotated_extra_data_columns else []
    parsed_pocket_matches = _parse_pocket_matches(
        annotated2extracted_pocket_matches
    )
    extracted_pocket_ids = [p["pocket_id"] for p in extracted_pockets]
    matched_pocket_ids = [
        p["extracted_pocket"]["pocket_id"]
        for p in parsed_pocket_matches
    ]

    unmatched_extracted_pocket_ids = \
        set(extracted_pocket_ids) - set(matched_pocket_ids)

    extracted_pockets = [
        _stringify_amino_acids(p, "amino_acids") for p in extracted_pockets
    ]

    rows = []

    if extracted_pockets:
        for pocket in extracted_pockets:
            if pocket["pocket_id"] in unmatched_extracted_pocket_ids:
                row = extra_constant_columns_values + \
                    [None for _ in annotated_pockets_columns] + \
                    [pocket[col] for col in extracted_pockets_columns] + \
                    ["fake"] + \
                    [None for _ in metrics_columns] + \
                    [_add_target_and_article_columns(col, annotated_extra_data)
                    for col in annotated_extra_data_columns]
                rows.append(row)
    else:
        row = extra_constant_columns_values + \
            [None for _ in annotated_pockets_columns] + \
            [None for col in extracted_pockets_columns] + \
            ["fake_paper"] + \
            [None for _ in metrics_columns] + \
            [_add_target_and_article_columns(col, annotated_extra_data)
            for col in annotated_extra_data_columns]
        rows.append(row)
    return rows


def _parse_pocket_matches(
    annotated2extracted_pocket_matches: list[dict]
) -> list[dict]:
    """
    Parse the pocket matches.
    """
    return [
        {
            "annotated_pocket": json.loads(pockets["annotated_pocket"]),
            "extracted_pocket": json.loads(pockets["extracted_pocket"])
        }
        for pockets in annotated2extracted_pocket_matches
    ]


def _calculate_amino_acid_metrics_per_pocket(
    annotated_pocket: dict,
    extracted_pocket: dict,
    matched_amino_acids: dict
) -> dict:
    """
    Calculate the metrics for the matched amino acids.
    """
    annotated_ids = annotated_pocket["amino_acids"]
    extracted_ids = extracted_pocket["amino_acids"]
    matched_ids = matched_amino_acids.keys()

    metrics = {
        "recall": recall(annotated_ids, matched_ids),
        "iou": intersection_over_union(annotated_ids, extracted_ids, matched_ids),
        "false_positives": percentage_of_false_positives(extracted_ids, matched_ids)
    }

    return metrics


def _prepare_amino_acids_wandb_table(
    annotated_pockets: list[dict],
    extracted_pockets: list[dict],
    annotated2extracted_pocket_matches: list[dict],
    matched_amino_acids: list[dict],
    constant_extra_columns: dict = None,
    annotated_extra_data: dict = None
) -> tuple[list[str], list[list]]:
    """
    Prepare a wandb table with the amino acids and their metadata.
    The expected minimal set of columns:
    - annotated_pocket_id
    - annotated_amino_acid
    - extracted_pocket_id
    - extracted_amino_acid
    """
    constant_extra_columns = constant_extra_columns if constant_extra_columns else {}
    annotated_extra_data_columns = list(list(annotated_extra_data.values())[
                                        0].keys()) if annotated_extra_data else []
    parsed_pocket_matches = _parse_pocket_matches(
        annotated2extracted_pocket_matches
    )
    amino_acids_rows = []

    # init with all amino acids and subsequently remove the matched ones
    unmatched_annotated_amino_acids = _get_all_amino_acids(annotated_pockets)
    unmatched_extracted_amino_acids = _get_all_amino_acids(extracted_pockets)

    # add rows for the matched amino acids
    for amino_acids, pockets in zip(
        matched_amino_acids, parsed_pocket_matches
    ):
        annotated_pocket = pockets["annotated_pocket"]
        extracted_pocket = pockets["extracted_pocket"]
        for annotated_amino_acid, extracted_amino_acid in amino_acids.items():
            amino_acids_rows.append(
                list(constant_extra_columns.values()) +
                [
                    annotated_pocket["pocket_id"],
                    annotated_amino_acid,
                    extracted_pocket["pocket_id"],
                    extracted_amino_acid,
                    "matched"
                ] +
                list(annotated_extra_data[
                    f'{annotated_pocket["pocket_id"]}___{annotated_amino_acid}'
                ].values())
            )
        # remove the matched amino acids
        unmatched_annotated_amino_acids -= \
            set(
                (annotated_pocket["pocket_id"], amino_acid)
                for amino_acid in amino_acids.keys()
            )

        unmatched_extracted_amino_acids -= \
            set(
                (extracted_pocket["pocket_id"], amino_acid)
                for amino_acid in amino_acids.values()
            )

    # add rows for the unmatched amino acids
    for pocket_id, unmatched_annotated_amino_acid in unmatched_annotated_amino_acids:
        amino_acids_rows.append(
            list(constant_extra_columns.values()) +
            [
                pocket_id,
                unmatched_annotated_amino_acid,
                None,
                None,
                "unmatched"
            ] +
            list(annotated_extra_data[
                f'{pocket_id}___{unmatched_annotated_amino_acid}'
            ].values())
        )

    for pocket_id, unmatched_extracted_amino_acid in unmatched_extracted_amino_acids:
        amino_acids_rows.append(
            list(constant_extra_columns.values()) +
            [
                None,
                None,
                pocket_id,
                unmatched_extracted_amino_acid,
                "fake",
            ] +
            [_add_target_and_article_columns(col, annotated_extra_data)
             for col in annotated_extra_data_columns])
        # TODO: add article and target column to the fake amino acids

    columns = list(constant_extra_columns.keys()) + \
        ["annotated_pocket_id", "annotated_amino_acid",
         "extracted_pocket_id", "extracted_amino_acid", "status"] + \
        annotated_extra_data_columns

    return columns, amino_acids_rows


def _stringify_amino_acids(pocket: dict, amino_acids_key: Union[str, list[str]]) -> dict:
    if isinstance(amino_acids_key, str):
        return {
            **pocket,
            amino_acids_key: "\n".join(pocket[amino_acids_key])
        }
    elif isinstance(amino_acids_key, list):
        return {
            **pocket,
            **{key: "\n".join(pocket[key]) for key in amino_acids_key}
        }
    else:
        raise ValueError("amino_acids_key must be either str or list of str")


def _get_all_amino_acids(pockets: list[dict]) -> set[str]:
    all_amino_acids = set()
    for pocket in pockets:
        all_amino_acids |= set(
            (pocket['pocket_id'], amino_acid)
            for amino_acid in pocket['amino_acids']
        )

    return all_amino_acids


def _add_target_and_article_columns(
    column_name: str,
    annotated_extra_data: dict[str, dict]
) -> Union[str, None]:
    entries = list(annotated_extra_data.values())

    if len(entries) == 0:
        return None

    if column_name in ["article_pdf_name", "target"]:
        return entries[0][column_name]
    else:
        return None


def _merge_dataframes(data1, columns1, data2, columns2):
    """
    Merge two lists of data with potentially mismatched columns, filling with NaNs where necessary.
    """
    df1 = pd.DataFrame(data1, columns=columns1)
    df2 = pd.DataFrame(data2, columns=columns2)

    if not df2.empty:
        df_new = pd.concat([df1, df2], axis=0, ignore_index=True)
        return df_new.values.tolist()
    return df1.values.tolist()


def rearrange_columns(data, columns):
    df = pd.DataFrame(data, columns=columns)
    columns = []
    df = df[columns]
    return df.values.tolist(), columns