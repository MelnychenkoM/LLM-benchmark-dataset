import pandas as pd
import numpy as np

def recall(annotated_ids: list, matched_ids: list) -> float:
    """
    Calculate the percentage of true items in the extracted.
    """
    if len(annotated_ids) == 0:
        return 0
    return len(set(annotated_ids) & set(matched_ids)) / len(annotated_ids)


def intersection_over_union(
    annotated_ids: list[str],
    extracted_ids: list[str],
    matched_ids: list[str]
) -> float:
    """
    Calculate the IoU metric for the matched amino acids.
    """
    if len(annotated_ids) == 0:
        return 0

    intersection = len(matched_ids)
    union = len(annotated_ids) + len(extracted_ids) - intersection

    return intersection / union


def percentage_of_false_positives(
    extracted_ids: list[str],
    matched_ids: list[str]
) -> float:
    """
    Calculate the percentage of false positives.
    """
    if len(extracted_ids) == 0:
        return 0

    return (len(extracted_ids) - len(matched_ids)) / len(extracted_ids)


def calculate_pockets_accuracy(row):
    if row['status'] == 'matched':
        if pd.notna(row['annotated_pocket_id']):
            return 1
        else:
            return 0
    elif row['status'] == 'fake_paper' and pd.isna(row['annotated_pocket_id']):
        return 1
    else:
        return 0
    

def calculate_pocket_number_accuracy(df):
    pocket_accuracy = []

    for article in df['article_pdf_name'].unique():
        rows = df.query(f"article_pdf_name == '{article}'")
        both_not_nan = ~(rows['annotated_pocket_id'].isna()) & ~(rows['extracted_pocket_id'].isna())
        both_nan = rows['annotated_pocket_id'].isna() & rows['extracted_pocket_id'].isna()

        if both_not_nan.all() or both_nan.all():
            pocket_accuracy.append(1)
        else:
            pocket_accuracy.append(0)
    return np.mean(pocket_accuracy)
