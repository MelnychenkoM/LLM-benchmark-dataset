import json
import re

from typing import List, Dict
from langchain_core.runnables import RunnableLambda


AMINO_ACIDS_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


match_pockets_runnable = RunnableLambda(lambda inputs: 
                                        match_pockets(
                                            inputs['annotated_pockets'], 
                                            inputs['extracted_pockets'],
                                            relevance=inputs['relevance']
                                            )
                                        )


def match_pockets(annotated_pockets: List[Dict], 
                  extracted_pockets: List[Dict],
                  relevance: dict,
                  threshold: float = 0.3) -> dict:

    matched_pockets = {}
    best_matches = {}
    annotated2extracted_pocket_matches = {}
    matched_amino_acids = {}

    for annotated_pocket in annotated_pockets:
        annotated_id = annotated_pocket['pocket_id']
        annotated_residues = annotated_pocket.get('amino_acids', [])

        for extracted_pocket in extracted_pockets:
            extracted_id = extracted_pocket['pocket_id']
            extracted_residues = extracted_pocket.get('amino_acids', [])

            if not annotated_residues or not extracted_residues:
                continue

            matched_residues, iou = amino_acid_overlap(annotated_residues, extracted_residues)

            if iou > threshold and iou > best_matches.get(annotated_id, 0):
                matched_pockets[annotated_id] = extracted_id
                best_matches[annotated_id] = iou
                annotated2extracted_pocket_matches[annotated_id] = {
                    'annotated_pocket': json.dumps(annotated_pocket),
                    'extracted_pocket': json.dumps(extracted_pocket),
                }
                matched_amino_acids[annotated_id] = matched_residues

    annotated2extracted_pocket_matches = list(annotated2extracted_pocket_matches.values())
    matched_amino_acids = list(matched_amino_acids.values())

    return {
        "annotated_pockets": annotated_pockets,
        "extracted_pockets": extracted_pockets,
        "annotated2extracted_pocket_matches": annotated2extracted_pocket_matches,
        "matched_amino_acids": matched_amino_acids,
        "relevance": relevance
    }


def amino_acid_overlap(annotated_residues: List[str], 
                       extracted_residues: List[str]):
    def extract_residue(residue):
        match = re.search(r'([A-Z][a-z]*)(\d+)', residue)
        if match:
            amino_acid, pos = match.group(1), match.group(2)
            return AMINO_ACIDS_3_TO_1.get(amino_acid.upper(), amino_acid), pos
        return None, None

    matches = {}
    for res_annotated in annotated_residues:
        annotated_amino_acid, annotated_pos = extract_residue(res_annotated)
        if not annotated_amino_acid:
            continue
        for res_extracted in extracted_residues:
            extracted_amino_acid, extracted_pos = extract_residue(res_extracted)
            if not extracted_amino_acid:
                continue
            if annotated_amino_acid == extracted_amino_acid and annotated_pos == extracted_pos:
                matches[res_annotated] = res_extracted

    intersection_count = len(matches)
    union_count = len(annotated_residues) + len(extracted_residues) - intersection_count
    iou = intersection_count / union_count if union_count else 0
    return matches, iou