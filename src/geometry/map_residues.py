from biotite.structure import AtomArray
from biotite.sequence import ProteinSequence
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Set, Optional
from .utils import *

from sklearn.cluster import DBSCAN, MeanShift
import logging


def generate_motif_variations_same_offset(motif_residues: List[str],
                                          max_offset= 10
                                          ):
    """ Generates motif variations with the same offset for all residues. """
    
    _, pos = parse_residues(motif_residues)
    variations = [pos]

    for offset in range(1, max_offset + 1):
        variations.append([resSeq + offset for resSeq in pos])
        variations.append([resSeq - offset for resSeq in pos])

    return variations


def find_best_matching_motif(motif_residues: List[str], 
                             structure: AtomArray,
                             max_offset: int = 10,
                             ) -> Tuple[list[int], float, dict]:
    """ Finds the best matching motif in the DataFrame, comparing both resSeq and resName. """

    variations = generate_motif_variations_same_offset(motif_residues, max_offset)

    best_match = None
    best_score = -1
    best_mapping = {}

    residues, pos = parse_residues(motif_residues)

    for variation in variations:
        match_count = 0
        mapping = {}

        for i, resSeq in enumerate(variation):
            residue_name = residues[i]
            original_res = motif_residues[i]

            match = structure[(structure.res_id == resSeq) & (structure.res_name == residue_name)]
            
            if match:
                match_count += 1
                found_res = match[0].res_name + str(resSeq)
                mapping[original_res] = found_res
            else:
                residue_at_pos = structure[structure.res_id == resSeq]
                if residue_at_pos:
                    actual_res_name = residue_at_pos[0].res_name
                    mapping[original_res] = f"{actual_res_name}{resSeq}"
                else:
                    mapping[original_res] = None

        score = match_count / len(motif_residues)

        if score > best_score:
            best_score = score
            best_match = variation
            best_mapping = mapping

        if score == 1:
            return best_match, best_score, best_mapping
    return best_match, best_score, best_mapping
    

def match_motifs_to_chains(
    all_motifs: List[List[str]],
    structure: AtomArray,
    max_offset: int = 10,
    threshold: float = 0.7,
) -> Dict[int, List[Dict[str, Optional[str]]]]:
    """
    Matches multiple binding site motifs to chains in a PDB DataFrame, 
    returning all matches above threshold.

    Args:
        all_motifs: A list of lists, where each inner list contains residue strings
                    for a single binding site/motif.
        structure: The AtomArray representing the PDB structure.
        max_offset: The maximum offset used when searching for motif variations.
        threshold: The minimum score for a match to be included in the results.

    Returns:
        A dictionary where:
            - Keys are the integer indices of the input motifs in `all_motifs`.
            - Values are lists of dictionaries, each dictionary representing a match:
                - "chain" (Optional[str]): The chainID of the match.
                - "score" (float): The score of the match.
                - "mapping" (Dict[str, Optional[str]]): The residue mapping for the match.
                - "best_match": (List[str]): the residues that made the match
    """
    motif_matches: Dict[int, List[Dict[str, Optional[str]]]] = {}

    chains = set(structure.chain_id)
    if len(chains) == 0:
        logging.warning("Warning: The AtomArray contains no chains.")
        return motif_matches

    for motif_idx, current_motif in enumerate(all_motifs):
        motif_matches[motif_idx] = []
        if not current_motif:
            continue

        for chain_id in chains:
            selected_chain = structure[structure.chain_id == chain_id]
            best_match, score, mapping = find_best_matching_motif(current_motif, 
                                                                  selected_chain, 
                                                                  max_offset)
            if score >= threshold:
                logging.debug(f"Matching result: LLM chain #{motif_idx} --> chain {chain_id}: {score:.2f}")
                motif_matches[motif_idx].append({
                    "chain": chain_id,
                    "score": score,
                    "mapping": mapping,
                    "best_match": best_match,
                })
    return motif_matches


def find_relevant_clusters(
        cluster_atom_dict,
        pdb_to_original_map,
        original_pockets,
        threshold: float = 0.7
    ):
    """
    Identifies clusters that significantly overlap with predefined pockets.
    If a cluster significantly overlaps with a pocket, it extends the atom list
    associated with that pocket match, ensuring only unique atoms are stored.
    """
    cluster_per_pocket_match = {}

    for cluster_id, atoms in cluster_atom_dict.items():
        represented_orig_res = set().union(*[pdb_to_original_map.get(pdb_res, set())
                                             for pdb_res in get_residues_from_atom_array(atoms)])

        if not represented_orig_res:
            continue

        for pocket_id, pocket_orig_residues in original_pockets.items():
            if not pocket_orig_residues:
                continue

            intersection = pocket_orig_residues.intersection(represented_orig_res)
            score = len(intersection) / len(pocket_orig_residues)

            if score >= threshold:
                logging.debug("Cluster to PDB: Cluster {} --> {}. Score: {}; Chains: {}; {}".format(
                    cluster_id,
                    pocket_id,
                    score,
                    set(atoms.chain_id),
                    intersection
                ))
                if cluster_id not in cluster_per_pocket_match:
                    cluster_per_pocket_match[cluster_id] = {
                        "pocket_id": pocket_id,
                        "score": score,
                        "atoms": atoms.copy()
                    }
                else:
                    existing_match = cluster_per_pocket_match[cluster_id]
                    existing_match["atoms"] = existing_match["atoms"] + atoms # TODO drop duplicates
                    existing_match["score"] = max(existing_match["score"], score)
    return list(cluster_per_pocket_match.values())


def contains_interface_pockets(clusters: list, protein: AtomArray) -> bool:
    if len(set(protein.chain_id)) == 1:
        return False
    interface_pockets = False
    for value in clusters:
        if len(set(value['atoms'].chain_id)) != 1:
            interface_pockets = True
            break
    return interface_pockets


def get_residues_per_pocket(matches, original_pockets):
    pocket_info_list = []
    for motif, chain_matches in matches.items():
        for chain_matched in chain_matches:
            residue_map = chain_matched['mapping']
            chain_id = chain_matched['chain']
            score = chain_matched['score']

            mapped_pockets = {}
            for pocket, residues in original_pockets.items():
                mapped_residues = set()
                for residue in residues:
                    if residue in residue_map:
                        mapped_residues.add(residue_map[residue])
                    else:
                        mapped_residues.add(residue)
                mapped_pockets[pocket] = mapped_residues

            for pocket_id, mapped_residues in mapped_pockets.items():
                res_names, res_ids = parse_residues(mapped_residues)

                pocket_info = {
                    'res_names': res_names,
                    'res_ids': res_ids,
                    'chain': chain_id,
                    'score': score,
                    'pocket_id': pocket_id,
                    'mapped_residues': mapped_residues
                }
                pocket_info_list.append(pocket_info)
    return pocket_info_list


def map_to_pdb(pockets: dict, 
               protein: AtomArray, 
               match_threshold: float = 0.7,
               cluster_match_threshold: float = 0.7,
               max_offset: int = 10,
               **kwargs
               ) -> List[Dict]:
    """ 
    Main function that maps the LLM output onto the provided PDB. 
    It considers two cases:
        1. If there are binding sites at the interfaces between chains, we use clustering results
        to map the residues to the original pockets.
        2. If there are no binding sited at the interfaces we map each pocket to each chain individually.
    """
    motif, unique_chains = stringfy_amino_acids_by_chain(pockets)
    matches = match_motifs_to_chains(motif, 
                                     protein, 
                                     max_offset=max_offset, 
                                     threshold=match_threshold)

    if not any(matches.values()):
        return None

    combined_mask = np.zeros(protein.chain_id.shape, dtype=bool)

    atoms_per_chain_match = {}

    for pocket_id, motif_match in matches.items():
        atoms_per_chain_match[pocket_id] = {}
        for chain_match_info in motif_match:
            chain_id = chain_match_info['chain']
            mapped_residues = list(chain_match_info['mapping'].values())
            
            res_names, res_ids = parse_residues(mapped_residues)

            mask = np.any(
                (protein.chain_id[:, np.newaxis] == chain_id) &
                (protein.res_id[:, np.newaxis] == res_ids) &
                (protein.res_name[:, np.newaxis] == res_names),
                axis=1
            )
            combined_mask |= mask
            selected_atoms = protein[combined_mask]
            atoms_per_chain_match[pocket_id][chain_id] = protein[mask]

    logging.debug(f"Total atoms: {len(protein[combined_mask])}")
    centroids, inverse_map = calculate_residue_centroids(protein[combined_mask])
    
    # clusters = DBSCAN(**kwargs).fit_predict(centroids)
    clusters = MeanShift(**kwargs).fit_predict(centroids)

    cluster_mask = clusters != -1
    unique_clusters = np.unique(clusters[cluster_mask])
    logging.debug(f"Num. unique clusters: {len(unique_clusters)}")
    cluster_atoms_dict = {}

    for cluster_id in unique_clusters:
        if cluster_id != -1: 
            cluster_residues = np.where(clusters == cluster_id)[0]
            atom_indices_in_cluster = np.where(np.isin(inverse_map, cluster_residues))[0]
            atoms_in_cluster = protein[combined_mask][atom_indices_in_cluster]
            cluster_atoms_dict[cluster_id] = atoms_in_cluster

    residue_pdb_map, pdb_to_original_map = map_residues_bidirectional(matches)

    original_pockets = {}
    for pocket in pockets['pockets']:
        original_pockets[pocket['pocket_id']] = \
            {parse_residue_id(aa) for aa in pocket['amino_acids']}

    clusters_to_pdb = find_relevant_clusters(
        cluster_atoms_dict, 
        pdb_to_original_map, 
        original_pockets,
        threshold=cluster_match_threshold
    )
    is_interface = contains_interface_pockets(clusters_to_pdb, protein)
    logging.debug(f"Interface binding sites: {is_interface}")

    if is_interface:
        return clusters_to_pdb, cluster_atoms_dict
    
    pocket_info_list = get_residues_per_pocket(matches, original_pockets)
    atoms_per_pocket_match = []

    for pocket_info in pocket_info_list:
        mask = np.any(
            (protein.chain_id[:, np.newaxis] == pocket_info['chain']) &
            (protein.res_id[:, np.newaxis] == pocket_info['res_ids']) &
            (protein.res_name[:, np.newaxis] == pocket_info['res_names']),
            axis=1
        )
        atoms_per_pocket_match.append({
            "pocket_id": pocket_info['pocket_id'],
            "score": pocket_info['score'],
            "atoms": protein[mask],
        })
    return atoms_per_pocket_match, cluster_atoms_dict


def calculate_residue_centroids(atom_array: AtomArray):
    """
    Calculates the centroid for each residue in an atom array.
    """
    residue_centroids = []
    unique_residues, unique_indices, inverse_map = np.unique(
        np.stack((atom_array.chain_id, atom_array.res_id, atom_array.res_name)), 
        axis=1, return_index=True, return_inverse=True
    )
    for chain_id, res_id, res_name in unique_residues.T:
        residue_atoms = atom_array[
            (atom_array.chain_id == chain_id) &
            (atom_array.res_id == res_id.astype(np.int16)) &
            (atom_array.res_name == res_name)
        ]
        if len(residue_atoms) > 0:
            centroid = np.mean(residue_atoms.coord, axis=0)
            residue_centroids.append(centroid)
    return np.array(residue_centroids), inverse_map
