from collections import defaultdict
import regex as re
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray
from typing import List, Union, Dict, Any, Set, Tuple


def stringfy_amino_acids_by_chain(input_data: Dict[str, Any]):
    """
    Extracts amino acids from ALL pockets in the input dictionary
    and formats them as '<ResName><ResID>'.

    Args:
        input_data: A dictionary representing the LLM output that contains pocket information, 
        expected to have a key 'pockets' which is a list of pocket dictionaries.

    Returns:
        A list of sorted lists, where each inner list represents a chain
        (List[List[str]]). Returns an empty list [] if no amino acids are found.
    """
    all_amino_acids_data: List[Tuple[Any, int, str]] = []
    unique_chain_ids: Set[Any] = set()

    if 'pockets' not in input_data or not isinstance(input_data['pockets'], list):
        return []

    for pocket in input_data['pockets']:
        if 'amino_acids' not in pocket or not isinstance(pocket['amino_acids'], list):
            continue
        for aa in pocket['amino_acids']:
            if not all(k in aa for k in ('chain_id', 'res_name', 'res_id')):
                 raise KeyError(f"Amino acid dictionary missing required keys: {aa}")
            if not isinstance(aa['res_id'], int) or not isinstance(aa['res_name'], str):
                 raise TypeError(f"Incorrect type for res_id or res_name in: {aa}")

            chain_id = aa.get('chain_id')
            res_id = aa['res_id']
            res_name = aa['res_name']

            all_amino_acids_data.append((chain_id, res_id, res_name))
            unique_chain_ids.add(chain_id)

    if not all_amino_acids_data:
        return []

    non_none_chains = {cid for cid in unique_chain_ids if cid is not None}
    has_multiple_chains = len(non_none_chains) > 1

    if not has_multiple_chains:
        all_amino_acids_data.sort(key=lambda item: item[1])
        result_list = [[f"{name}{rid}" for cid, rid, name in all_amino_acids_data]]
        return result_list, []
    else:
        grouped_by_chain: Dict[Any, List[Tuple[int, str]]] = {}

        for chain_id, res_id, res_name in all_amino_acids_data:
            if chain_id not in grouped_by_chain:
                grouped_by_chain[chain_id] = []
            grouped_by_chain[chain_id].append((res_id, res_name))

        result_list_of_lists: List[List[str]] = []
        chain_ids_list: List[str] = []
        sorted_chain_ids = sorted(grouped_by_chain.keys(), key=lambda x: (x is None, x))

        for chain_id in sorted_chain_ids:
            amino_acids_in_chain = grouped_by_chain[chain_id]
            amino_acids_in_chain.sort(key=lambda item: item[0])
            formatted_chain_list = list(set(f"{name}{rid}" for 
                                            rid, name in amino_acids_in_chain))
            if len(formatted_chain_list) > 1:
                result_list_of_lists.append(formatted_chain_list)
            chain_ids_list.append(chain_id)

        return result_list_of_lists, chain_ids_list
    

def parse_residues(residues: List[str]):
    """ Parses residues into two separate lists with res_names and res_ids """
    amino_acids = []
    res_ids = []
    converter = ProteinSequence()

    for residue in residues:
        if residue:
            match = re.search(r'([A-Za-z]{1,3})(\d+)', residue)
            if match:
                residue_name, residue_pos = match.group(1).upper(), match.group(2)
                res_ids.append(int(residue_pos))
                if len(residue_name) == 1:
                    three_letter_code = converter.convert_letter_1to3(residue_name)
                    amino_acids.append(three_letter_code)
                elif len(residue_name) == 3:
                    amino_acids.append(residue_name.upper())
                else:
                    # logging.warning(f"Incorrect residue passed: {residue_name}")
                    continue
    return amino_acids, res_ids


def get_residues_from_atom_array(atoms: AtomArray) -> List[str]:
    """ Get residues as a string from biotite's AtomArray """
    unique_residues = set()
    for atom in atoms:
        unique_residues.add(f"{atom.chain_id}:{atom.res_name}{atom.res_id}")
    return [str(residue) for residue in unique_residues]


def parse_residue_id(res_dict: dict) -> str:
    return f"{res_dict['res_name']}{res_dict['res_id']}"


def map_residues_bidirectional(mapping_data: dict):
    """
    Maps original residue identifiers to PDB chain and residue identifiers, 
    and vice-versa, based on mapping data.
    """
    residue_pdb_map = defaultdict(list)
    pdb_to_original_map = defaultdict(set)

    for chain_mappings_list in mapping_data.values():
        for chain_map in chain_mappings_list:
            pdb_chain = chain_map['chain']
            for original_res_id, pdb_res_id_str in chain_map['mapping'].items():
                mapping_string = f"{pdb_chain}:{pdb_res_id_str}"
                residue_pdb_map[original_res_id].append(mapping_string)
                pdb_to_original_map[mapping_string].add(original_res_id)

    return dict(residue_pdb_map), dict(pdb_to_original_map)


def snake_case_name(pocket_id: str) -> str:
    """
    Converts a pocket_id string to snake_case.
    
    Examples:
        "Anionic and Selectivity Binding Pocket" -> "anionic_and_selectivity_binding_pocket"
        "example-name" -> "example_name"
    """
    name = re.sub(r'[-\s]+', '_', pocket_id)
    name = re.sub(r'[^\w_]', '', name)
    return name.lower()