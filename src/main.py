from utils.readers import MarkdownReader, PDFGrobidReader
from chains import build_llm_pipeline, get_llm_model, write_pockets_to_md
from chains.refine.prompts.builder import RefineChainPromptsFromArtifactsBuilder
from dotenv import load_dotenv
import json
import os
from pathlib import Path
import shutil
import numpy as np
import logging
import argparse
import yaml
import copy
import collections.abc

from geometry import (
    FPocket, 
    map_to_pdb, 
    recursive_merge_and_match, 
    create_grid,
    get_alpha_spheres,
    correct_volume_grid,
    save_to_mol2,
    save_to_pqr,
    get_pocket_match_mask,
    correct_volume_grid,
    clean_grid
)
from geometry.utils import snake_case_name, get_residues_from_atom_array

from biotite.structure.io import load_structure, save_structure
from biotite.structure import AtomArray

from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

load_dotenv('keys.env')

def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def _parse_value(value_str):
    """Attempt to parse string to bool, int, float, or keep as string."""
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    if value_str.lower() == 'none':
        return None
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def set_nested_value(config_dict, key_path_str, value_str):
    """
    Sets a value in a nested dictionary based on a dot-separated key path.
    e.g., key_path_str = "extraction.model"
    """
    keys = key_path_str.split('.')
    d = config_dict
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]

    final_key = keys[-1]
    parsed_value = _parse_value(value_str)
    d[final_key] = parsed_value


CONFIG = {
    "extraction": {
        "model": "gpt-4o-mini",
        "max_tokens": 5000,
        "temperature": 0
    },
    "refine": {
        "model": "gpt-4o-mini",
        "max_tokens": 5000,
        "temperature": 0
    },
    "filter": {
        "model": "gpt-4o-mini",
        "max_tokens": 2500,
        "temperature": 0
    },
    "parse": {
        "model": "gpt-4o-mini",
        "max_tokens": 3000,
        "temperature": 0
    },
    "refine_call": True,
    "filter_call": True,
    "spacing": 1.5,
    "max_offset": 1,
    "threshold_chain_match": 0.6,
    "threshold_cluster_match": 0.7,
    "threshold_merge": 0.7,
    "threshold_residue_match": 0.3,
    "threshold_pocket_match": 0.1,
    "return_best_match": True,
    "clustering": {
        "bandwidth": 10,
        "cluster_all": False
    },
    "debug": False
}


class PrefixAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
        self.prefix = self.extra.get('prefix', '')

    def process(self, msg, kwargs):
        return f"[{self.prefix}] {msg}", kwargs
    

def setup_output_directory(out_path: str, 
                           pdb_path: str, 
                           config: dict,
                           folder_name: str = "pkres/",
                           ):
    out_path = Path(out_path)
    pdb_path = Path(pdb_path)

    full_out_path = out_path / folder_name

    pdb_filename = pdb_path.name
    destination_pdb_path = full_out_path / pdb_filename

    pockets_path = full_out_path / "pockets"
    papers_path = full_out_path / "papers"
    debug_path = full_out_path / "debug"
    alpha_spheres_path = debug_path / "alpha_spheres"
    residues_path = full_out_path / "residues"

    config_filename = "config.json"
    config_filepath = full_out_path / config_filename

    if full_out_path.exists():
        shutil.rmtree(full_out_path)

    full_out_path.mkdir(parents=True, exist_ok=True)
    pockets_path.mkdir(parents=True, exist_ok=True)
    papers_path.mkdir(parents=True, exist_ok=True)
    debug_path.mkdir(parents=True, exist_ok=True)
    alpha_spheres_path.mkdir(parents=True, exist_ok=True)
    residues_path.mkdir(parents=True, exist_ok=True)

    shutil.copy2(str(pdb_path), destination_pdb_path)

    with open(config_filepath, 'w') as f:
        json.dump(config, f, indent=4)

    return {
        "full_out_path": full_out_path, 
        "pockets_path": pockets_path,
        "residues_path": residues_path, 
        "papers_path": papers_path,
        "debug_path": debug_path,
        "alpha_spheres_path": alpha_spheres_path
    }


def generate_debug_info(fpocket: FPocket,
                        assigned_rows: np.ndarray):
    num_total_pockets = len(fpocket)
    all_indices = []

    for mask in assigned_rows:
        indices = np.where(mask == 1)[0].tolist()
        if indices:
            all_indices.extend(indices)

    num_matched_unique: int = len(set(all_indices))
    percent_selected: float = (
        num_matched_unique / num_total_pockets if num_total_pockets > 0 else 0.0
    )

    pockets_per_llm_pocket = len(all_indices) / len(assigned_rows) if \
                                    len(assigned_rows) > 0 else 0
    return {
        "num_total_pockets": num_total_pockets,
        "num_matched_unique": num_matched_unique,
        "percent_selected": percent_selected,
        "pockets_per_llm_pocket": pockets_per_llm_pocket
    }


def get_reader(article_path):
    _, extention = os.path.splitext(article_path)
    if extention == ".md":
        return MarkdownReader()
    elif extention == ".pdf":
        return PDFGrobidReader()
    else:
        logging.warning(f"Wrong format passed: {extention}. Only .md and .pdf are supported.") 
        return


def move_fpocket_output(pdb_path, out_path):
    pdb_path = Path(pdb_path)

    pdb_filename = pdb_path.stem
    pdb_filename_out_folder = pdb_path.parent / (pdb_filename + "_out")

    if pdb_filename_out_folder.exists() and pdb_filename_out_folder.is_dir():
        try:
            shutil.move(str(pdb_filename_out_folder), str(out_path))
        except Exception as e:
            logging.warning(f"Error moving fpocket folder: {e}")
    return


def run_llm_chain(text: str, 
                  target_protein: str, 
                  logger: logging.LoggerAdapter,
                  config: dict
                  ):
    refine_chain_prompts_builder = RefineChainPromptsFromArtifactsBuilder()

    llm_extraction = get_llm_model(**config['extraction'])
    llm_refine = get_llm_model(**config['refine'])
    llm_parse = get_llm_model(**config['parse'])
    llm_filter = get_llm_model(**config['filter'])

    chain = build_llm_pipeline(
        llm_filter,
        llm_extraction,
        llm_refine,
        llm_parse,
        refine_chain_prompts_builder, 
        use_refine_call=config['refine_call'],
        use_filter_call=config['filter_call'],
        logger=logger
    )
    response = chain.invoke(
    {
        "target_protein": target_protein,
        "context": text
    })
    return response


def run_map_pockets(pockets, protein, config):
    llm_pockets, cluster_atoms_dict = map_to_pdb(
        pockets, 
        protein, 
        cluster_match_threshold=config['threshold_cluster_match'],
        match_threshold=config['threshold_chain_match'],
        max_offset=config['max_offset'],
        **config['clustering']
    )
    return llm_pockets, cluster_atoms_dict


def create_pockets(jaccard_matrix: np.ndarray, 
                   fpocket: FPocket, 
                   protein: AtomArray,
                   threshold_pocket_match: float,
                   spacing: float,
                   clean_radius: float = 1.5,
                   return_best_match: bool = True
                   ):
    created_pockets = []
    assigned_rows = get_pocket_match_mask(
        jaccard_matrix,
        threshold_pocket_match,
        return_best_match=return_best_match
    )

    for i, mask in enumerate(assigned_rows):
        indicies = np.where(mask == 1)[0].tolist()
        if indicies:
            fpocket_pockets = fpocket[indicies]
            alpha_spheres = get_alpha_spheres(fpocket_pockets).to_numpy()
            
            save_to_mol2(f"pocket{i}.mol2", alpha_spheres[:, :-1])

            grid = create_grid(alpha_spheres, spacing)

            corrected_grid = correct_volume_grid(grid, protein, alpha_spheres)
            corrected_grid = clean_grid(corrected_grid, protein, clean_radius)
            created_pockets.append(corrected_grid)

    return created_pockets           


def main(article_path, pdb_path, target_protein, config):
    if config['debug']:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )

    logging.info(f"Target protein: {target_protein}")

    protein = load_structure(pdb_path, model=1, altloc='first')
    protein = protein[~protein.hetero]
    
    path_dict = setup_output_directory(
        "../data/", 
        pdb_path,
        config
    )
    
    article_path = Path(article_path)
    if article_path.is_dir():
        articles = article_path.iterdir()
    else:
        articles = [article_path]

    all_llm_pockets = []

    for article in articles:
        article_name = article.stem
    
        chain_logger = logging.getLogger('chains.refine.chain')
        llm_logger = PrefixAdapter(chain_logger, {'prefix': f'{article.name}'})

        reader = get_reader(article)

        llm_logger.info(f"Parsing the paper.")
        try:
            text = reader(article)
        except Exception as e:
            llm_logger.warning(f"File parsing failed: {e}. Skipping the paper.")
            continue

        response = run_llm_chain(text, target_protein, llm_logger, config)

        if response and hasattr(response, 'pockets') and response.pockets:
            pockets = response.model_dump()

            output_prefix = path_dict['papers_path'] / article_name
            with open(str(output_prefix) + ".json", 'w') as f:
                json.dump(pockets, f, indent=4, ensure_ascii=False)
            write_pockets_to_md(pockets, filename=str(output_prefix) + ".md")
        else:
            llm_logger.warning(f"No pockets found by LLM. Skipping.")
            continue

        llm_logger.info(f"Mapping LLM results to PDB.")
        try:
            llm_pockets, clusters = run_map_pockets(pockets, protein, config)
        except Exception as e:
            llm_logger.warning(f"PDB mapping failed: {e}.")

        if not llm_pockets:
            llm_logger.info(f"PDB mapping failed.")
            continue
        else:
            for value in llm_pockets:
                if not value['atoms']:
                    continue
                residues = get_residues_from_atom_array(value['atoms'])
                chains = "".join(np.unique(value['atoms'].chain_id))
                llm_logger.info("LLM pocket: {}. Chain(s): {}, Num. residues: {}".format(
                    value['pocket_id'], chains, len(residues)
                ))

                pocket_id = snake_case_name(value['pocket_id'])
                residue_name = f"{pocket_id}_{chains}.pdb"
                save_structure(path_dict['residues_path'] / residue_name, value['atoms'])

            clusters_path = path_dict['residues_path'] / "clusters"
            clusters_path.mkdir(parents=True, exist_ok=True)

            for cluster_id, values in clusters.items():
                save_structure(clusters_path / f'cluster{cluster_id}.pdb', values)

        all_llm_pockets.extend(llm_pockets)
    
    if not all_llm_pockets:
        logging.warning("PDB mapping failed for all papers.")
        return

    logging.info("Running Fpocket.")
    fpocket = FPocket()
    fpocket.run(pdb_path)

    logging.info("Merging & mapping pockets.")
    new_pockets, jaccard_matrix = recursive_merge_and_match(
        protein, 
        all_llm_pockets, 
        fpocket, 
        threshold_merge=config['threshold_merge'],
        threshold_residue=config['threshold_residue_match']
    )

    assert len(new_pockets) == jaccard_matrix.shape[1]

    logging.info("Creating pockets.")

    assigned_rows = get_pocket_match_mask(
        jaccard_matrix,
        config['threshold_pocket_match'],
        return_best_match=config['return_best_match']
    )

    for i, mask in enumerate(assigned_rows):
        indices = np.where(mask == 1)[0].tolist()
        if indices:
            fpocket_pockets = fpocket[indices]
            alpha_spheres = get_alpha_spheres(fpocket_pockets).to_numpy()

            for j in indices:
                fpocket_name = f"pocket{i}-{j}_alpha_spheres.mol2"
                alpha_spheres_out = path_dict['alpha_spheres_path'] / fpocket_name
                fpocket.to_mol2(alpha_spheres_out, indicies=[j])

            grid = create_grid(alpha_spheres, config['spacing'])

            corrected_grid = correct_volume_grid(grid, protein, alpha_spheres)
            corrected_grid = clean_grid(corrected_grid, protein, radius='auto')
            
            logging.info("Geometric pocket: {}, Num. grid points: {}, Matched pockets: {}".format(
                    new_pockets[i]['pocket_id'], len(corrected_grid), indices
            ))

            pocket_name = f"pocket{i}.mol2"
            pocket_name_original = f"pocket{i}_original.mol2"
            pocket_name_pqr = f"pocket{i}.pqr"

            save_path_mol2 = path_dict['pockets_path'] / pocket_name
            save_path_mol2_debug = path_dict['debug_path'] / pocket_name_original
            save_path_pqr = path_dict['alpha_spheres_path'] / pocket_name_pqr

            save_to_mol2(save_path_mol2, corrected_grid)
            save_to_mol2(save_path_mol2_debug, grid)
            save_to_pqr(save_path_pqr, fpocket, indices)
    
    debug_info = generate_debug_info(fpocket, assigned_rows)
    with open(str(path_dict['debug_path'] / "debug_info.json"), 'w') as f:
            json.dump(debug_info, f, indent=4, ensure_ascii=False)

    move_fpocket_output(pdb_path, path_dict['debug_path'])
    fpocket.to_mol2(path_dict['debug_path'] / "fpocket_out.mol2")
    logging.info("Completed.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process articles and PDB files with configurable settings.")

    parser.add_argument("article_path", help="Path to the article file.")
    parser.add_argument("pdb_path", help="Path to the PDB file.")
    parser.add_argument("target_protein", help="Name of the target protein.")

    parser.add_argument(
        "--config-file",
        help="Path to a YAML or JSON configuration file to override default settings. "
             "Settings in this file will be overridden by --set options."
    )

    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        help="Override a specific configuration value using dot notation "
             "(e.g., --set extraction.model=gpt-4.1-mini --set debug=True). "
             "These take precedence over config files."
    )

    args = parser.parse_args()

    current_config = copy.deepcopy(CONFIG)

    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                if args.config_file.endswith((".yaml", ".yml")):
                    file_config = yaml.safe_load(f)
                elif args.config_file.endswith(".json"):
                    file_config = json.load(f)
                else:
                    file_config = yaml.safe_load(f)
                
                if file_config:
                    deep_update(current_config, file_config)
        except FileNotFoundError:
            parser.error(f"Config file not found: {args.config_file}")
        except Exception as e:
            parser.error(f"Error loading config file {args.config_file}: {e}")

    if args.set:
        for override_str in args.set:
            if '=' not in override_str:
                parser.error(f"Invalid --set format: '{override_str}'. Expected KEY=VALUE.")
            key_path, value_str = override_str.split('=', 1)
            try:
                set_nested_value(current_config, key_path, value_str)
            except Exception as e:
                parser.error(f"Error setting value for '{key_path}': {e}")

    main(args.article_path, args.pdb_path, args.target_protein, current_config)