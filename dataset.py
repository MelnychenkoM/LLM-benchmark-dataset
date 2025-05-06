import os
import pandas as pd
import json
import argparse

ROOT_DIR = "."
ARTICLES_DIR = os.path.join(ROOT_DIR, "articles")
POCKETS_DIR = os.path.join(ROOT_DIR, "pockets")


class BenchmarkDatasetUpdater:
    def __init__(self, root_dir="../../LLM-benchmark-dataset-paper"):
        self.root_dir = root_dir
        self.pockets_dir = os.path.join(self.root_dir, "pockets")

    def update_pockets_excel_to_json(self):
        residues = pd.read_excel(os.path.join(self.root_dir, "tables/amino_acids.xlsx"))
        pockets = pd.read_excel(os.path.join(self.root_dir, "tables/pockets.xlsx"))

        df_merged = pd.merge(pockets, residues, 
                             on=["target", "paper_id", "pocket_id"], 
                             how="left"
                             )
        for (target, paper_id), group in df_merged.groupby(["target", "paper_id"]):
            pockets = []

            target_folder = os.path.join(self.pockets_dir, group.folder_name.values[0])
            filename = paper_id

            json_filename = filename + ".json"
            json_filepath = os.path.join(target_folder, json_filename)

            doi = group['DOI'].values[0]
            paper_name = group['paper_name'].values[0]

            for pocket_id, sub_group in group.groupby("pocket_id", dropna=False):
                description = sub_group["pocket_description"].dropna().unique()
                amino_acids = sub_group["amino_acid"].dropna().tolist()
                ligands = list(set(l for ligand_list in sub_group["ligand"].dropna().unique() 
                                   for l in ligand_list.split(", ")
                                    )
                                )
                if pd.isnull(pocket_id):
                    continue

                pockets.append({
                    "pocket_id": pocket_id.strip('"'),
                    "description": description[0] if len(description) > 0 else "",
                    "amino_acids": amino_acids,
                    "ligands": ligands,
                })

            with open(json_filepath, "w") as json_file:
                json.dump(
                    {
                        "target": target, 
                        "paper_name": paper_name,
                        "DOI": doi,
                        "paper_id": filename, 
                        "pockets": pockets
                    }, 
                    json_file, 
                    indent=4, 
                    ensure_ascii=False
                )
        return

    def update_residues_json_to_excel(self, filename="tables/amino_acids.xlsx"):
        data = []
        output_path = os.path.join(self.root_dir, filename)
        
        for folder in os.listdir(self.pockets_dir):
            pockets_folder_path = os.path.join(self.pockets_dir, folder)
            
            if not os.path.isdir(pockets_folder_path):
                continue
            
            for paper in os.listdir(pockets_folder_path):

                if paper.endswith(".json"):
                    json_file_path = os.path.join(pockets_folder_path, paper)

                    with open(json_file_path, "r") as json_file:
                        json_data = json.load(json_file)
                        target = json_data.get("target", "")
                        paper_id = json_data.get("paper_id", "")
                        pockets = json_data.get("pockets", [])

                        for pocket in pockets:
                            pocket_id = pocket.get("pocket_id", "")
                            amino_acids = pocket.get("amino_acids", [])
                            ligands = pocket.get("ligands", [])
                            ligands_str = ", ".join(ligands) if ligands else None
                            
                            for aa in amino_acids:
                                data.append([
                                    paper_id,
                                    target, 
                                    pocket_id, 
                                    aa, 
                                    ligands_str,
                                ])
        df = pd.DataFrame(data, columns=[
            "paper_id", 
            "target", 
            "pocket_id", 
            "amino_acid",
            "ligand"
        ])
        df.to_excel(output_path, index=False)
        return
    
    def update_pockets_json_to_excel(self, filename="tables/pockets.xlsx"):
        data = []
        output_path = os.path.join(self.root_dir, filename)
        
        for folder in os.listdir(self.pockets_dir):
            pockets_folder_path = os.path.join(self.pockets_dir, folder)
            
            if not os.path.isdir(pockets_folder_path):
                continue
            
            for paper in os.listdir(pockets_folder_path):

                if paper.endswith(".json"):
                    json_file_path = os.path.join(pockets_folder_path, paper)
                    
                    with open(json_file_path, "r") as json_file:
                        json_data = json.load(json_file)
                        target = json_data.get("target", "")
                        paper_id = json_data.get("paper_id", "")
                        paper_name = json_data.get("paper_name", "")
                        doi = json_data.get("DOI", "")
                        pockets = json_data.get("pockets", [])
                    
                        if pockets:
                            for pocket in pockets:
                                data.append([
                                    paper_id,
                                    paper_name, 
                                    doi,
                                    target, 
                                    "TRUE",
                                    pocket.get("pocket_id", ""),
                                    pocket.get("description", ""),
                                    ", ".join(pocket.get("ligands", [])),
                                    folder
                                ])
                        else:
                            data.append([
                                paper_id,
                                paper_name,
                                doi, 
                                target, 
                                "FALSE", 
                                "", 
                                "", 
                                "",
                                folder
                            ])
        
        df = pd.DataFrame(data, columns=[
            "paper_id", 
            "paper_name",
            "DOI",
            "target", 
            "article_contains_pocket_description", 
            "pocket_id", 
            "pocket_description", 
            "ligands",
            "folder_name"
        ])
        df.to_excel(output_path, index=False)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute specific functions.")
    parser.add_argument('--update_pockets_excel_to_json', action='store_true', help="Update pockets from Excel to JSON.")
    parser.add_argument('--update_residues_json_to_excel', action='store_true', help="Update residues from JSON to Excel.")
    parser.add_argument('--update_pockets_json_to_excel', action='store_true', help="Update pockets from JSON to Excel.")
    
    args = parser.parse_args()

    dataset = BenchmarkDatasetUpdater(ROOT_DIR)

    if args.update_pockets_excel_to_json:
        dataset.update_pockets_excel_to_json()
    if args.update_residues_json_to_excel:
        dataset.update_residues_json_to_excel()
    if args.update_pockets_json_to_excel:
        dataset.update_pockets_json_to_excel()