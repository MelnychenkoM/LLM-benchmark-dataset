from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import List, Union, TypedDict, Optional

class Residue(BaseModel):
    chain_id: Union[str, None] = Field(default=None, description="Optional chain identifier.")
    res_name: str = Field(description="Residue name in the paper: Ala or A or ALA etc.")
    res_id: int = Field(description="Residue number in the paper.")
    
    def to_string(self):
        if self.chain_id:
            return f"{self.chain_id}{self.res_name}{self.res_id}"
        return f"{self.res_name}{self.res_id}"
    
    @model_validator(mode='before')
    def convert_res_id(cls, values):
        res_id = values.get('res_id')
        if isinstance(res_id, str):
            values['res_id'] = int(res_id)
        return values
    
    def __eq__(self, other):
        if isinstance(other, Residue):
            return (self.res_name == other.res_name and
                    self.res_id == other.res_id)
        return False
    
    def __hash__(self):
        return hash((self.res_name, self.res_id))


class Pocket(BaseModel):
    pocket_id: str = Field(
        description="Binding site name.")
    description: str = Field(
        description="Short binding site description."
    )
    amino_acids: List[Residue] = Field(
        description="List of residues that make up the binding site.", 
        default=[]
    )

    def __and__(self, other):
        if isinstance(other, Pocket):
            overlap_residues = [residue for residue in self.amino_acids if residue in other.amino_acids]
            return Pocket(
                pocket_id=f"{self.pocket_id}&{other.pocket_id}",
                description=f"Overlap between {self.pocket_id} and {other.pocket_id}",
                amino_acids=overlap_residues
            )
        raise TypeError("Unsupported operand type(s) for &: 'Pocket' and '{}'".format(type(other).__name__))
    
    def __len__(self):
        return len(self.amino_acids)


class ExtractedPockets(BaseModel):
    pockets: List[Pocket] = Field(
        description="List of found binding sites in the paper."
    )
    
    @model_validator(mode='before')
    def filter_empty_pockets(cls, values):
        pockets = values.get('pockets', [])
        filtered_pockets = []
        for pocket in pockets:
            if isinstance(pocket, dict):
                pocket = Pocket(**pocket)
            valid_amino_acids = [aa for aa in pocket.amino_acids if aa.res_id is not None]
            if valid_amino_acids:
                pocket.amino_acids = valid_amino_acids
                filtered_pockets.append(pocket)
        values['pockets'] = filtered_pockets
        return values
    
    @model_validator(mode='after')
    def merge_pockets(cls, values, threshold=0.3):
        def calculate_iou(pocket1, pocket2):
            set1 = set(pocket1.amino_acids)
            set2 = set(pocket2.amino_acids)
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            return len(intersection) / len(union)

        pockets = values.pockets
        merged_pockets = []
        while pockets:
            pocket1 = pockets.pop(0)
            if isinstance(pocket1, dict):
                pocket1 = Pocket(**pocket1)
            to_merge = []
            for pocket2 in pockets:
                if isinstance(pocket2, dict):
                    pocket2 = Pocket(**pocket2)
                iou = calculate_iou(pocket1, pocket2)
                if iou > threshold:
                    to_merge.append(pocket2)
            for pocket2 in to_merge:
                pockets.remove(pocket2)
                pocket1 = Pocket(
                    pocket_id=f"{pocket1.pocket_id} & {pocket2.pocket_id}",
                    description=f"{pocket1.description} & {pocket2.description}",
                    amino_acids=list(set(pocket1.amino_acids + pocket2.amino_acids))
                )
            merged_pockets.append(pocket1)
        values.pockets = merged_pockets
        return values


class ExtractedPocketsWithRelevance(ExtractedPockets):
        relevance: Optional[dict] = Field(default=None, description="Relevance of the paper")
        refine: Optional[str] = Field(default=None, description="Refine call results.")


class PaperRelevance(BaseModel):
    name: str = Field(description="Paper name.")
    is_relevant: bool = Field(description="Is this paper relevant to the question.")
    reasoning: str = Field(description="Evidence to support the claim.")