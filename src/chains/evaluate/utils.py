from artifacts.benchmark_dataset import BenchmarkDatasetArtifact

from langchain_core.runnables import RunnableLambda, Runnable
from .chain import match_pockets_runnable

def generate_extract_and_evaluate_chain_for_article(annotated_pockets, 
                                                    refine_chain, 
                                                    ) -> Runnable:
    def stich_chains_runnable(refine_chain_output):
        extracted_pockets = []

        if refine_chain_output is None:
            return []
        if refine_chain_output.pockets is None:
            return []
        for pocket in refine_chain_output.pockets:
            if pocket is not None:
                pocket_dict = pocket.model_dump()
                pocket_dict["amino_acids"] = [residue.to_string()
                                              for residue in pocket.amino_acids]
                extracted_pockets.append(pocket_dict)
        result = {
            "extracted_pockets": extracted_pockets,
            "annotated_pockets": annotated_pockets
        }
        if hasattr(refine_chain_output, 'relevance'):
            result["relevance"] = refine_chain_output.relevance
        return result
    return refine_chain | RunnableLambda(stich_chains_runnable) | match_pockets_runnable


def generate_chain_and_their_inputs(
    benchmark_dataset: BenchmarkDatasetArtifact, 
    article_path,
    target_protein,
    reader,
    refine_chain,
) -> tuple[Runnable, dict, dict]:
    article_data = benchmark_dataset.prepare_article_for_evaluation(
        article_path, reader,
    )[target_protein]
    chain = generate_extract_and_evaluate_chain_for_article(
        article_data['annotated_pockets'], refine_chain
    )
    chain_input = {
        "target_protein": target_protein,
        "context": article_data['text']
    }

    metadata = {
        "general": _prepare_general_metadata(article_path, target_protein, article_data),
        "PocketTableCallback": {
            "annotated_extra_data": article_data['annotated_pockets_extra_data'],
        },
        "AminoAcidsTableCallback": {
            "annotated_extra_data": article_data['annotated_amino_acids_extra_data']
        }
    }

    return chain, chain_input, metadata


def _prepare_general_metadata(article_path, target_protein, article_data):
    metadata = {'article_pdf_name': article_path}
    
    # Artcile doesn't have annotated pockets, so no target info of the article will be automatically added
    if len(article_data['annotated_pockets']) == 0:
        metadata['target'] = target_protein

    return metadata