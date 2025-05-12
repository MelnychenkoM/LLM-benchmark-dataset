# System instructions

SYSTEM_MESSAGE = """\
1. Act as a senior structural biologist proficient in the analysis of protein-ligand complexes.
2. Your task is to analyze the research paper that will be given to you and to find descriptions of all binding sites for small molecules in the target protein {target_protein}, which are mentioned in this paper.
3. You should ignore any other proteins except target protein.
4. For target protein you should ignore descriptions of the binding sites for any ligands, which are not small molecules, such as other proteins, antibodies, or nucleic acids.
5. You should analyze everything in the paper, which describes the binding of small molecules to target protein and mentions concrete amino acid residues.
6. The binding sites could also be called binding pockets, interaction sites, binding motifs, interaction motifs, etc. Also there might be other mentions of amino acid residues involved in binding of small molecule ligands.
7. You should be very precise and make sure that you extract only amino acid residues constituting the binding sites of the target protein, not something else that looks similar. 
8. You must not include protein functional elements other than binding sites for small molecules.
9. You should never make up or hypothesize anything! Your answer should be based strictly on the information from the provided paper.
10. For each correctly extracted binding site you will be rewarded by 1K USD.
11. For each incorrectly extracted binding site you will be fined by 1K USD.
"""

SYSTEM_MESSAGE_FILTER = """
You are a senior structural biologist with expertise in analyzing scientific papers on protein-ligand complexes.
Your task is to evaluate the relevance of scientific papers based on specific criteria.
"""

# Extraction and refinement instructions

INITIAL_CALL_PROMPT = """\
{extraction_instructions}
<article>
{context}
</article>
{format_instructions}
"""

REFINE_CALL_PROMPT = """\
The original task was as follows: {extraction_instructions}
The previous result of this task is: {extraction_result}
<article>
{context}
</article>
Your task is to find mistakes and fix them. Be very strict with your reasoning. Do the following:
1. Analyze the scientific paper text indicated by <article> HTML tags and the previous result.
2. Be very attentive to each point made in the previous instructions and check the correctness of the results.
3. If there are missed amino acids in the binding sites, add them. Look explicitly for the residues that represent binding sites for small molecule ligands only. 
4. Combine binding sites if they represent the same one based on the context and/or amino acid overlap. 
        They are typically described in the context of binding to drugs, toxins, inhibitors, activators, 
        agonists, antagonists or other small molecule ligands.
5. Exclude any identified binding sites that are not intended for small molecule ligands. 
            Specifically, remove sites that bind RNA, DNA, proteins, antibodies, or large peptides.
6. For each missed residue found, you will be paid 500 USD.
7. If you successfully merge the same binding sites or remove incorrect ones, you will be paid 1K USD.
8. If you fail to find a missed residue, you will be fined 300 USD for each.
9. For each incorrect merging or missed merging, you will be fined 1K USD
10. The binding site result should be strictly in the following format:
{format_instructions}\
11. The thinking result should be in the following format:
# Summary of changes
## Added residues:
    - Residue: Reasoning
    - Residue: Reasoning
## Removed residues:
    - Residue: Reasoning
## Merged binding sites:
    - Binding site names: Reasoning
## Removed binding sites:
    - Binding site name: Reasoning
## Overall result
[Your thoughts on target, binding sites, residues, etc. Should be 1-4 sentences.]
"""

EXTRACTION_INSTRUCTIONS = """\
There is a scientific paper text to analyze indicated by <article> html tags. Do the following:
1. Determine the number of unique binding sites for small molecules in target protein "{target_protein}" that are described in the text.
2. Provide a very laconic, very specific and discriminative characteristic for each binding site that you identified.
3. If the context permits, include any relevant small molecules in the binding site description, location, and target protein.
4. Output the list of amino acid residues constituting each of identified binding sites. Use the following notation for defining amino acid residues: <chain_id><res_name><res_id>.
    4.1 chain_id is an optional chain identifier for each amino acid. Crucially, if the paper specifies a chain ID for a residue (e.g., α, β, γ, A, B, C), you must include that chain ID in the output.
    4.2 res_name is the amino acid name in either single-letter (A123) or three letter notation (Ala123). Use the same notation as in the paper.
    4.3 res_id is the integer amino acid number.
    4.4 Some residues may be followed by a punctuation mark or a space. Any characters that follow them are not part of the residue.
    Examples:
    Input Text: "Glu89" -- Extracted Residue: {'chain_id: None, res_name: Glu, res_id: 89'}
    Input Text: "A134" -- Extracted Residue: {'chain_id: None, res_name: A, res_id: 134'}
    Input Text: "GLY1204" -- Extracted Residue: {'chain_id: None, res_name: GLY, res_id: 1204'}
    Input Text: "Ala123 ELC" -- Extracted Residue: {'chain_id: None, res_name: Ala, res_id: 123'}
    Input Text: "γPhe222" -- Extracted Residue: {'chain_id: γ, res_name: Phe, res_id: 222'}
    Input Text: "αY45" -- Extracted Residue: {'chain_id: α, res_name: Y, res_id: 45'}
    Input Text: "Tyr101C" -- Extracted Residue: {'chain_id: C, res_name: Tyr, res_id: 101'}
    Input Text: 'β2M286' -- Extracted Residue: {'chain_id': 'β2', 'res_name': 'M', 'res_id': 286'}
    Input Text: 'α1S270' -- Extracted Residue: {'chain_id': 'α1', 'res_name': 'S', 'res_id': 270'}
5. Ensure that each binding site is unique. Carefully analyze the article to avoid splitting a single binding site into multiple smaller ones without justification.
6. A binding site could bind several small molecules, but it is still considered the same binding site. Do not create separate entries for each small molecule if they bind to the same site.
"""

FORMAT_INSTRUCTIONS = """
The output should be a JSON object with the following structure:
    {
        "pockets": [
            {
                "pocket_id": "string",  # Descriptive binding site name.
                "description": "string",  # Conside binding site description. Should be 2-3 sentences.
                "amino_acids": [
                    {
                        "chain_id": None or "char", # Examples: None, A, B, γ, α;
                        "res_name": "string", # Examples: A, Ala, GLY;
                        "res_id": "integer", # 10, 1023, 1;
                    }
                ]
            }
        ]
    }
"""

# Output restructure instructions

RETRY_CORRECTION_PROMPT = """\
The previous response was not in the correct format. Please correct it to match the following JSON structure:
{format_instructions}
1. If there's a missing residue name - do not include it.
2. 1f there's a missing residue id - do not include it.
Here is the previous response:
{llm_output}
"""


SYSTEM_MESSAGE_FILTER = """
You are a senior structural biologist with expertise in analyzing scientific papers on protein-ligand complexes.
Your task is to evaluate the relevance of scientific papers based on specific criteria.
"""

FILTER_PAPERS_PROMPT = """
You will be provided with the text of a scientific paper enclosed within <article> HTML tags. 
Your task is to determine whether the paper meets the following criteria:

1. Does the paper describe the target protein, {target_protein}?
2. Does the paper include a description of the small molecules binding site for the {target_protein}?
3. Does the paper provide structural details, such as specific residues of the binding pocket (e.g., L123, Arg31)
or detailed binding site description for these molecules?

Consider the {target_protein} ONLY, not any other.
To be considered a binding site description, the paper must explicitly name the small molecules or classes that bind to it.
If the answer to all three questions is 'Yes', return True. Otherwise, return False.

Additionally, provide a detailed reasoning and evidence for each decision you make.
The output should be strictly in the following format:
{format_instructions}
The text:
<article>
{context}
<article>
"""
# Small molecules include single nucleotides and nucleosides, but exclude DNA/RNA chains.