from langchain_core.runnables import RunnableLambda, RunnablePassthrough, chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI

from langchain.output_parsers import (
    PydanticOutputParser, RetryWithErrorOutputParser, 
)
from langchain.prompts import PromptTemplate

from typing import List, Union, TypedDict
from .prompts.builder import RefineChainPromptsFromArtifactsBuilder
from .pockets import (
    ExtractedPockets,
    ExtractedPocketsWithRelevance,
    PaperRelevance
)
from pydantic import ValidationError
import re

import logging
logger = logging.getLogger(__name__)


def build_llm_pipeline(
        llm_filter,
        llm_extraction,
        llm_refine,
        llm_parse,
        prompts_builder: RefineChainPromptsFromArtifactsBuilder,
        use_refine_call=False,
        use_filter_call=False,
        logger=None
    ) -> Union[ExtractedPockets, None]:
    """
    Main function that extracts the binding sites from the provided scientific paper.

    Parameters:
    - llm_extraction: The language model used for the initial extraction of binding sites.
    - llm_refine: The language model used for refining the extracted binding sites.
    - prompts_builder: An instance of RefineChainPromptsFromArtifactsBuilder used to build prompts for the language models.
    - refine_call: A boolean indicating whether to perform the refine call.
    - filter_call: A boolean indicating whether to perform the filter call.

    Returns:
    - ExtractedPockets: A pydantic model containing the list of extracted binding sites.
    - None: If the paper is not relevant.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    extraction_call = build_extraction_call_runnable(llm_extraction, prompts_builder, logger)
    refine_call = build_refine_call_runnable(llm_refine, prompts_builder, logger)
    parse_call = build_retry_parser_call_runnable(llm_parse, prompts_builder, logger)
    filter_call = build_filter_call_runnable(llm_filter, llm_parse, prompts_builder, logger)

    chain = extraction_call | (refine_call if use_refine_call else RunnablePassthrough()) | parse_call

    if use_filter_call:
        def combined_processing(inputs):
            filtered_inputs = filter_call.invoke(inputs)
            if filtered_inputs is None or isinstance(filtered_inputs, ExtractedPockets):
                return filtered_inputs
            parsed_result = chain.invoke(filtered_inputs)
            if parsed_result is None:
              return None
            return ExtractedPocketsWithRelevance(
                pockets=parsed_result.pockets,
                relevance=filtered_inputs.get("relevance"),
                refine=filtered_inputs.get("refine")
            )

        return RunnableLambda(combined_processing)
    else:
        func = lambda x: \
            ExtractedPocketsWithRelevance(pockets=x.pockets) if \
            x is not None else x
        return chain | RunnableLambda(func)
    

def build_extraction_call_runnable(
        llm,
        prompts_builder: RefineChainPromptsFromArtifactsBuilder,
        logger
    ) -> RunnableLambda:
    """
    Builds runnable that extracts the binding site information from the provided paper and target protein.

    Parameters:
    - llm: The language model used for the extraction process.
    - prompts_builder: An instance of RefineChainPromptsFromArtifactsBuilder used to build prompts for the language model.

    Returns:
    - RunnableLambda: A runnable lambda function that performs the extraction.
    """
    extraction_instructions = \
        prompts_builder.build_prompt('EXTRACTION_INSTRUCTIONS')
    format_instructions = \
        prompts_builder.build_prompt('FORMAT_INSTRUCTIONS')

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts_builder.build_prompt('SYSTEM_MESSAGE')),
        ("human", prompts_builder.build_prompt('INITIAL_CALL_PROMPT'))
        ]).partial(
            extraction_instructions=extraction_instructions,
            format_instructions=format_instructions
        )
    chain = prompt | llm

    def initial_call_runnable(inputs: dict) -> Union[dict, ExtractedPockets, None]:
        if inputs is None or isinstance(inputs, ExtractedPockets):
            return inputs
        logger.info("Invoking LLM [EXTRACTION]")
        response = chain.invoke({
            "target_protein": inputs['target_protein'],
            "context": inputs['context']
        })
        if not response.content:
            return None
        inputs['content'] = response.content
        return inputs
    return RunnableLambda(initial_call_runnable)


def build_refine_call_runnable(
        llm,
        prompts_builder: RefineChainPromptsFromArtifactsBuilder,
        logger
    ) -> RunnableLambda:
    """
    Builds runnable that refines the existing answer.

    Parameters:
    - llm: The language model used for the refinement process.
    - prompts_builder: An instance of RefineChainPromptsFromArtifactsBuilder used to build prompts for the language model.

    Returns:
    - RunnableLambda: A runnable lambda function that performs the refinement.
    """
    extraction_instructions = \
        prompts_builder.build_prompt('EXTRACTION_INSTRUCTIONS')
    format_instructions = \
        prompts_builder.build_prompt('FORMAT_INSTRUCTIONS')
    
    prompt = ChatPromptTemplate.from_messages([
            ("system", prompts_builder.build_prompt('SYSTEM_MESSAGE')),
            ("human", prompts_builder.build_prompt('REFINE_CALL_PROMPT'))
        ]).partial(
            extraction_instructions=extraction_instructions,
            format_instructions=format_instructions
        )
    
    chain = prompt | llm

    def refine_runnable(inputs: dict) -> Union[dict, ExtractedPockets, None]:
        if inputs is None or isinstance(inputs, ExtractedPockets):
            return inputs
        logger.info("Invoking LLM [REFINE]")
        response = chain.invoke({
            "context": inputs["context"],
            "target_protein": inputs["target_protein"],
            "extraction_result": inputs["content"],
        })

        def get_refine_results(response: str) -> str:
            match = re.search(r'(# Summary[\s\S]*)', response)
            txt = match.group(1) if match else None
            return txt
        
        if not response.content:
            return None
        inputs['content'] = response.content
        inputs['refine'] = get_refine_results(response.content)
        return inputs
    return RunnableLambda(refine_runnable)


def build_retry_parser_call_runnable(
        llm,
        prompts_builder: RefineChainPromptsFromArtifactsBuilder,
        logger,
        max_retries: int = 2,
    ) -> RunnableLambda:
    """
    Builds runnable that tries to parse the existing answer into the pydantic objects.

    Parameters:
    - llm: The language model used for the parsing process.
    - prompts_builder: An instance of RefineChainPromptsFromArtifactsBuilder used to build prompts for the language model.

    Returns:
    - RunnableLambda: A runnable lambda function that performs the parsing.
    """
    retry_correction_prompt = \
        prompts_builder.build_prompt('RETRY_CORRECTION_PROMPT')
    parser = PydanticOutputParser(pydantic_object=ExtractedPockets)

    def parse_runnable(inputs: dict, max_retries: int = max_retries) -> ExtractedPockets:
        if inputs is None or isinstance(inputs, ExtractedPockets):
            return inputs

        correction_prompt_value = PromptTemplate.from_template(retry_correction_prompt).format_prompt(
                format_instructions=parser.get_format_instructions(),
                llm_output=inputs['content']
            )
        logger.info("Invoking LLM [PARSE]")
        retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=parser, llm=llm, max_retries=max_retries
        )
        try: 
            response = retry_parser.parse_with_prompt(
                inputs['content'],
                correction_prompt_value
        )
        except ValidationError as e:
            logger.warning(f"Pydantic validation error. Exiting.")
        return response
    return RunnableLambda(parse_runnable)


def build_filter_call_runnable(
        llm_filter,
        llm_parse,
        prompts_builder: RefineChainPromptsFromArtifactsBuilder,
        logger
    ) -> RunnableLambda:
    """
    Builds runnable that filters the papers based on relevance.

    Parameters:
    - llm: The language model used for the filtering process.
    - prompts_builder: An instance of RefineChainPromptsFromArtifactsBuilder used to build prompts for the language model.

    Returns:
    - RunnableLambda: A runnable lambda function that performs the filtering and returns the pydantic object.
    """
    retry_correction_prompt = \
        prompts_builder.build_prompt('RETRY_CORRECTION_PROMPT')

    def retry_parser_call(response: dict, 
                          max_retries=2
                          ) -> PaperRelevance:
        parser = PydanticOutputParser(pydantic_object=PaperRelevance)
        correction_prompt_value = PromptTemplate.from_template(retry_correction_prompt).format_prompt(
            format_instructions=parser.get_format_instructions(),
            llm_output=response.content
        )
        retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=parser, llm=llm_parse, max_retries=max_retries
        )
        return retry_parser.parse_with_prompt(
                response.content,
                correction_prompt_value
        )

    parser = PydanticOutputParser(pydantic_object=PaperRelevance)
    prompt = ChatPromptTemplate.from_messages([
            ("system", prompts_builder.build_prompt('SYSTEM_MESSAGE_FILTER')),
            ("human", prompts_builder.build_prompt('FILTER_PAPERS_PROMPT'))
        ]).partial(
            format_instructions=parser.get_format_instructions()
        )
    chain = prompt | llm_filter | RunnableLambda(retry_parser_call)
    
    def filter_runnable(inputs):
        logger.info("Invoking LLM [FILTER]")
        response = chain.invoke({
            "target_protein": inputs['target_protein'],
            "context": inputs['context']
        })
        if response.is_relevant:
            logger.info("Filter results: RELEVANT.")
            inputs['relevance'] = response.dict()
            return inputs
        logger.info("Filter results: NOT RELEVANT.")
        return ExtractedPocketsWithRelevance(**{
            "pockets": [],
            "relevance": response.dict()
        })
    return RunnableLambda(filter_runnable)
    

def get_llm_model(model: str,
                  temperature: int = 0,
                  max_tokens: int = 1000,
                  max_retries: int = 1
                  ) -> Union[ChatVertexAI, ChatOpenAI]:
    if "gemini" in model.lower():
        return ChatGoogleGenerativeAI(model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            max_retries=max_retries
                            )
    elif "gpt" in model.lower():
        return ChatOpenAI(model=model,
                          temperature=temperature,
                          max_tokens=max_tokens,
                          max_retries=max_retries
                          )
    else:
        raise ValueError("Unsupported model.")
    

def write_pockets_to_md(data: dict, 
                        filename: str = "pockets_output.md"
                        ) -> None:
    """
    Creates a Markdown file and formats the given pockets data into it.

    Args:
        data (dict): A dictionary containing pocket and relevance information.
        filename (str): The name of the Markdown file to create. Defaults to "pockets_output.md".
    """
    def capitalize(string):
        return ' '.join(word.capitalize() for word in string.split())
    try:
        with open(filename, "w", encoding="utf-8") as f:
            if "relevance" in data and data["relevance"]:
                f.write(f"# {data['relevance']['name']}\n")
                f.write(f"{data['relevance']['reasoning']}\n")
            if "pockets" in data and data["pockets"]:
                f.write(f"# Binding Sites\n")
                for pocket in data["pockets"]:
                    f.write(f"##  {capitalize(pocket['pocket_id'])}\n")
                    f.write(f"## Description: \n{pocket['description']}\n")
                    f.write("## Residues:\n")
                    for amino_acid in pocket["amino_acids"]:
                        chain_id = amino_acid.get("chain_id")
                        res_name = amino_acid["res_name"]
                        res_id = amino_acid["res_id"]
                        if chain_id:
                            f.write(f"- {chain_id}{res_name}{res_id}\n")
                        else:
                            f.write(f"- {res_name}{res_id}\n")
                    f.write("\n")
    except Exception as e:
        print(f"Error creating Markdown file: {e}")
