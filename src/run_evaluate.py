from utils.wandb.tracer import CustomWandbTracer
from langchain.globals import set_verbose

from artifacts.benchmark_dataset import BenchmarkDatasetArtifact
from utils.readers import MarkdownReader
from chains import build_refine_chain, get_llm_model
from chains.refine.prompts.builder import RefineChainPromptsFromArtifactsBuilder
from chains.evaluate.callbacks import PocketTableCallback, AminoAcidsTableCallback
from chains.evaluate.utils import generate_chain_and_their_inputs
from chains.callbacks import PromptsUseArtifactCallback
from constants import (
    ARTICLES_WITH_POCKETS, 
    ARTICLES_DEBUG, 
    ARTICLES_POCKETS, 
    ARTICLES_WITHOUT_POCKETS
)
from dotenv import load_dotenv
import time

load_dotenv('keys.env')
set_verbose(True)

READER = MarkdownReader()
ARTICLES = ARTICLES_POCKETS

LLM_MODEL = "gpt-4o-mini-2024-07-18"

# gpt-4.1-mini-2025-04-14
# gemini-2.5-flash-preview-04-17
# gemini-2.0-pro-exp-02-05 
# gemini-2.0-flash-thinking-exp
# gemini-2.0-flash 
# gemini-1.5-pro-002
# gpt-4o-mini
# gpt-4o-mini-2024-07-18

CONFIG = {
    "extraction_model": LLM_MODEL, 
    "refine_model": LLM_MODEL,
    "parse_model": "gpt-4o-mini-2024-07-18",
    "filter_model": "gpt-4o-mini-2024-07-18",
    "refine_call": True,
    "filter_call": True,
    "max_tokens": {
        "extraction": 5000,
        "refine": 5000,
        "filter": 5000,
        "parse": 3000,
    },
    "extraction_chain": "refine",
    "articles": ARTICLES,
    "reader": READER.__class__.__name__,
    "prompts_versions": {
        "refine": "v68",  # "v71" or None
        "evaluate": None, # None,
    },
    "benchmark_dataset_version": None,
}

WANDB_RUN_ARGS = {
    "project": "AI-Assistant-Pocket-Extractor",
    "job_type": "debug",  # validation, debug, test, experimental, no_pockets_articles
    "name": f"{CONFIG['extraction_model']}; Refine: {CONFIG['refine_call']}; Full pipeline", # f"Exp. 7: All prompt updates included | model={CONFIG['extraction_model']}", 
    "group": 'Prompt improvement after initial baseline', # 'Evaluate the baseline and get initial metrics.',
    "tags": [CONFIG["extraction_model"]],
    "notes": "Get metrics", # "Test on GPT-4o",
    "config": CONFIG
}

# MODELS
llm_extraction = get_llm_model(
    CONFIG['extraction_model'], 
    max_tokens=CONFIG['max_tokens']['extraction']
)
llm_refine = get_llm_model(
    CONFIG['refine_model'], 
    max_tokens=CONFIG['max_tokens']['refine']
)
llm_parse = get_llm_model(
    CONFIG['parse_model'], 
    max_tokens=CONFIG['max_tokens']['parse']
)
llm_filter = get_llm_model(
    CONFIG['filter_model'], 
    max_tokens=CONFIG['max_tokens']['filter']
)

# PROMPTS

refine_chain_prompts_builder = RefineChainPromptsFromArtifactsBuilder(
    project=WANDB_RUN_ARGS['project'],
    version=CONFIG["prompts_versions"]["refine"],
    automatically_update_version=True
)

benchmark_dataset = BenchmarkDatasetArtifact(
    folder="../../LLM-benchmark-dataset",
    project=WANDB_RUN_ARGS['project'],
    version=CONFIG["benchmark_dataset_version"],
    automatically_update_version=True
)

# CHAINS

refine_chain = build_refine_chain(
    llm_extraction,
    llm_refine,
    llm_parse,
    llm_filter,
    refine_chain_prompts_builder, 
    use_refine_call=CONFIG['refine_call'],
    use_filter_call=CONFIG['filter_call']
)

tracer = CustomWandbTracer(
    trace_columns_functions=None,
    callbacks=[
        PocketTableCallback(),
        AminoAcidsTableCallback(),
        PromptsUseArtifactCallback([
            refine_chain_prompts_builder,
            benchmark_dataset
        ])
    ],
    run_args=WANDB_RUN_ARGS
)


for article, target in ARTICLES:
    chain, chain_input, metadata = generate_chain_and_their_inputs(
        benchmark_dataset, article, target, READER, refine_chain
    )
    r = chain.invoke(
        chain_input,
        {
            "callbacks": [tracer],
            "tags": [],
            "metadata": metadata
        },
    )

tracer.finish()
