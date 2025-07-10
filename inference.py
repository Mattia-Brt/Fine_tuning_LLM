import sys
import argparse

from core.classify_models.llama_3B import LLAMAclass_3B
from core.classify_models.mistral import MISTRALclass_7B
from core.classify_models.llama2_7B import LLAMAclass_7B
from core.classify_models.llama3_8B import LLAMA3class_8B
from core.classify_models.gemma_7B import GEMMAclass_7B
from core.classify_models.gemma_2B import GEMMAclass_2B
from core.classify_models.recurrentGemma_2B import RecurrentGEMMAclass_2B
from core.classify_models.mistral_8x7B import MISTRALclass_8x7B
from core.classify_models.esgBERT import esgBERTclass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference.')

    # system
    parser.add_argument('--debug', action="store_true", help="Debug Mode to output detail info")

    # base
    parser.add_argument('--data', default=None, help="The DIR of test data", type=str)
    parser.add_argument('--model_type', default="llama_3B", choices=['esgBERT', 'llama_3B', 'llama2_7B', 'llama3_8B', 'mistral', 'gemma_7B', 'gemma_2B', 'recurrentGemma_2B'])
    parser.add_argument('--labels', default="[\"0\", \"1\"]",
                        help="Labels to classify, only used when task_type is classify")
    parser.add_argument('--adapter_weights', default="None", type=str, help="The DIR of adapter weights") #mettere la cartella output/dir dove sta salvato il modello
    parser.add_argument('--load_8bit', action="store_true")
    
    # generate
    parser.add_argument('--temperature', default="0.7", type=float, help="temperature higher, LLM is more creative")
    parser.add_argument('--top_p', default="0.9", type=float)
    parser.add_argument('--top_k', default="40", type=int)
    parser.add_argument('--max_new_tokens', default="512", type=int)
    
    #inference 
    parser.add_argument('--InferenceFolder', default="default", type=str)
    parser.add_argument('--EvalFolder', default="default", type=str)
    parser.add_argument('--LoadWeights', default="True", type=str)
    
    #huggingface 
    parser.add_argument('--TokenHugging', default="default", type=str)


    args, _ = parser.parse_known_args()
    
    #
    #
    #
    #
    #
    #check here what model you want to use call model 
    if args.model_type == "llama_3B":
        llm = LLAMAclass_3B()
    elif args.model_type == "llama2_7B":
        llm = LLAMAclass_7B()
    elif args.model_type == "llama3_8B":
        llm = LLAMA3class_8B()
    elif args.model_type == "mistral":
        llm = MISTRALclass_7B()
    elif args.model_type == "mistral_8x7B":
        llm = MISTRALclass_8x7B()
    elif args.model_type == "gemma_7B":
        llm = GEMMAclass_7B()
    elif args.model_type == "gemma_2B":
        llm = GEMMAclass_2B()
    elif args.model_type == "recurrentGemma_2B":
        llm = RecurrentGEMMAclass_2B()
    elif args.model_type == "esgBERT":
        llm = esgBERTclass()
    else:
        print("model_type should be llama_3B or llama2_7B or llama3_8B or mistral or gemma_7B or recurrentGemma_2B")
        sys.exit(-1)
        

    llm.debug = args.debug

    llm.adapter_weights = args.adapter_weights
    llm.model_type = args.model_type
    
    llm.load_8bit = args.load_8bit

    llm.temperature = args.temperature
    llm.top_p = args.top_p
    llm.top_k = args.top_k
    llm.max_new_tokens = args.max_new_tokens
    
    llm.InferenceFolder = args.InferenceFolder
    llm.EvalFolder = args.EvalFolder
    llm.LoadWeights = args.LoadWeights
    
    llm.TokenHugging = args.TokenHugging


    llm.generate(args.data)

