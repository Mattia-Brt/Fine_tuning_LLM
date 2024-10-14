import os
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune')
    #example of input
    #python finetune.py --model_type llama --data "data/train/" --model_path "LLMs/open-llama/open-llama-3b/" --adapter "lora" --output_dir "output/llama"

    #
    #
    #
    #
    #
    #take here parameter from input
    # base
    parser.add_argument('--data', type=str, default="data/train/", help='the data used for instructing tuning')
    parser.add_argument('--model_type', default="llama_3B", choices=['llama_3B', 'llama2_7B', 'llama3_8B', 'mistral', 'mistral_8x7B', 'gemma_7B', 'gemma_2B', 'recurrentGemma_2B'])
    parser.add_argument('--labels', default="[\"0\", \"1\"]", help="Labels to classify") #aggiungila sotto perché non c'è
    parser.add_argument('--output_dir', default="output/", type=str, help="The DIR to save the model")
    parser.add_argument('--input_dir', default="default", type=str, help="The DIR get the weights of a pretrained model")
    parser.add_argument('--n_splits', default=10, type=int)
    parser.add_argument('--modality', default="cross", choices=['cross', 'normal', 'crossANDnormal'])
    parser.add_argument('--data2', type=str, default="data/train/", help='the second dataset used (in our case only for esgBert+GS)')



    # adapter
    parser.add_argument('--adapter', default="lora", choices=['lora', 'qlora', 'adalora'])
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--lora_target_modules', nargs='+',
                        help="the module to be injected, e.g. q_proj/v_proj/k_proj/o_proj for llama",
                        default=None)
    parser.add_argument('--adalora_init_r', default=12, type=int)
    parser.add_argument("--adalora_tinit", type=int, default=200,
                        help="number of warmup steps for AdaLoRA wherein no pruning is performed")
    parser.add_argument("--adalora_tfinal", type=int, default=1000,
                        help=" fix the resulting budget distribution and fine-tune the model for tfinal steps when using AdaLoRA ")
    parser.add_argument("--adalora_delta_t", type=int, default=10, help="interval of steps for AdaLoRA to update rank")
    parser.add_argument('--num_virtual_tokens', default=20, type=int)
    parser.add_argument('--mapping_hidden_dim', default=128, type=int)
    parser.add_argument('--patient', default=5, type=int)

    # train
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--epochsCross', default=3, type=int)
    parser.add_argument('--max_steps', default=-1, type=int)
    
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--cutoff_len', default=51200, type=int)
    parser.add_argument('--val_set_size', default=0.2, type=float)
    parser.add_argument('--group_by_length', action="store_true")
    parser.add_argument('--logging_steps', default=5, type=int)

    parser.add_argument('--load_8bit', action="store_true") #action="store_true", quando viene passata come parametro la imposta in automatico a true
    parser.add_argument('--add_eos_token', action="store_true") #action="store_true", quando viene passata come parametro la imposta in automatico a true
    parser.add_argument('--resume_from_checkpoint', nargs='?', default=None, const=True,
                        help='resume from the specified or the latest checkpoint, e.g. `--resume_from_checkpoint [path]` or `--resume_from_checkpoint`')
    parser.add_argument('--per_gpu_train_batch_size', default=4, type=int, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--gradient_accumulation_steps', default=32, type=int)

    #eval 
    parser.add_argument('--EvalFolder', default="default", type=str)
    
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
    else:
        print("model_type should be llama_3B or llama2_7B or llama3_8B or mistral or gemma_7B or recurrentGemma_2B")
        sys.exit(-1)
            


    #
    #
    #
    #
    #
    # association of params to variables
    llm.data_path = args.data
    llm.model_type = args.model_type
    llm.output_dir = args.output_dir
    llm.input_dir = args.input_dir
    llm.n_splits = args.n_splits
    llm.modality = args.modality
    llm.data_path2 = args.data2

    llm.adapter = args.adapter
    llm.lora_r = args.lora_r
    llm.lora_alpha = args.lora_alpha
    llm.lora_dropout = args.lora_dropout
    llm.lora_target_modules = args.lora_target_modules
    llm.adalora_init_r = args.adalora_init_r
    llm.adalora_tinit = args.adalora_tinit
    llm.adalora_tfinal = args.adalora_tfinal
    llm.adalora_delta_t = args.adalora_delta_t

    llm.num_virtual_tokens = args.num_virtual_tokens
    llm.mapping_hidden_dim = args.mapping_hidden_dim
    llm.epochs = args.epochs
    llm.epochsCross = args.epochsCross
    llm.max_steps = args.max_steps
    llm.learning_rate = args.learning_rate
    llm.cutoff_len = args.cutoff_len
    llm.val_set_size = args.val_set_size
    llm.group_by_length = args.group_by_length
    llm.logging_steps = args.logging_steps
    llm.patient = args.patient

    llm.load_8bit = args.load_8bit
    llm.add_eos_token = args.add_eos_token
    llm.resume_from_checkpoint = args.resume_from_checkpoint
    llm.per_gpu_train_batch_size = args.per_gpu_train_batch_size
    llm.gradient_accumulation_steps = args.gradient_accumulation_steps

    llm.EvalFolder = args.EvalFolder
    
    llm.TokenHugging = args.TokenHugging

    if not os.path.exists(llm.output_dir):
        os.makedirs(llm.output_dir)
        print("Warning: Directory {} Not Found, create automatically")

    llm.finetune()

