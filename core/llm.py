import os
import csv
import torch
import json

from peft import (
    LoraConfig,
    AdaLoraConfig,
    TaskType,
    get_peft_model,
)

from typing import List
from datasets import load_dataset
from datetime import datetime
from sklearn.model_selection import KFold


class LLM:
    #
    #LLM PARAMS
    #
    #
    # system
    debug: bool = False

    # base params
    model_type: str = "llama_3B"
    modality: str = "cross"
    data_path: str = "data/train"
    data_path2: str = "data/train"
    labels: list = ["0", "1"]
    output_dir: str = "./output"
    input_dir: str = "default"
    disable_wandb: bool = False
    n_splits: int = 10

    # adapter params
    adapter: str = "prefix"
    adapter_weights: str = "output/chatglm"

    # lora hyperparams
    lora_r: int = 8     #16 verifica questi
    lora_alpha: int = 32    #16 verifica questi
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = [
        "query_key_value"
    ]
    

    # adalora hyperparams
    adalora_init_r: int = 12
    adalora_tinit: int = 200
    adalora_tfinal: int = 1000
    adalora_delta_t: int = 10

    # prefix/prompt tuning/ptuning hyperparams
    num_virtual_tokens: int = 32
    mapping_hidden_dim: int = 1024

    # training hyperparams
    epochs: int = 3
    epochsCross: int = 3    #used for second train
    max_steps: int = -1
    learning_rate: float = 3e-4
    cutoff_len: int = 256
    val_set_size: float = 0.15
    num_bootstrap_samples: int = 10
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    logging_steps: int = 10
    load_8bit: bool = False
    add_eos_token: bool = False
    resume_from_checkpoint: str = None  # either training checkpoint or final adapter
    per_gpu_train_batch_size: int = 1
    gradient_accumulation_steps: int = 5
    patient: int = 5

    # auto set, user cannot control
    device: str = None
    use_mps_device: bool = False
    is_fp16: bool = True
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # generate
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_new_tokens: int = 11512
    
    
    #evaluation
    EvalFolder: str = "default"
    InferenceFolder: str = "default"
    LoadWeights: str = "True" 
    
    #Token huggingface
    TokenHugging: str = "default"
    
    #performances
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []


    """_summary_
    Lora and Qlora adapter
    settings for Qlora (quantization) are setted after in ...
    """
    def load_adapter_config(self, model):
        #t_type = TaskType.CAUSAL_LM per seq2seq
        
        #per classify
        t_type = TaskType.SEQ_CLS

        
        #for Lora and Qlora config
        """
        LoraConfig allows you to control how LoRA is applied to the base model through the following parameters:

        r:  the rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters.
        target_modules:     The modules (for example, attention blocks) to apply the LoRA update matrices.
        alpha:  LoRA scaling factor.
        bias:   Specifies if the bias parameters should be trained. Can be 'none', 'all' or 'lora_only'.
        modules_to_save:    List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. These typically include model’s custom head that is randomly initialized for the fine-tuning task.
        layers_to_transform:     List of layers to be transformed by LoRA. If not specified, all layers in target_modules are transformed.
        layers_pattern:     Pattern to match layer names in target_modules, if layers_to_transform is specified. By default PeftModel will look at common layer pattern (layers, h, blocks, etc.), use it for exotic and custom models.
        rank_pattern:   The mapping from layer names or regexp expression to ranks which are different from the default rank specified by r.
        alpha_pattern:  The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by lora_alpha.
        """
        
        if self.adapter == "lora" or self.adapter == "qlora":
            config = LoraConfig(
                task_type=t_type,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias="none",
                inference_mode=False,
            )
            if self.adapter =="qlora":
                self.is_fp16 = False
        elif self.adapter == 'adalora':
            config = AdaLoraConfig(
                task_type=t_type,
                init_r=self.adalora_init_r,
                r=self.lora_r,
                beta1=0.85,
                beta2=0.85,
                tinit=self.adalora_tinit,
                tfinal=self.adalora_tfinal,
                deltaT=self.adalora_delta_t,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                inference_mode=False,
            )
        else:
            raise KeyError("Unknow adapter: {}".format(self.adapter))
        
        model = get_peft_model(model, config)
        print("number of trainable param ")
        model.print_trainable_parameters() #PEFT function 

        return model





    """_summary_
    find automatically the device -> Cuda for intel GPU and MPS for Mac_M1
    """
    def auto_device(self):
        
        if not self.device:
            try:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except:
                self.device = "cpu"

        if self.device == "mps":
            self.use_mps_device = True
            self.is_fp16 = False
            self.device_map = {"": self.device}
        else:
            if self.load_8bit:
                self.is_fp16 = False
            self.device_map = "auto"

    
    """_summary_
    Load train data only .json and .jsonl
    """
    def load_train_data(self, type=None):
        data = None
        if type == "data2":
            path = self.data_path2
        else:
            path = self.data_path
            
        if path.endswith(".json") or path.endswith(".jsonl"):
                data = load_dataset("json", data_files=path)
        else:
            raise TypeError ("Data must be .json or .jsonl")

        return data
    
    
    
    """_summary_
    return index to create cross-validation k-fold = 10
    """
    def get_train_val_indices(self, data):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        train_indices = []
        val_indices = []
        for train_index, val_index in kf.split(data):
            train_indices.append(train_index)
            val_indices.append(val_index)
        return train_indices, val_indices
    

    """_summary_
    print time in a dedicate txt to comput total tile for training
    """
    def write_time(self, operation):
        
        #only a check if folder exists
        eval_folder = "./eval/" + self.model_type
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)

        current_dateTime = datetime.now()
        file = open(f"./eval/{self.model_type}/{self.EvalFolder}.txt", "a")  #se non c'è la cartella la crea
        
        if operation == "start":
            file.write("\n START train time -> " + str(current_dateTime))
        elif operation == "start-CrossValidation":
            file.write("\n START train time -> " + str(current_dateTime))
        elif operation == "restart":
            file.write("\n RESTART train time -> " + str(current_dateTime))
        elif operation == "inferenceSTART":
            file.write("\n\n START INFERENCE time -> " + str(current_dateTime))
        elif operation == "inferenceSTOP":
            file.write("\n STOP INFERENCE time -> " + str(current_dateTime)+"\n\n")
        elif operation == "start-Normal":
            file.write("\n no_cross START train time -> " + str(current_dateTime))
        elif operation == "start-normal+Cross":
            file.write("\n normal+Cross START train time -> " + str(current_dateTime))
        else:
            file.write("\n STOP train time -> " + str(current_dateTime))
            file.write("\n .")
            file.write("\n .")
        file.close()
        
        print("Time :"+ str(current_dateTime))




    """_summary_
    Inference class, load evaluation data.
    """
    def get_eval_input(self, s_data):
        result = []
        
        # apre il json
        if s_data:
            with open(s_data, "r") as f:
                test_items = json.loads(f.read())
            result = test_items                     #result contiene tutto il json

        print("Find {} cases".format(len(result)))

        return result
    
    
    
    """_summary_
    inference class, create output final, csv and txt
    results of fine tuned model
    """
    def eval_output(self, eval_inputs, s_data):
        output_text = ""
        output_csv = [["Expected", "Output"]]  # Intestazione del CSV

        if s_data:
            case_cnt = 0
            for item in eval_inputs:
                case_cnt += 1
                expected = item["output"]
                output = item["ac_output"]
                
                # Aggiungi riga al testo di output
                output_text += "[*] Case: {}\n--------\nExpect: \n{}\n--\nOutput: \n{}\n".format(case_cnt, expected, output)
                
                # Aggiungi riga al CSV
                output_csv.append([expected, output])
        else:
            expected = eval_inputs[0]["ac_output"]
            # Aggiungi riga al testo di output
            output_text += "LLM says: \n{}".format(expected)
            
            # Aggiungi riga al CSV
            output_csv.append([expected, ""])

        #print(output_text)

        # Scrivi l'output su un file
        #with open(f'./eval/{self.InferenceFolder}.txt', 'w') as file:
        #    file.write(output_text)
        
        # Scrivi l'output su file CSV
        with open(f'./eval/{self.model_type}/{self.InferenceFolder}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(output_csv)

        print("Inference finish")
