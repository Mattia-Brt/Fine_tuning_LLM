import os
import torch
import transformers

from transformers import (
    BitsAndBytesConfig,
    LlamaForSequenceClassification,
    AutoTokenizer
)

from huggingface_hub import login

from peft import (
    prepare_model_for_int8_training,
    PeftModel
)

from core.llm import LLM





class LLAMA3class_8B(LLM):
    tokenizer = None
    

    def get_model_tokenizer(self):
        bnb_config = None
        
        model_name = "meta-llama/Meta-Llama-3-8B"
        if (self.TokenHugging == "default"):
            login(token=os.getenv("HUGGINGFACE_TOKEN"))
        else :
            login(token=self.TokenHugging)
     
        
        if self.adapter == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        
        
        model = LlamaForSequenceClassification.from_pretrained(
            model_name,
            load_in_8bit=self.load_8bit,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token

        model.config.use_cache = False
        model.config.pad_token_id = model.config.eos_token_id

        return model, tokenizer


    def get_train_args(self, trainRow) -> transformers.TrainingArguments:
        
        total_batch_size = self.per_gpu_train_batch_size * self.gradient_accumulation_steps * (self.world_size if self.ddp else 1)
        total_optim_steps = trainRow // total_batch_size
        saving_step = int(total_optim_steps / 10)
        warmup_steps = int(total_optim_steps / 10)
        
        return transformers.TrainingArguments(
            per_device_train_batch_size=self.per_gpu_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            fp16=self.is_fp16,
            #se lora togli il commento (va aggiunto un if perché qlora non accetta che pf16 true ovviamente)
            optim="adamw_torch",
            logging_steps=self.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=saving_step,
            save_steps=saving_step,
            #max_steps=-1,
            output_dir=self.output_dir,
            metric_for_best_model = 'eval_f1',
            save_total_limit=11,
            load_best_model_at_end=True,
            group_by_length=self.group_by_length,
            use_mps_device=self.use_mps_device
        )
    
    
    def tokenize_prompt(self, data_point):
        tokenize_res = self.tokenizer(data_point["input"], truncation=True, padding=False)
        tokenize_res["labels"] = torch.tensor(self.labels.index(data_point["output"]))

        return tokenize_res



    def finetune(self):
        
        self.auto_device()

        if not self.lora_target_modules:
            self.lora_target_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj"
            ]

        #download del modello base 
        model, self.tokenizer = self.get_model_tokenizer()
         
        
        # preparo ora il modello per il training con Lora config
        
        #quindi non quantizzato con qlora che vuole 4bit
        if self.load_8bit:
            model = prepare_model_for_int8_training(model)      #"funziona"

        #lora config nello script llm
        model = self.load_adapter_config(model) #stampa anche il numero di parametri        -       PEFT 
        
        #load data
        dataset = self.load_train_data()
        print(dataset)

        #raise ValueError("prova finisce qui")
        if not dataset:
            print("Warning! Empty Train Data!")
            return
        
        train_indices, val_indices = self.get_train_val_indices(dataset['train'])

        
        #print start time
        operation = "start"
        self.write_time(operation)
        
        # Iterate over kfold samples
        for fold, (train_index, val_index) in enumerate(zip(train_indices, val_indices)):
            print(f"Training fold {fold}")
            #print(f"Train indices: {train_index}")
            #print(f"Validation indices: {val_index}")

            train_data = dataset['train'].select(train_index)#funzionano, contengono i dati sottoforma di array
            val_data = dataset['train'].select(val_index)

            print(f"Train dataset size after selection: {len(train_data)}")
            print(f"Validation dataset size after selection: {len(val_data)}")
            
            
            train_dataset = (
                train_data.shuffle().map(self.tokenize_prompt).remove_columns(["input", "instruction", "output"])
            )
            val_dataset = (
                val_data.shuffle().map(self.tokenize_prompt).remove_columns(["input", "instruction", "output"])
            )
            


            
            # Trainer initialization
            train_args = self.get_train_args(len(train_data))
            trainer = transformers.Trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                args=train_args,
                data_collator=transformers.DataCollatorWithPadding(self.tokenizer, return_tensors="pt")
            )

            model.config.use_cache = False

            # Train model
            trainer.train()
            
            
        model.save_pretrained(self.output_dir)
        print("--------- Final Model saved ---------")
        
        # Stop time function 
        operation = "stop"
        self.write_time(operation)

        print("--------- Model saved ---------")
        








    """Inference classes
    #
    #
    #
    #
    #
    #
    """
    def evaluate(self, model, instruction=None, input=None, **kwargs):
        inputs = self.tokenizer(instruction, input, return_tensors="pt")
        
        with torch.no_grad():
            
            inputs_cuda = {key: tensor.to(self.device) for key, tensor in inputs.items()}
            logits = model(**inputs_cuda).logits
            
            predicted_class_idx = torch.argmax(logits, dim=1).item()

            return self.labels[predicted_class_idx]
        


    
    def generate(self, data):
        self.auto_device()

        model, self.tokenizer = self.get_model_tokenizer()
        
        if self.LoadWeights == "True":
            if self.adapter_weights != "None":
                model = PeftModel.from_pretrained(
                    model,
                    self.adapter_weights,       #adapterWeight è la cartella contenente i pesi del fine tuning
                )
            else:
                raise FileExistsError("Pesi non trovati")
        

        if not self.load_8bit and self.device != "cpu":
            model.half()

        #print time inference
        operation = "inferenceSTART"
        self.write_time(operation)
        
        model.to(self.device).eval()

        eval_inputs = self.get_eval_input(data)     #eval_inputs contiene tutto il json
        
        for item in eval_inputs:
            try:
                response = self.evaluate(model, item["instruction"], item["input"])
            except Exception as e:
                if self.debug:
                    print("[DEBUG] Error: " + str(e))
                response = "Eval Error"

            item["ac_output"] = response

        self.eval_output(eval_inputs, data)
        
        #print time inference
        operation = "inferenceSTOP"
        self.write_time(operation)
        

if __name__ == "__main__":
    llama = LLAMA3class_8B()
    llama.finetune()
