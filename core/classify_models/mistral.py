import torch
import transformers

from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import (
    prepare_model_for_kbit_training,
    PeftModel
)

from core.llm import LLM


class MISTRALclass_7B(LLM):
    tokenizer = None
    
    
    

    def get_model_tokenizer(self):
        
        
        base_model_id = "mistralai/Mistral-7B-v0.1"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id,
            token=self.TokenHugging,
            quantization_config=bnb_config,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
            load_in_8bit=self.load_8bit,
            )
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            token=self.TokenHugging
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
            #optim="adamw_torch",
            optim = "paged_adamw_8bit",
            logging_steps=self.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=saving_step,
            save_steps=saving_step,
            #max_steps=-1,
            output_dir=self.output_dir,
            warmup_ratio= 0.3,
            weight_decay= 0.001,
            bf16= False,
            max_grad_norm = 0.3,
            save_total_limit=11,
            lr_scheduler_type= "constant",
            load_best_model_at_end=True,
            group_by_length=self.group_by_length,
            use_mps_device=self.use_mps_device,
            metric_for_best_model = 'eval_f1'
            

        )
        

    def tokenize_prompt(self, data_point):
        max_length = 512

        tokenize_res = self.tokenizer(data_point["input"], truncation=True, max_length= max_length, padding="max_length")
        tokenize_res["labels"] = torch.tensor(self.labels.index(data_point["output"]))

        return tokenize_res
    
    
    
    



    def finetune(self):
        
        self.auto_device()
        
        #lora param
        self.lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            
        ]
        
        #download del modello base 
        model, self.tokenizer = self.get_model_tokenizer()
         
        # preparo ora il modello per il training con Lora config

        model = prepare_model_for_kbit_training(model)
        
        model = self.load_adapter_config(model)
        
        #load data
        dataset = self.load_train_data()                           #"funziona", splitta gia il train set, il val non mi serve per ora
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
        

        #if not self.load_8bit and self.device != "cpu":
         #   model.half()

        #print time inference
        operation = "inferenceSTART"
        self.write_time(operation)
        
        #model.to(self.device).eval()
        model.eval()

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
    mistral = MISTRALclass_7B()
    mistral.finetune()
