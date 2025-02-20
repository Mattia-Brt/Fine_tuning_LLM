import torch
import transformers

from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import (
    prepare_model_for_int8_training,
    PeftModel
)

from core.llm import LLM





class GEMMAclass_2B(LLM):
    tokenizer = None
    

    def get_model_tokenizer(self):
        bnb_config = None
        
        model_name = "google/gemma-2b"
        """if (self.TokenHugging == "default"):
            login(token=os.getenv("HUGGINGFACE_TOKEN"))
        else :
            login(token=self.TokenHugging)"""
        
        if self.adapter == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            load_in_8bit = self.load_8bit,
            device_map = self.device_map,
            quantization_config = bnb_config,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_eos_token=True
            
        )
        
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
            optim="adamw_torch",
            logging_steps=self.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=saving_step,
            save_steps=saving_step,
            max_steps=self.max_steps,
            output_dir=self.output_dir,
            metric_for_best_model = 'eval_loss',
            save_total_limit=2,
            load_best_model_at_end=False,
            group_by_length=self.group_by_length,
            use_mps_device=self.use_mps_device
        )
    
    
    def tokenize_prompt(self, data_point):
        tokenize_res = self.tokenizer(data_point["input"], truncation=True, padding=False)
        tokenize_res["labels"] = torch.tensor(self.labels.index(data_point["output"]))

        return tokenize_res
    
    
    """    
    def split_train_data(self, data):
        if self.val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle().map(self.tokenize_prompt).remove_columns(["input", "instruction", "output"])
            )
            val_data = (
                train_val["test"].shuffle().map(self.tokenize_prompt).remove_columns(["input", "instruction", "output"])
            )
        else:
            train_data = data["train"].shuffle().map(self.tokenize_prompt).remove_columns(["input", "instruction", "output"])
            val_data = None

        return train_data, val_data
    """
    
    
    
    def finetune(self):

        self.auto_device()

        if not self.lora_target_modules:
            self.lora_target_modules = [
                'k_proj',
                'q_proj',
                'gate_proj',
                'o_proj',
                'v_proj',
                'down_proj',
                'up_proj'
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
            
            
        print("--------- Model finish ---------")
        










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
        
        
        
        
    """_summary_
    Deprecate
    Utilizzata per eseguire l'inferenza su più fold ed ottenere il risultato tramite il voto di maggioranza
    """    
    def generate_crossValidation(self, data):
        self.auto_device()
        
        model, self.tokenizer = self.get_model_tokenizer()
        
        if not self.load_8bit and self.device != "cpu":
            model.half()

        #print time inference
        operation = "inferenceSTART"
        self.write_time(operation)

        eval_inputs = self.get_eval_input(data)  # eval_inputs contiene tutto il json

        # Dizionario per raccogliere tutte le risposte dai modelli
        all_responses = {i: [] for i in range(len(eval_inputs))}

        for fold in range(0, 10):
            fold_model_path = f"{self.adapter_weights}/fold_{fold}"
            fold_model = PeftModel.from_pretrained(model, fold_model_path)
            fold_model.to(self.device).eval()

            for idx, item in enumerate(eval_inputs):
                try:
                    response = self.evaluate(fold_model, item["instruction"], item["input"])
                except Exception as e:
                    if self.debug:
                        print("[DEBUG] Error: " + str(e))
                    response = "Eval Error"
                    
                all_responses[idx].append(response)
            
            del fold_model  # Libera memoria GPU
            print(f'--------- Finish inference fold {fold} ---------')

        # Aggregazione delle risposte utilizzando il voto di maggioranza
        results = []
        for idx, item in enumerate(eval_inputs):
            responses = all_responses[idx]
            final_response = max(set(responses), key=responses.count)
            item["ac_output"] = final_response
            results.append(item)

        self.eval_output(results, data)
        print(f'--------- Finish comparing voto di maggioranza ---------')
        
        #print time inference
        operation = "inferenceSTOP"
        self.write_time(operation)


    
        


    def run_manually(self, weights):
        self.auto_device()
        

        model, self.tokenizer = self.get_model_tokenizer()

        self.adapter_weights = weights
        
        if self.adapter_weights != "None":
            model = PeftModel.from_pretrained(
                model,
                self.adapter_weights,       #adapterWeight è la cartella contenente i pesi del fine tuning
            )
        else:
            raise FileExistsError("Pesi non trovati")

        return model, self.tokenizer




if __name__ == "__main__":
    llama = GEMMAclass_2B()
    llama.finetune()
