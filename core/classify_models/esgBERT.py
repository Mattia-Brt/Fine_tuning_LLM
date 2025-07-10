import os
import torch
import transformers

from transformers import (
    BitsAndBytesConfig,
    BertForSequenceClassification,
    AutoTokenizer
)

from huggingface_hub import login

from peft import (
    prepare_model_for_int8_training,
    PeftModel
)

from core.llm import LLM





class esgBERTclass(LLM):
    tokenizer = None
    

    def get_model_tokenizer(self):
        bnb_config = None
        
        model_name = "nbroad/ESG-BERT"
     
        
        if self.adapter == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        
        
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            load_in_8bit=self.load_8bit,
            quantization_config=bnb_config,
            ignore_mismatched_sizes=True,  # evita l'errore sui pesi del classifier
            num_labels=2
            
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
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
            #se lora togli il commento (va aggiunto un if perché qlora non accetta che pf16 true ovviamente)
            optim="adamw_torch",
            logging_steps=self.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=saving_step,
            save_steps=saving_step,
            #max_steps=-1,
            output_dir=self.output_dir,
            metric_for_best_model = 'eval_loss',
            save_total_limit=2,
            load_best_model_at_end=False,
            group_by_length=self.group_by_length,
            use_mps_device=self.use_mps_device
        )
        
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
    
    
    def tokenize_prompt(self, data_point):
        tokenize_res = self.tokenizer(data_point["input"], truncation=True, padding=False)
        tokenize_res["labels"] = torch.tensor(self.labels.index(data_point["output"]))

        return tokenize_res



    def finetune(self):
        
        self.auto_device()

        if not self.lora_target_modules:
            self.lora_target_modules = [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense"
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
        dataset = self.load_train_data("data1")
        print(dataset)

        #raise ValueError("prova finisce qui")
        if not dataset:
            print("Warning! Empty Train Data!")
            return

        operation = "start-Normal"
        self.write_time(operation)
        
        model = self.start_noCrossValidation(dataset, model)
            
        model.save_pretrained(self.output_dir)

        
        print("--------- Final Model saved ---------")
        
        # Stop time function 
        operation = "stop"
        self.write_time(operation)

        print("--------- Model saved ---------")
        

    """_summary_
    funzione che implementa il finetune normale con split del train e val set senza crossVal
    """
    def start_noCrossValidation(self, dataset, model:PeftModel):
        train_data, val_data = self.split_train_data(dataset)

        train_args = self.get_train_args(len(train_data))
        model = model.to("cuda")
        
        # Trainer initialization
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=train_args,
            data_collator=transformers.DataCollatorWithPadding(self.tokenizer, return_tensors="pt")
        )

        model.config.use_cache = False

        trainer.train()
        
        
        return model  # Restituisci il modello addestrato


    

    



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
    llama = esgBERTclass()
    llama.finetune()
