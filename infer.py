from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import pandas as pd
import matplotlib.pyplot as plt
from modeling.modeling_moe import MoEForCausalLM
from modeling.configuration_moe import MoEConfig



def generate(tokenizer, model, text):
    inputs = [text]
    tokens = tokenizer(inputs,return_tensors="pt")
    input_ids = tokens.input_ids
    generate_ids = model.generate(inputs=input_ids,
                num_beams=1, 
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=256,top_p=0.9, temperature=1.0, do_sample=True)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = [outputs[i][len(inputs[i]):] for i in range(len(outputs))][0]
    return response    
    
    

if __name__ == "__main__":
    model_path = "AnLan577/Dynamic_MoE" # path to the model (huggingface model hub)
    #model_path = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"
    #model_path = "ModelCloud/tinyllama-15M-stories"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token

    model_config = MoEConfig.from_pretrained(model_path)
    model = MoEForCausalLM.from_pretrained(
        model_path,
        from_tf=False,
        config=model_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    # model_config = AutoConfig.from_pretrained(model_path,trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,
    #     from_tf=False,
    #     config=model_config,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True
    # )
    model.eval() 

    response = generate(tokenizer, model, 'The highest mountain in the world is')
    print(response)

    # load excel file 
    filename = "activated_expert_num.xlsx"

    table = pd.read_excel(filename)

    # draw graph (x: layer, y: activated expert number)
    
    plt.plot(table['activated_expert_num'])
    plt.xlabel('layer')
    plt.ylabel('activated_expert_num')
    plt.title('Activated Expert Number')
    plt.show()
    
