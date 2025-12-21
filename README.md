# LLaDA-V-Qwen: Using the tiny model of Qwen as the diffusion model to generate the radiology report.
Language model pretrain weight:
[![deploy](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1)]
Vision encoder pretrain weight:
[![deploy](https://huggingface.co/google/siglip2-so400m-patch14-384)]

  
## Envirment preparation
 Need install two set up  package. dllm and LLaDA-V.
   ```bash
   git clone https://github.com/ML-GSAI/LLaDA-V
   cd LLaDA-V/train
   bash init_env.sh
   git clone https://github.com/ZHZisZZ/dllm.git
   cd dllm
   pip install -e .
   ```

<!-- ### Quick Inference Demo
The [LLaDA-V model](https://huggingface.co/GSAI-ML/LLaDA-V) is now available on Hugging Face Hub. To quickly test the model with a visual instruction demo, follow these simple steps:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/ML-GSAI/LLaDA-V
   cd LLaDA-V/train
   ```
2. **Initialize the environment**  
   Run the environment setup script to install necessary dependencies:
   ```bash
   bash init_env.sh
   ```
3. **Run the demo script**  
   Execute the demo script to test LLaDA-V on an example image:
   ```bash
   python generate_demo.py
   ``` -->

## Training from LLaVA-Qwen
To success train the LLaVA-Qwen model, few changes made to the code to deal with the transformer vision does not match problem:

1. Change the file llada_trainer.py package import to match the transformer version.

2. Change the modeling_llada.py to modeling_Qwem.py, from the transformer copy the Qwen3 structure and then add the mask logic and change the corresponding generate logic to the Qwen3ForCausalLM forward.

2.1 [Mask] Token replace and reture t and [Mask] token corresponding embedding function ---> function forward_process_embeds

2.2 [Mask] token index find ---> function get_masked_indices_from_embeds

2.3 Single modality inference ---> function generate

2.4 multiple modalities inference ---> function generate_with_embeds

2.5 Calculate each step need reover how many tokens ---> get_num_transfer_tokens

3. Change the llava_llada.py to llava_qwen.py with the vision encoder and mapper not change, change the language model to the Qwen3ForCausalLM.

4. From the transformer get the Qwen3 configuration and change to the configuration_Qwen.py with the same format as the configuration_LLaDA.py.

5. Change the converstion.py and the Qwen3 data format, replace the system prompt with the prompt: SYSTEM_PROMPT = "You are an assistant in radiology, responsible for analyzing medical imaging studies and generating detailed, structured, and accurate radiology reports."

6. Change the Qwen3 version in the train.py to suit the llava_qwen.py and the data preprocess of Qwen.

### Data Preparation
As an example, we outlined the data preparation process for training LLaDA-V using the LLaVA-NeXT dataset. You need to prepare the following datasets:

1. Mainly use the file `annotation_with_clinic_info.json` to suit the model except format.

2. Divide the train and test entities list to the train and test two jason files.

3. Create the conversation list with from human and from gpt. from human value is the prompt with the concatenation of the ASSISSTANT_PROMPT = "<image>\nProvide a detailed description of the findings in the radiology image." and CLINICAL_PROMPT = "Following clinical context:" and the key value of the indication, comparison, history and technique. The from gpt value is the value of the ground truth report. The train and test file is `train_llave_llada.json` and `test_llave_llada.json`

4. The dataset format is:
   ```
  {
    "id": "0c4eb1e1-b801903c-bcebe8a4-3da9cd3c-3b94a27c",
    "image": "p10/p10000898/s50771383/0c4eb1e1-b801903c-bcebe8a4-3da9cd3c-3b94a27c.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nProvide a detailed description of the findings in the radiology image. Following clinical context: Indication: F with chest pain pna"
      },
      {
        "from": "gpt",
        "value": "PA and lateral views of the chest provided. Lung volumes are somewhat low. There is no focal consolidation, effusion, or pneumothorax. The cardiomediastinal silhouette is normal. Imaged osseous structures are intact. No free air below the right hemidiaphragm is seen."
      }
    ]
  },
   ```


### Add Complementary Mask

Following the LaViDa Complementary Mask handle, change the forward logic of Qwen3ForCausalLM.

1. Add the ~masked-indicate to mask the 1-t tokens in the copy of the label. and make sure the tokens will not contain the prompt token

   ```python

   prompt_index = (labels == -100).to(torch.int64)

   target_mask = (1 - prompt_index).bool()

   masked_indices_env = (~masked_indices) & (target_mask)

   ```

2. Replace the following index with the [Mask] token embeddings and concate with the original noisy embeddings.

   ```python

   noisy_embeds_inv = torch.where(masked_indices_env.view(bsz,seq_len,1),masked_embed,inputs_embeds)

   noisy_embeds = torch.cat([noisy_embeds,noisy_embeds_inv])

   ```

3. Change the label shape and construct. The original label calculate using the:

   ```python

   token_loss = F.cross_entropy(logits[masked_indices], labels[masked_indices], ignore_index=-100,
                                         reduction='none') / p_mask[masked_indices]
   loss = torch.sum(token_loss / noisy_data_length[masked_indices]) / labels.shape[0]

   ```

The loss will be normlized by t and the loss is calculated in the [Mask] token position and will be divided by the [Mask] token length.

Since the embedding will be concate and one part has t percent [Mask] token and another part has 1-t percent [Mask] token, so we concate the label together, too. With the [Mask] token corresponding position set to -100, in order to ignore the loss. And the loss value will be normlized by t and 1-t, respectively. Since the number of [Mask] token of the concatenation should equals to the length of the label, so it will be normlized by the labels length. The code implementation can be seen below:

   ```python

   noisy_embeds_inv = torch.where(masked_indices_env.view(bsz,seq_len,1),masked_embed,inputs_embeds)
   labels_env = labels.clone()
   labels_env[masked_indices]= -100
   labels[masked_indices_env]= -100
   labels =  torch.cat([labels,labels_env])
   noisy_embeds = torch.cat([noisy_embeds,noisy_embeds_inv])
   p_mask_inv = 1.0 - p_mask
   p_mask = torch.cat([p_mask, p_mask_inv], dim=0)
   noisy_data_length = noisy_data_length.repeat(2, 1)
   token_loss = F.cross_entropy(logits, labels, ignore_index=-100,
                              reduction='none') / p_mask
   loss = torch.sum(token_loss / noisy_data_length) / labels.shape[0]

   ```

### Evaluation scripts change

1. Change the evaluate.sh scripts to the different gpu cards to different tasks to multiple gpu cards response for one task. Add the radiology report generation task "mimic-cxr" and set the corresponding eval settings.

2. Add the mimic-cxr task file folder in `eval/lmms-eval/lmms_eval/tasks/mimic_cxr` the file contains the evaluation metric and the path to the jason label file and the image file.

3. Follow the format of the llava_onevision_llada.py to add the llava_oneversion_qwen.py, containing the inference tool function of the diffustion language model.

4. Change the builder.py to set the version `qwen` and change model to the llava_qwen.py

### Stage 1 train the mapper in order to better align the two modalities' embeddings, with the vision encoder and the language model be freezed
```bash
Pretrain Script:
   cd train && bash scripts/llada_v_pretrain.sh
```

### Stage 2 load the mapper, vision encoder and language model pretrain weight and finetune from LLaDA-V
```bash
Script: 
   cd train && bash scripts/llada_v_finetune.sh
   note: you need to add the path of "data_path", "image_folder", "video_folder" in llada_v_finetune.sh.
```


## Evaluation
We provide the evaluation code in this repository, following the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) library.  
   Execute the demo script to test LLaDA-V on an example image:
   ```bash
   cd eval && bash scripts/evaluate.sh
   ```

## Contact
If you have any questions, please feel free to contact us at zebin@ruc.edu.cn.


## Acknowledgments
The code is largely based on the [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), [MAmmoTH-VL](https://github.com/MAmmoTH-VL/MAmmoTH-VL), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [dLLM-cache](https://github.com/maomaocun/dLLM-cache/tree/main). We thank the authors for their great work. 

We are also very grateful to Chengyue for helping us adapt [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM), which significantly accelerates the generation process.

## Discussion

Feel free to scan the WeChat QR code below to participate in the discussion and stay updated with the latest progress.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="vx.jpg" style="width: 50%" />
</div>

## Citation

```bibtex
@article{you2025llada,
  title={LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning},
  author={You, Zebin and Nie, Shen and Zhang, Xiaolu and Hu, Jun and Zhou, Jun and Lu, Zhiwu and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2505.16933},
  year={2025}
}
```


