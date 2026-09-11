"""Fresh-process adapter reload and fixed held-out behavior check."""
import sys,json,hashlib
from pathlib import Path
from unsloth import FastModel
import torch
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from safetensors.torch import load_file
request=json.loads(Path(sys.argv[1]).read_text());cfg=request['config'];root=Path(request['workspace'])
model,processor=FastModel.from_pretrained(model_name=str(root/cfg['model_directory']),max_seq_length=cfg['max_seq_length'],dtype=torch.bfloat16,load_in_4bit=False,full_finetuning=False,local_files_only=True,trust_remote_code=False,device_map={'':0})
tok=getattr(processor,'tokenizer',processor)
model=FastModel.get_peft_model(model,r=cfg['rank'],lora_alpha=cfg['alpha'],lora_dropout=0,bias='none',target_modules=cfg['target_modules'],use_gradient_checkpointing='unsloth',random_state=cfg['seed'])
rows=[json.loads(line) for line in (root/'data/evaluation.jsonl').read_text().splitlines() if line.strip()]
def evaluate():
 FastModel.for_inference(model);model.eval();results=[]
 for row in rows:
  ids=torch.tensor([row['prompt_ids']],device='cuda');mask=torch.ones_like(ids)
  with torch.inference_mode():out=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=8,do_sample=False,pad_token_id=tok.eos_token_id)
  text=tok.decode(out[0,len(row['prompt_ids']):],skip_special_tokens=True).strip()
  results.append({'id':row['id'],'expected':row['answer'],'output':text,'passed':text==row['answer']})
 return {'passed':sum(r['passed'] for r in results),'total':len(results),'rows':results}
before=evaluate();set_peft_model_state_dict(model,load_file(str(Path(sys.argv[2])/'adapter_model.safetensors')))
h=hashlib.sha256()
for name,tensor in sorted(get_peft_model_state_dict(model).items()):h.update(name.encode());h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
after=evaluate();result={'base':before,'adapted':after,'adapter_sha256':h.hexdigest(),'fresh_process':True,'gpu':torch.cuda.get_device_name(0)}
Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
