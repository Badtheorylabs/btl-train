"""CPU staging and tokenizer/data integrity checks before allocating an A100."""
import os,sys,json,hashlib,importlib.metadata
from pathlib import Path
from collections.abc import Mapping
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer,AutoConfig
from btl_train.finetune_data import PACKAGES,digest,audit
work=Path(sys.argv[1]);model_id=sys.argv[2];revision=sys.argv[3]
model=Path(snapshot_download(model_id,revision=revision,local_dir=str(work/'model'),max_workers=4,ignore_patterns=['*.md','.gitattributes']))
tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
config=AutoConfig.from_pretrained(str(model),local_files_only=True)
if config.model_type!='qwen3_5':raise ValueError('Model architecture mismatch')
p=json.loads(Path('/root/protocol.json').read_text());data=work/'data';data.mkdir(exist_ok=True)
identity={'id':model_id,'revision':revision}
manifest={'schema_version':1,'accepted':True,'tokenizer':identity}
for name,key in [('train','train'),('evaluation','evaluation')]:
 rows=p[key]
 for row in rows:
  ids=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  if isinstance(ids,Mapping):ids=ids['input_ids']
  if hasattr(ids,'tolist'):ids=ids.tolist()
  if ids and isinstance(ids[0],list):
   if len(ids)!=1:raise ValueError('Unexpected tokenizer batch')
   ids=ids[0]
  if ids!=row['prompt_ids']:raise ValueError('Tokenizer token mismatch: '+json.dumps({'type':str(type(ids)),'actual_length':len(ids),'expected_length':len(row['prompt_ids']),'actual_prefix':ids[:20],'expected_prefix':row['prompt_ids'][:20]}))
  if ids+tok.encode(row['answer'],add_special_tokens=False)+[tok.eos_token_id]!=row['input_ids']:raise ValueError('Completion tokenizer mismatch')
 path=data/(name+'.jsonl');path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n');manifest[name]={'path':path.name,'sha256':digest(path)}
(data/'acceptance.json').write_text(json.dumps({'scope':'synthetic field-extraction qualification only','tokenizer_parity_verified':True,'train_rows':24,'evaluation_rows':12})+'\n')
manifest['acceptance_receipt']={'path':'acceptance.json','sha256':digest(data/'acceptance.json')}
(data/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
files=[{'path':str(f.relative_to(model)),'sha256':digest(f)} for f in model.rglob('*') if f.is_file() and '.cache' not in f.parts]
(work/'model-manifest.json').write_text(json.dumps({'tokenizer':identity,'files':files},indent=2)+'\n')
versions={name:importlib.metadata.version(name) for name in PACKAGES}
cfg={'schema_version':1,'project_id':'workspace/btl-train','method':'lora','precision':'bf16','expected_model_type':'qwen3_5',
 'model_directory':'model','model_manifest':'model-manifest.json','model_manifest_sha256':digest(work/'model-manifest.json'),
 'dataset_manifest':'data/manifest.json','dataset_manifest_sha256':digest(data/'manifest.json'),'max_seq_length':128,'max_steps':8,
 'micro_batch_size':1,'gradient_accumulation_steps':2,'save_steps':4,'wall_seconds':500,'rank':4,'alpha':8,'seed':731,
 'learning_rate':1e-5,'max_grad_norm':1.0,'target_modules':['q_proj','v_proj'],'versions':versions}
audit(work,cfg,check_weights=True)
request={'workspace':str(work),'config':cfg,'implementation_sources':{name:digest(Path('/root/adapt-source/btl_train')/name) for name in ['finetune_worker.py','finetune_data.py','process.py']}}
(work/'request.json').write_text(json.dumps(request,indent=2)+'\n')
print(json.dumps({'status':'prepared','model_type':config.model_type,'versions':versions,'weights_bytes':sum(p.stat().st_size for p in model.glob('*.safetensors')),'tokenizer_parity':True}))
