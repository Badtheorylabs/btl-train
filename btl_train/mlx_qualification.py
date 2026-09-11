"""Bounded local SFT qualification; explicitly MLX, not an Unsloth CUDA test."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path

STEPS=8
LR=1e-5
SEED=731
LIMIT=8*1024**3


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def setup(model_path):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    mx.set_memory_limit(LIMIT);mx.set_cache_limit(128*1024**2);mx.random.seed(SEED)
    model,tok=load(str(model_path));model.freeze()
    linear_to_lora_layers(model,2,{'rank':4,'dropout':0.0,'scale':8.0})
    mx.eval(model.parameters())
    return model,tok


def guard(start):
    import mlx.core as mx
    if time.monotonic()-start>300:raise TimeoutError('300-second phase budget')
    if mx.get_peak_memory()>LIMIT:raise MemoryError('8 GiB MLX peak exceeded')


def tasks(split,n):
    vocabulary=(['amber','cedar','coral','delta','elm','fern','gold','hazel'] if split=='train'
                else ['indigo','jade','kelp','lilac','mint','navy','olive','pearl'])
    rows=[]
    for i in range(n):
        fields={'owner':vocabulary[i%8], 'code':vocabulary[(i+3)%8], 'state':vocabulary[(i+5)%8]}
        target=['owner','code','state'][i%3]
        text='Read the record. Return only the value of '+target+'. No explanation.\n'
        text+='; '.join(f'{key}={value}' for key,value in fields.items())
        rows.append({'id':f'{split}-{i}', 'source_id':f'{split}-record-{i}', 'prompt':text,'answer':fields[target]})
    return rows


def prepare(args):
    from .finetune_data import check_row, digest
    _,tok=setup(args.model)
    output={'scope':'Synthetic field extraction qualification; no public benchmark claim',
            'model_revision':args.model.name,'steps':STEPS,'learning_rate':LR,'seed':SEED,
            'lora':{'layers':2,'rank':4,'scale':8,'dropout':0},'max_sequence_tokens':128,
            'gates':{'loss_formula_abs_error':1e-5,'resume_relative_l2':1e-4,
                     'repeat_relative_l2':1e-4,'loss_max_increase':0.01,'behavioral_regression_allowed':0}}
    for split,n in [('train',24),('evaluation',12)]:
        rows=[]
        for raw in tasks(split,n):
            prompt=tok.apply_chat_template([{'role':'user','content':raw['prompt']}],tokenize=True,
                                          add_generation_prompt=True,enable_thinking=False)
            if hasattr(prompt,'tolist'):prompt=prompt.tolist()
            completion=tok.encode(raw['answer'],add_special_tokens=False)+[tok.eos_token_id]
            row=raw|{'prompt_ids':prompt,'input_ids':prompt+completion,'labels':[-100]*len(prompt)+completion}
            check_row(row,128);rows.append(row)
        output[split]=rows
    write(args.root/'protocol.json',output)
    print(json.dumps({'phase':'prepared','train':24,'evaluation':12,'revision':args.model.name}),flush=True)


def adapter_hash(model):
    import numpy as np
    from mlx.utils import tree_flatten
    h=hashlib.sha256()
    for name,tensor in sorted(tree_flatten(model.trainable_parameters())):
        h.update(name.encode());h.update(np.asarray(tensor).tobytes())
    return h.hexdigest()


def loss_fn(model,batch):
    import mlx.core as mx
    import mlx.nn as nn
    ids=mx.array(batch['input_ids']);labels=mx.array(batch['labels'])[:,1:]
    mask=labels!=-100;targets=mx.where(mask,labels,0)
    logits=model(ids[:,:-1]).astype(mx.float32)
    losses=nn.losses.cross_entropy(logits,targets,reduction='none')
    return (losses*mask).sum()/mask.sum()


def eval_loss(model,rows,pad):
    from .finetune_data import pad_batch
    model.train()  # differentiable GDN path also used for the numerical reference
    numerator=denominator=0
    for row in rows:
        count=sum(x!=-100 for x in row['labels'][1:])
        numerator+=float(loss_fn(model,pad_batch([row],pad)))*count;denominator+=count
    return numerator/denominator


def behavioral(model,tok,rows):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    model.eval();results=[]
    for row in rows:
        text=generate(model,tok,prompt=row['prompt_ids'],max_tokens=8,sampler=make_sampler(temp=0.0),verbose=False)
        results.append({'id':row['id'],'expected':row['answer'],'output':text,'passed':text.strip()==row['answer']})
    return {'passed':sum(r['passed'] for r in results),'total':len(results),'rows':results}


def checkpoint(model,opt,output,step,contract):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.eval(model.parameters(),opt.state)
    mx.save_safetensors(str(output/'adapter.safetensors'),dict(tree_flatten(model.trainable_parameters())))
    arrays={}
    def encode(v):
        if isinstance(v,mx.array):
            name='array_'+str(len(arrays));arrays[name]=v;return {'array':name}
        if isinstance(v,dict):return {'dict':{k:encode(x) for k,x in v.items()}}
        if isinstance(v,(list,tuple)):return {'list':[encode(x) for x in v]}
        return {'scalar':v}
    structure=encode(opt.state)
    mx.save_safetensors(str(output/'optimizer.safetensors'),arrays)
    write(output/'checkpoint.json',{'step':step,'contract':contract,'optimizer':structure})


def restore(model,opt,path,contract):
    import mlx.core as mx
    meta=json.loads((path/'checkpoint.json').read_text())
    if meta['contract']!=contract:raise ValueError('Checkpoint protocol differs')
    model.load_weights(str(path/'adapter.safetensors'),strict=False)
    arrays=mx.load(str(path/'optimizer.safetensors'))
    def decode(v):
        if 'array' in v:return arrays[v['array']]
        if 'dict' in v:return {k:decode(x) for k,x in v['dict'].items()}
        if 'list' in v:return [decode(x) for x in v['list']]
        return v['scalar']
    opt.state=decode(meta['optimizer']);mx.eval(model.parameters(),opt.state)
    return meta['step']


def train(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten,tree_map
    from .finetune_data import pad_batch,digest
    start=time.monotonic();protocol=json.loads((args.root/'protocol.json').read_text());contract=digest(args.root/'protocol.json')
    model,tok=setup(args.model);opt=optim.Adam(learning_rate=LR,bias_correction=True)
    params=dict(tree_flatten(model.trainable_parameters()))
    if not params or any('lora' not in name for name in params):raise ValueError('Unexpected trainable parameters')
    first=restore(model,opt,args.resume,contract) if args.resume else 0
    initial=adapter_hash(model);pad=tok.pad_token_id or tok.eos_token_id
    baseline_loss=eval_loss(model,protocol['evaluation'],pad)
    baseline_behavior=behavioral(model,tok,protocol['evaluation'])
    rows=protocol['train'];model.train();grad_fn=nn.value_and_grad(model,loss_fn)
    batch=pad_batch(rows[:2],pad)
    # Independently compute masked negative log likelihood from the same logits.
    ids=mx.array(batch['input_ids']);labels=mx.array(batch['labels'])[:,1:];mask=labels!=-100
    logits=model(ids[:,:-1]).astype(mx.float32);logp=logits-mx.logsumexp(logits,axis=-1,keepdims=True)
    direct=-(mx.take_along_axis(logp,mx.where(mask,labels,0)[...,None],axis=-1).squeeze(-1)*mask).sum()/mask.sum()
    numeric_error=abs(float(direct)-float(loss_fn(model,batch)))
    if numeric_error>1e-5:raise ValueError('Masked loss formula mismatch')
    curve=[]
    for step in range(first,args.stop_step):
        guard(start);model.train();tick=time.monotonic()
        sample=[rows[(step*2+j)%len(rows)] for j in range(2)];batch=pad_batch(sample,pad)
        loss,grads=grad_fn(model,batch);mx.eval(loss,grads)
        norm=math.sqrt(sum(float(mx.sum(g.astype(mx.float32)**2)) for _,g in tree_flatten(grads)))
        if not math.isfinite(float(loss)) or not math.isfinite(norm) or norm==0:raise ValueError('Bad loss/gradient')
        scale=min(1.0,1.0/norm);grads=tree_map(lambda g:g*scale,grads)
        opt.update(model,grads);mx.eval(model.parameters(),opt.state)
        curve.append({'step':step+1,'seconds':time.monotonic()-tick,'loss':float(loss),'gradient_norm':norm,
                      'tokens':sum(len(r['input_ids']) for r in sample),'supervised_tokens':sum(sum(t!=-100 for t in r['labels'][1:]) for r in sample)})
        print(json.dumps(curve[-1]),flush=True);write(args.out/'curve.json',curve)
    checkpoint(model,opt,args.out,args.stop_step,contract)
    final=adapter_hash(model)
    if final==initial:raise ValueError('Adapter did not change')
    final_loss=eval_loss(model,protocol['evaluation'],pad)
    final_behavior=behavioral(model,tok,protocol['evaluation']);guard(start)
    result={'backend':'mlx','unsloth_validated':False,'initial_step':first,'final_step':args.stop_step,
            'initial_adapter_hash':initial,'final_adapter_hash':final,'loss_formula_error':numeric_error,
            'baseline_loss':baseline_loss,'final_loss':final_loss,'baseline_behavior':baseline_behavior,
            'final_behavior':final_behavior,'steps':curve,'peak_memory_bytes':mx.get_peak_memory(),
            'elapsed_seconds':time.monotonic()-start,'trainable_parameters':sum(p.size for p in params.values())}
    write(args.out/'result.json',result);print(json.dumps({k:result[k] for k in ['final_step','baseline_loss','final_loss','peak_memory_bytes','elapsed_seconds']}),flush=True)


def reload(args):
    import mlx.core as mx
    import mlx.optimizers as optim
    from .finetune_data import digest
    model,tok=setup(args.model);p=json.loads((args.root/'protocol.json').read_text())
    step=restore(model,optim.Adam(learning_rate=LR,bias_correction=True),args.resume,digest(args.root/'protocol.json'))
    result={'step':step,'adapter_hash':adapter_hash(model),'loss':eval_loss(model,p['evaluation'],tok.pad_token_id or tok.eos_token_id),
            'behavior':behavioral(model,tok,p['evaluation']),'fresh_process':True}
    write(args.out/'reload.json',result);print(json.dumps({'phase':'fresh-reload','step':step,'passed':result['behavior']['passed']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','reload']);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--stop-step',type=int,default=STEPS)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    if a.out:a.out.mkdir(parents=True,exist_ok=False)
    if a.phase=='prepare':prepare(a)
    elif a.phase=='train':train(a)
    else:reload(a)

if __name__=='__main__':main()
