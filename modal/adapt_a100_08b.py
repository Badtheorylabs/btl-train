"""BTL Adapt: one bounded Modal A100 qualification; no retries or public deployment."""
import json, re, time
from pathlib import Path
import modal

SESSION='btl-adapt-a100-08b-20260909'
WORK='/vol/adapt-08b'
MODEL='Qwen/Qwen3.5-0.8B'
REVISION='2fc06364715b967f1860aea9cf38778875588b17'
IMAGE='unsloth/unsloth@sha256:27ee1e4a487ae58f7e92785d3394f3f1e8f3318d496cf235b7b8e0c55faa4868'
PYTHON='/opt/unsloth-venv/bin/python'
app=modal.App(SESSION)
volume=modal.Volume.from_name(SESSION,create_if_missing=True)
image=modal.Image.from_registry(IMAGE).entrypoint([])
if modal.is_local():
    ROOT=next(p for p in Path(__file__).resolve().parents if (p/'.btl-workspace.json').is_file())
    image=(image.add_local_dir(ROOT/'platform/btl-train/btl_train','/root/adapt-source/btl_train',ignore=['__pycache__','**/*.pyc'])
           .add_local_file(ROOT/'private/workspace/lab-state/qualification-08b-20260909/protocol.json','/root/protocol.json')
           .add_local_file(Path(__file__).with_name('adapt_prepare.py'),'/root/adapt_prepare.py')
           .add_local_file(Path(__file__).with_name('adapt_reload.py'),'/root/adapt_reload.py'))

@app.function(image=image,volumes={'/vol':volume},cpu=2,memory=4096,timeout=600,startup_timeout=240,retries=0,max_containers=1,scaledown_window=2)
def prepare():
    import os,subprocess
    start=time.monotonic();Path(WORK).mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONPATH']='/root/adapt-source'
    p=subprocess.run([PYTHON,'/root/adapt_prepare.py',WORK,MODEL,REVISION],env=env,capture_output=True,text=True,timeout=540)
    receipt={'phase':'cpu-prepare','returncode':p.returncode,'seconds':time.monotonic()-start,'stdout':p.stdout[-8000:],'stderr':p.stderr[-4000:],'gpu_allocated':False}
    (Path(WORK)/'prepare-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');volume.commit();return receipt

@app.function(image=image,volumes={'/vol':volume},gpu='A100-40GB',cpu=2,memory=16384,timeout=1200,startup_timeout=240,retries=0,max_containers=1,scaledown_window=2)
def qualify(attempt: str):
    import os,subprocess,tarfile,io,hashlib,traceback
    if not re.fullmatch(r'a[0-9]{2}',attempt):raise ValueError('Invalid attempt identifier')
    volume.reload();start=time.monotonic();out=Path(WORK)/attempt;out.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(PYTHONPATH='/root/adapt-source',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='disabled',UNSLOTH_DISABLE_STATISTICS='1')
    request=json.loads((Path(WORK)/'request.json').read_text());receipts=[]
    result={'product':'BTL Adapt','backend':'Unsloth Core','attempt':attempt,'status':'failed','phases':receipts}
    def phase(name,command,seconds=350):
        log=out/(name+'.log');tick=time.monotonic()
        with log.open('w') as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=seconds)
        r={'name':name,'exit_code':p.returncode,'seconds':time.monotonic()-tick,'log':str(log)};receipts.append(r)
        print(json.dumps(r),flush=True)
        if p.returncode:raise RuntimeError(name+' failed: '+log.read_text()[-6000:])
    try:
        # Each phase is a new process; source and optimizer contracts remain fixed.
        full=out/'full';req=out/'full-request.json';req.write_text(json.dumps(request))
        phase('full',[PYTHON,'-m','btl_train.finetune_worker','--request',str(req),'--output',str(full)],500)
        checkpoint=full/'checkpoints/checkpoint-4'
        resumed=out/'resumed';resume_req=out/'resume-request.json';resume_req.write_text(json.dumps(request|{'resume':str(checkpoint)}))
        phase('resume',[PYTHON,'-m','btl_train.finetune_worker','--request',str(resume_req),'--output',str(resumed)],350)
        phase('fresh-reload-and-behavior',[PYTHON,'/root/adapt_reload.py',str(req),str(full/'adapter'),str(out/'reload.json')],180)
        a=json.loads((full/'worker-result.json').read_text());b=json.loads((resumed/'worker-result.json').read_text());reload=json.loads((out/'reload.json').read_text())
        result.update(full=a,resumed=b,reload=reload)
        result['gates']={'actual_steps':a['final_step']==8 and b['initial_step']==4 and b['final_step']==8,
                         'adapter_serialization':a['serialization_reload_passed'] and b['serialization_reload_passed'],
                         'resume_exact_match':a['updated_adapter_sha256']==b['updated_adapter_sha256'],
                         'fresh_reload_exact_match':a['updated_adapter_sha256']==reload['adapter_sha256'],
                         'behavior_retained':reload['adapted']['passed']>=reload['base']['passed'],
                         'loss_nonregression':a['final_loss']<=a['baseline_loss']+0.01}
        result['status']='passed' if all(result['gates'].values()) else 'failed-gate'
    except Exception as e:
        result['error']=str(e);result['traceback']=traceback.format_exc()
    finally:
        result['gpu_function_seconds']=time.monotonic()-start
        result['cost_estimate_usd']=result['gpu_function_seconds']*(0.000583+2*0.0000131+16*0.00000222)
        result['cost_scope']='Estimate for function execution at published resource rates; excludes image preparation/startup and is not a billing receipt'
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    # Return only this attempt's artifacts; never the downloaded base model.
    total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if total>200*1024**2:raise RuntimeError('Unexpectedly large result directory; artifacts remain on volume')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz',compresslevel=1) as tar:tar.add(out,arcname=attempt)
    return {'result':result,'archive':buffer.getvalue(),'uncompressed_bytes':total}

@app.local_entrypoint()
def main(budget_usd: float, authorization_ref: str, attempt: str='a01', skip_prepare: bool=False):
    if not 0<budget_usd<5 or not authorization_ref.strip():raise ValueError('This job requires explicit authorization below $5')
    dest=ROOT/'private/workspace/lab-state'/SESSION;dest.mkdir(parents=True,exist_ok=True)
    if not skip_prepare:
        prep=prepare.remote();(dest/(attempt+'-prepare.json')).write_text(json.dumps(prep,indent=2)+'\n');print(json.dumps(prep),flush=True)
        if prep['returncode']:return
    response=qualify.remote(attempt)
    (dest/(attempt+'.tar.gz')).write_bytes(response['archive'])
    (dest/(attempt+'-result.json')).write_text(json.dumps(response['result'],indent=2)+'\n')
    print(json.dumps({'result':response['result'],'archive_bytes':len(response['archive']),'budget_usd':budget_usd,'authorization_ref':authorization_ref}),flush=True)
