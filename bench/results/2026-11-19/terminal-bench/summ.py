import json,glob,sys,datetime
for d in sorted(glob.glob(sys.argv[1]+"/*/result.json")):
    r=json.load(open(d)); n=d.split('/')[-2].split('__')[0]
    v=(r.get('verifier_result') or {}).get('rewards',{}).get('reward')
    ex=(r.get('exception_info') or {}).get('exception_type')
    s,f=r.get('started_at'),r.get('finished_at')
    dur=(datetime.datetime.fromisoformat(f.rstrip('Z'))-datetime.datetime.fromisoformat(s.rstrip('Z'))).total_seconds()/60 if s and f else None
    ar=r.get('agent_result') or {}
    print(f"{n:34s} reward={v} exc={ex} min={dur:.1f} out_tok={ar.get('n_output_tokens')} ep={(ar.get('metadata') or {}).get('n_episodes')}")
