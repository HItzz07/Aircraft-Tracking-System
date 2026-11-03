#!/usr/bin/env python3
"""Replay requests; assert HTTP statuses; retain bodies for schema-independent review."""
import argparse, json, sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://localhost:8000');p.add_argument('--races',action='store_true');p.add_argument('--start',type=int,default=1);p.add_argument('--stop',type=int);p.add_argument('--output',default='responses.jsonl');a=p.parse_args()
here=Path(__file__).resolve().parent

def request(s):
    data=s.get('raw_body')
    if data is None and 'body' in s:data=json.dumps(s['body'],ensure_ascii=False)
    req=Request(a.base_url.rstrip('/')+s['path'],data=None if data is None else data.encode(),method=s['method'],headers={'Content-Type':'application/json'})
    try:
        with urlopen(req,timeout=30) as r:status,body=r.status,r.read().decode()
    except HTTPError as r:status,body=r.code,r.read().decode()
    try:body=json.loads(body)
    except ValueError:pass
    return {'step_id':s.get('step_id'), 'status':status,'response':body}

failed=0
with open(a.output,'w') as out:
    if a.races:
        for case in json.loads((here/'concurrent-cases.json').read_text()):
            barrier=Barrier(len(case['requests']))
            def run(s):barrier.wait(timeout=10);return request(s)
            with ThreadPoolExecutor(max_workers=len(case['requests'])) as pool:results=list(pool.map(run,case['requests']))
            ok=sorted(r['status'] for r in results)==case['expected_status_multiset']
            failed+=not ok;out.write(json.dumps({'case':case['name'],'results':results,'status_check_passed':ok})+'\n')
            print('PASS' if ok else 'FAIL',case['name'])
    else:
        expected={x['step_id']:x['expected_status'] for x in json.loads((here/'expected-statuses.json').read_text())}
        for n,line in enumerate((here/'arrival-stream.jsonl').read_text().splitlines(),1):
            if n<a.start or a.stop is not None and n>a.stop:continue
            s=json.loads(line);r=request(s);ok=r['status']==expected[s['step_id']];failed+=not ok
            r.update(expected_status=expected[s['step_id']],status_check_passed=ok);out.write(json.dumps(r,ensure_ascii=False)+'\n')
            if not ok:print('FAIL',s['step_id'],s['label'],r['status'],'expected',expected[s['step_id']])
print(f'{failed} status failures. Inspect response content separately; statuses alone do not prove correctness.')
sys.exit(1 if failed else 0)
