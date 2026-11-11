"""One quick test for the Gemini 4 Carbon video: Carbon vs Argon on the same fresh, hard Sudoku.

Standard library + requests. Key: GEMINI_API_KEY env var, else D:\\Youtube video\\.env.

  python sudoku_test.py make --seed 7            # write puzzle.json (unique solution, hard)
  python sudoku_test.py run --runs 5             # run both models, write results/run-*.json
  python sudoku_test.py summary                  # print the table the video uses

Every answer is checked by code: it must keep all the given digits and be a valid Sudoku
(with one solution, that also means it equals the solution).
"""
import argparse, json, os, random, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
MODELS = ['gemini-4-carbon', 'gemini-4-argon']
# USD per 1M tokens, Google's announced introductory rates (same for both models). Thinking tokens bill as output.
PRICE_IN, PRICE_OUT = 2.00, 10.00


# ---------- Sudoku helpers ----------
def peers():
    P = []
    for i in range(81):
        r, c = divmod(i, 9)
        s = set()
        for k in range(9):
            s.add(r * 9 + k); s.add(k * 9 + c)
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for a in range(3):
            for b in range(3):
                s.add((br + a) * 9 + bc + b)
        s.discard(i)
        P.append(tuple(s))
    return P
PEERS = peers()


def count_solutions(g, limit=2, stats=None):
    g = list(g)
    def cands(i):
        used = {g[j] for j in PEERS[i]}
        return [d for d in range(1, 10) if d not in used]
    def rec():
        best, bc = -1, None
        for i in range(81):
            if g[i] == 0:
                c = cands(i)
                if not c: return 0
                if bc is None or len(c) < len(bc):
                    best, bc = i, c
                    if len(c) == 1: break
        if best < 0: return 1
        n = 0
        for d in bc:
            if stats is not None: stats['nodes'] += 1
            g[best] = d
            n += rec()
            if n >= limit: break
        g[best] = 0
        return n
    return rec()


def solve(g):
    g = list(g)
    def rec():
        best, bc = -1, None
        for i in range(81):
            if g[i] == 0:
                used = {g[j] for j in PEERS[i]}
                c = [d for d in range(1, 10) if d not in used]
                if not c: return False
                if bc is None or len(c) < len(bc): best, bc = i, c
        if best < 0: return True
        for d in bc:
            g[best] = d
            if rec(): return True
        g[best] = 0
        return False
    return g if rec() else None


def full_grid(rng):
    g = [0] * 81
    def rec(i):
        if i == 81: return True
        if g[i]: return rec(i + 1)
        ds = list(range(1, 10)); rng.shuffle(ds)
        used = {g[j] for j in PEERS[i]}
        for d in ds:
            if d not in used:
                g[i] = d
                if rec(i + 1): return True
        g[i] = 0
        return False
    rec(0)
    return g


def human_difficulty(p):
    """Search nodes a plain constraint solver needs (singles only, then guessing). Higher = harder."""
    st = {'nodes': 0}
    count_solutions(p, limit=2, stats=st)
    return st['nodes']


def make_puzzle(seed, tries=40, budget_s=100):
    rng = random.Random(seed)
    best = None
    t0 = time.time()
    for t in range(tries):
        if time.time() - t0 > budget_s: break
        sol = full_grid(rng)
        p = sol[:]
        order = list(range(81)); rng.shuffle(order)
        for i in order:
            keep = p[i]; p[i] = 0
            if count_solutions(p) != 1: p[i] = keep
        clues = sum(1 for x in p if x)
        diff = human_difficulty(p)
        if best is None or diff > best['difficulty_nodes']:
            best = {'seed': seed, 'try': t, 'clues': clues, 'difficulty_nodes': diff, 'puzzle': p, 'solution': sol}
    return best


def grid_text(p):
    rows = []
    for r in range(9):
        row = ''.join(str(x) if x else '.' for x in p[r * 9:r * 9 + 9])
        rows.append(' '.join([row[0:3], row[3:6], row[6:9]]))
        if r in (2, 5): rows.append('')
    return '\n'.join(rows)


def is_valid_solution(ans, puzzle):
    if len(ans) != 81 or any(d not in range(1, 10) for d in ans): return False
    if any(p and p != a for p, a in zip(puzzle, ans)): return False
    for i in range(81):
        if any(ans[i] == ans[j] for j in PEERS[i]): return False
    return True


PROMPT = """Solve this Sudoku puzzle. Each row, column and 3x3 box must contain the digits 1-9 exactly once. Dots are empty cells.

{grid}

Reply with the completed grid only: exactly 9 lines, each with 9 digits and nothing else, inside a block that starts with the line ANSWER and ends with the line END."""


def parse_answer(text):
    m = re.search(r'ANSWER\s*(.*?)\s*END', text, re.S)
    body = m.group(1) if m else text
    rows = [re.sub(r'\D', '', ln) for ln in body.splitlines()]
    rows = [r for r in rows if len(r) == 9]
    if len(rows) < 9: return None
    return [int(ch) for r in rows[-9:] for ch in r]


# ---------- API ----------
def key():
    if os.environ.get('GEMINI_API_KEY'): return os.environ['GEMINI_API_KEY']
    for p in [os.path.join(HERE, '..', '..', '.env'), os.path.join(HERE, '..', '.env')]:
        if os.path.exists(p):
            raw = open(p, encoding='utf-8').read()
            for k in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
                m = re.search(rf'^\s*(?:export\s+)?{k}\s*=\s*(.*?)\s*$', raw, re.M)
                if m and m.group(1).strip().strip('"\''): return m.group(1).strip().strip('"\'')
    raise RuntimeError('No GEMINI_API_KEY found')


def call(model, prompt, level, timeout=900):
    import requests
    body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'generationConfig': {'thinkingConfig': {'thinkingLevel': level}, 'maxOutputTokens': 32768}}
    t0 = time.time()
    r = requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                      headers={'x-goog-api-key': key(), 'Content-Type': 'application/json'},
                      data=json.dumps(body), timeout=timeout)
    dt = time.time() - t0
    j = r.json()
    if r.status_code != 200: return {'error': j.get('error', {}).get('message', r.text[:500]), 'seconds': dt}
    cand = (j.get('candidates') or [{}])[0]
    text = ''.join(p.get('text', '') for p in cand.get('content', {}).get('parts', []) if not p.get('thought'))
    return {'text': text, 'seconds': dt, 'usage': j.get('usageMetadata', {}), 'finish': cand.get('finishReason'),
            'modelVersion': j.get('modelVersion')}


def _req(method, path, body=None):
    import requests
    r = requests.request(method, f'https://generativelanguage.googleapis.com/v1beta/{path}',
                         headers={'x-goog-api-key': key(), 'Content-Type': 'application/json'},
                         data=json.dumps(body) if body is not None else None, timeout=60)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, {'error': {'message': r.text[:300]}}


def finish_record(model, idx, level, puz, text, usage, seconds, extra):
    tin = usage.get('total_input_tokens', 0)
    thought = usage.get('total_thought_tokens', 0)
    tout = usage.get('total_output_tokens', 0) + thought
    ans = parse_answer(text)
    return {'model': model, 'run': idx, 'level': level, 'text': text, 'seconds': seconds, 'usage': usage,
            'answer': ans, 'correct': bool(ans) and is_valid_solution(ans, puz['puzzle']),
            'wrong_cells': None if not ans else sum(1 for a, s in zip(ans, puz['solution']) if a != s),
            'tokens_in': tin, 'thought_tokens': thought, 'tokens_out_incl_thinking': tout,
            'cost_usd': round(tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT, 5), **extra}


def _secs_between(epoch, iso):
    import calendar
    try: return round(calendar.timegm(time.strptime(iso, '%Y-%m-%dT%H:%M:%SZ')) - epoch, 1)
    except Exception: return None


def bg_submit(models, runs, level, puz):
    """Start every attempt as a background Interaction (no connection held open)."""
    pend_f = os.path.join(RES, 'pending.json')
    pend = json.load(open(pend_f)) if os.path.exists(pend_f) else {}
    prompt = PROMPT.format(grid=grid_text(puz['puzzle']))
    for i in range(1, runs + 1):
        for m in models:
            tag = f'{m}-{level}-{i}'
            if tag in pend or os.path.exists(os.path.join(RES, f'run-{tag}.json')): continue
            t0 = time.time()
            code, j = _req('POST', 'interactions', {'model': m, 'input': prompt, 'background': True,
                                                     'generation_config': {'thinking_level': level}})
            if code != 200:
                print('submit failed', tag, code, j.get('error', {}).get('message')); continue
            pend[tag] = {'id': j['id'], 'model': m, 'run': i, 'level': level, 'submitted_epoch': t0,
                         'submitted_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0))}
            print('submitted', tag, flush=True)
    json.dump(pend, open(pend_f, 'w'), indent=1)


def bg_poll(puz, budget_s=160, every=4):
    pend_f = os.path.join(RES, 'pending.json')
    pend = json.load(open(pend_f)) if os.path.exists(pend_f) else {}
    t_end = time.time() + budget_s
    while pend and time.time() < t_end:
        for tag, p in list(pend.items()):
            code, j = _req('GET', f"interactions/{p['id']}")
            st = j.get('status')
            if code != 200 or st in ('in_progress', 'queued', None):
                if code != 200: print('poll error', tag, code, j.get('error', {}).get('message'))
                continue
            now = time.time()
            text = ''.join(c.get('text', '') for s in j.get('steps', []) if s.get('type') == 'model_output'
                           for c in s.get('content', []))
            rec = finish_record(p['model'], p['run'], p['level'], puz, text, j.get('usage', {}),
                                round(now - p['submitted_epoch'], 1),
                                {'status': st, 'interaction_id': p['id'], 'submitted_utc': p['submitted_utc'],
                                 'server_created': j.get('created'), 'server_updated': j.get('updated'),
                                 'seconds_to_server_updated': _secs_between(p['submitted_epoch'], j.get('updated')),
                                 'api': 'interactions (background)', 'timing_note': f'submit -> first poll seeing it done (polled every {every}s)'})
            json.dump(rec, open(os.path.join(RES, f'run-{tag}.json'), 'w'), indent=1)
            print('done', tag, st, rec['correct'], rec['seconds'], 's', rec['thought_tokens'], 'thought', flush=True)
            del pend[tag]
        json.dump(pend, open(pend_f, 'w'), indent=1)
        if pend: time.sleep(every)
    print('pending', len(pend), {t: round(time.time() - p['submitted_epoch']) for t, p in pend.items()})


def run_one(model, idx, puz, level):
    out = os.path.join(RES, f'run-{model}-{level}-{idx}.json')
    if os.path.exists(out): return json.load(open(out))
    res = call(model, PROMPT.format(grid=grid_text(puz['puzzle'])), level)
    rec = {'model': model, 'run': idx, 'level': level, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **res}
    if 'text' in res:
        ans = parse_answer(res['text'])
        u = res['usage']
        tin = u.get('promptTokenCount', 0)
        tout = u.get('candidatesTokenCount', 0) + u.get('thoughtsTokenCount', 0)
        rec.update({'answer': ans, 'correct': bool(ans) and is_valid_solution(ans, puz['puzzle']),
                    'wrong_cells': None if not ans else sum(1 for a, s in zip(ans, puz['solution']) if a != s),
                    'tokens_in': tin, 'tokens_out_incl_thinking': tout,
                    'cost_usd': round(tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT, 5)})
    json.dump(rec, open(out, 'w'), indent=1)
    print(model, idx, rec.get('correct'), round(rec['seconds'], 1), 's', rec.get('tokens_out_incl_thinking'), rec.get('error', ''), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['make', 'run', 'submit', 'poll', 'summary'])
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--level', default='high')
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--parallel', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    pf = os.path.join(HERE, 'puzzle.json')
    if a.cmd == 'make':
        puz = make_puzzle(a.seed)
        json.dump(puz, open(pf, 'w'), indent=1)
        print(grid_text(puz['puzzle'])); print('clues', puz['clues'], 'difficulty nodes', puz['difficulty_nodes'])
        return
    puz = json.load(open(pf))
    if a.cmd == 'submit':
        bg_submit(a.models.split(','), a.runs, a.level, puz); return
    if a.cmd == 'poll':
        bg_poll(puz)
        if os.path.exists(os.path.join(RES, 'pending.json')) and json.load(open(os.path.join(RES, 'pending.json'))): return
    if a.cmd == 'run':
        jobs = [(m, i) for i in range(1, a.runs + 1) for m in a.models.split(',')]
        with ThreadPoolExecutor(a.parallel) as ex:
            list(ex.map(lambda mi: run_one(mi[0], mi[1], puz, a.level), jobs))
    rows = {}
    for f in sorted(os.listdir(RES)):
        if f.startswith('run-') and f.endswith('.json'):
            r = json.load(open(os.path.join(RES, f)))
            rows.setdefault((r['model'], r['level']), []).append(r)
    summ = []
    for (m, lv), rs in sorted(rows.items()):
        ok = [r for r in rs if r.get('correct')]
        done = [r for r in rs if 'text' in r]
        s = {'model': m, 'level': lv, 'runs': len(rs), 'correct': len(ok),
             'median_seconds': sorted(r['seconds'] for r in done)[len(done) // 2] if done else None,
             'avg_tokens_out': round(sum(r['tokens_out_incl_thinking'] for r in done) / len(done)) if done else None,
             'avg_cost_usd': round(sum(r['cost_usd'] for r in done) / len(done), 4) if done else None,
             'errors': [r.get('error') for r in rs if 'error' in r],
             'wrong_cells': [r.get('wrong_cells') for r in rs]}
        summ.append(s); print(json.dumps(s))
    json.dump({'puzzle': puz, 'price_per_1m_usd': [PRICE_IN, PRICE_OUT], 'summary': summ}, open(os.path.join(RES, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
