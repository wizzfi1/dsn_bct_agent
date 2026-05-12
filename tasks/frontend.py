"""
Agent Demo Webpage
===================
Serves a simple interactive frontend for judges to test both tasks.
Mount this onto either FastAPI app or run standalone.
"""

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DSN × BCT LLM Agent — Demo</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Arial, sans-serif; background: #F5F7FA; color: #222; }

    .header {
      background: #1B3A6B; color: white; padding: 24px 40px;
      border-bottom: 4px solid #F4A226;
    }
    .header h1 { font-size: 22px; font-weight: 700; }
    .header p  { font-size: 13px; color: #A8BFDF; margin-top: 4px; }

    .tabs {
      display: flex; background: #fff;
      border-bottom: 1px solid #DDD; padding: 0 40px;
    }
    .tab {
      padding: 14px 24px; cursor: pointer; font-size: 14px;
      font-weight: 600; color: #888; border-bottom: 3px solid transparent;
    }
    .tab.active { color: #1B3A6B; border-bottom-color: #F4A226; }

    .container { max-width: 900px; margin: 32px auto; padding: 0 24px; }
    .panel { display: none; }
    .panel.active { display: block; }

    .card {
      background: #fff; border-radius: 10px;
      border: 1px solid #E0E6EF; padding: 28px; margin-bottom: 20px;
    }
    .card h2 { font-size: 16px; color: #1B3A6B; margin-bottom: 16px; font-weight: 700; }

    label { font-size: 13px; font-weight: 600; color: #444; display: block; margin-bottom: 6px; }
    input, textarea, select {
      width: 100%; padding: 10px 14px; border: 1px solid #CDD5E0;
      border-radius: 6px; font-size: 14px; font-family: Arial;
      background: #FAFBFC; margin-bottom: 16px;
    }
    textarea { resize: vertical; min-height: 100px; }

    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

    button {
      background: #1B3A6B; color: white; border: none;
      padding: 12px 28px; border-radius: 6px; font-size: 14px;
      font-weight: 600; cursor: pointer; width: 100%;
    }
    button:hover { background: #2E5FA3; }
    button:disabled { background: #999; cursor: not-allowed; }

    .result {
      background: #EEF4FB; border: 1px solid #BDD0E8;
      border-radius: 8px; padding: 20px; margin-top: 20px;
      display: none;
    }
    .result.show { display: block; }
    .result h3 { font-size: 14px; font-weight: 700; color: #1B3A6B; margin-bottom: 12px; }

    .rating-badge {
      display: inline-block; background: #F4A226; color: white;
      padding: 4px 14px; border-radius: 20px; font-weight: 700;
      font-size: 18px; margin-bottom: 12px;
    }
    .review-text {
      font-size: 15px; line-height: 1.7; color: #333;
      background: white; padding: 16px; border-radius: 6px;
      border-left: 4px solid #F4A226; margin-bottom: 12px;
    }
    .rec-item {
      background: white; border-radius: 6px; padding: 14px 16px;
      margin-bottom: 10px; border-left: 4px solid #1B3A6B;
    }
    .rec-rank { font-size: 11px; color: #888; font-weight: 600; }
    .rec-name { font-size: 15px; font-weight: 700; color: #1B3A6B; margin: 2px 0; }
    .rec-score { font-size: 12px; color: #2E5FA3; }
    .rec-explain { font-size: 13px; color: #555; margin-top: 4px; }

    .spinner {
      display: inline-block; width: 16px; height: 16px;
      border: 2px solid #fff; border-top-color: transparent;
      border-radius: 50%; animation: spin 0.7s linear infinite;
      vertical-align: middle; margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .error { background: #FEE; border-color: #F99; color: #922; }
    .tag { display: inline-block; background: #E8F0FB; color: #2E5FA3;
           font-size: 11px; padding: 2px 8px; border-radius: 12px; margin: 2px; }
    .confidence { font-size: 12px; color: #888; margin-top: 8px; }
  </style>
</head>
<body>

<div class="header">
  <h1>DSN × BCT LLM Agent Challenge 3.0</h1>
  <p>Unified User Modelling &amp; Personalised Recommendation Agent &nbsp;·&nbsp;</p>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('a')">Task A — Review Simulation</div>
  <div class="tab" onclick="switchTab('b')">Task B — Recommendation</div>
  <div class="tab" onclick="switchTab('cold')">Task B — Cold Start</div>
</div>

<!-- ── Task A ──────────────────────────────────────────────────────────────── -->
<div class="container">
<div id="panel-a" class="panel active">
  <div class="card">
    <h2>Simulate a user review for an unseen item</h2>

    <label>User Review History (one review per line, format: RATING | CATEGORY | review text)</label>
    <textarea id="a-history" rows="6">5 | Nigerian Cuisine | The jollof rice was perfect and service was fast!
3 | Fast Food | Good food but the wait was too long abeg.
5 | Nigerian Cuisine | Amazing suya spot, will definitely come back!
2 | Fine Dining | Overpriced and the portion was small. E no worth am.
4 | Street Food | Nice pepper soup, very affordable and filling.</textarea>

    <div class="row">
      <div>
        <label>Item Name</label>
        <input id="a-item-name" value="Mama Cass Restaurant" />
      </div>
      <div>
        <label>Category</label>
        <input id="a-category" value="Nigerian Cuisine" />
      </div>
    </div>

    <label>Item Attributes (comma separated)</label>
    <input id="a-attrs" value="local food, affordable, pepper soup, outdoor seating" />

    <div class="row">
      <div>
        <label>Price Range</label>
        <input id="a-price" value="affordable" />
      </div>
      <div>
        <label>Location</label>
        <input id="a-location" value="Lagos Island" />
      </div>
    </div>

    <button onclick="runTaskA()" id="btn-a">Simulate Review</button>
  </div>

  <div class="result" id="result-a">
    <h3>Simulated Review</h3>
    <div class="rating-badge" id="a-rating">★ 4.5</div>
    <div class="review-text" id="a-review-text"></div>
    <div class="confidence" id="a-confidence"></div>
    <div id="a-reasoning" style="font-size:13px;color:#666;margin-top:8px;"></div>
  </div>
</div>

<!-- ── Task B Warm ──────────────────────────────────────────────────────────── -->
<div id="panel-b" class="panel">
  <div class="card">
    <h2>Personalised recommendations for an existing user</h2>

    <label>User Review History (one review per line, format: RATING | CATEGORY | review text)</label>
    <textarea id="b-history" rows="5">5 | Nigerian Cuisine | The jollof rice was perfect abeg!
2 | Fine Dining | Overpriced and slow service. E no worth am.
5 | Street Food | Best pepper soup in Lagos, very affordable.
4 | Fast Food | Quick service and tasty food. Will return.
1 | Hotels | Very dirty room and rude staff. Never again.</textarea>

    <label>Candidate Items (JSON array)</label>
    <textarea id="b-candidates" rows="8">[
  {"item_id":"c001","item_name":"Bukka Hut VI","category":"Nigerian Cuisine","attributes":["local food","affordable","amala"],"avg_rating":4.7,"popularity":850},
  {"item_id":"c002","item_name":"The Yellow Chilli","category":"Nigerian Fine Dining","attributes":["premium","local dishes","elegant"],"avg_rating":4.8,"popularity":500},
  {"item_id":"c003","item_name":"Mama Put Central","category":"Street Food","attributes":["affordable","local","pepper soup"],"avg_rating":4.3,"popularity":2000},
  {"item_id":"c004","item_name":"Sky Lounge","category":"Bar","attributes":["expensive","cocktails","slow service"],"avg_rating":3.5,"popularity":200},
  {"item_id":"c005","item_name":"Tantalizers","category":"Fast Food","attributes":["fast service","local menu","affordable"],"avg_rating":3.8,"popularity":1200}
]</textarea>

    <button onclick="runTaskB()" id="btn-b">Get Recommendations</button>
  </div>

  <div class="result" id="result-b">
    <h3>Personalised Recommendations</h3>
    <div id="b-recs"></div>
  </div>
</div>

<!-- ── Task B Cold Start ──────────────────────────────────────────────────── -->
<div id="panel-cold" class="panel">
  <div class="card">
    <h2>Recommendations for a brand new user (cold start)</h2>
    <p style="font-size:13px;color:#888;margin-bottom:20px;">
      No review history needed — just 3 quick questions.
    </p>

    <label>Q1: What types of things do you usually enjoy?</label>
    <input id="cold-q1" value="I love Nigerian food especially jollof rice and pepper soup" />

    <label>Q2: On a scale of 1–5, how would you rate something just "okay"?</label>
    <input id="cold-q2" value="3" type="number" min="1" max="5" />

    <label>Q3: What's one thing that would immediately make you give a low rating?</label>
    <input id="cold-q3" value="Bad service and overpriced food with no value for money" />

    <label style="margin-top:8px;">Candidate Items (JSON array)</label>
    <textarea id="cold-candidates" rows="8">[
  {"item_id":"c001","item_name":"Bukka Hut VI","category":"Nigerian Cuisine","attributes":["local food","affordable","amala"],"avg_rating":4.7,"popularity":850},
  {"item_id":"c002","item_name":"Chinese Palace","category":"Chinese","attributes":["dim sum","quiet","expensive"],"avg_rating":4.2,"popularity":300},
  {"item_id":"c003","item_name":"Mama Put Central","category":"Street Food","attributes":["affordable","local","pepper soup"],"avg_rating":4.3,"popularity":2000},
  {"item_id":"c004","item_name":"Sky Lounge","category":"Bar","attributes":["expensive","cocktails","slow service"],"avg_rating":3.5,"popularity":200},
  {"item_id":"c005","item_name":"Tantalizers","category":"Fast Food","attributes":["fast service","local menu","affordable"],"avg_rating":3.8,"popularity":1200}
]</textarea>

    <button onclick="runColdStart()" id="btn-cold">Get Cold-Start Recommendations</button>
  </div>

  <div class="result" id="result-cold">
    <h3>Cold-Start Recommendations</h3>
    <div id="cold-recs"></div>
  </div>
</div>
</div>

<script>
  // ── Config ────────────────────────────────────────────────────────────────
  // Update these URLs after deploying to Render
  const TASK_A_URL = window.location.origin.includes('8000')
    ? 'http://localhost:8000'
    : 'https://dsn-bct-task-a.onrender.com';

  const TASK_B_URL = window.location.origin.includes('8001')
    ? 'http://localhost:8001'
    : 'https://dsn-bct-task-b.onrender.com';

  // ── Tab switching ─────────────────────────────────────────────────────────
  function switchTab(name) {
    document.querySelectorAll('.tab').forEach((t, i) => {
      t.classList.toggle('active', ['a','b','cold'][i] === name);
    });
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.getElementById('panel-' + name).classList.add('active');
  }

  // ── Parse history textarea ─────────────────────────────────────────────────
  function parseHistory(text) {
    return text.trim().split('\n').map(line => {
      const parts = line.split('|').map(s => s.trim());
      return {
        rating: parseFloat(parts[0]) || 3.0,
        category: parts[1] || 'General',
        text: parts[2] || line,
      };
    }).filter(r => r.text);
  }

  // ── Loading state ─────────────────────────────────────────────────────────
  function setLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    btn.disabled = loading;
    btn.innerHTML = loading
      ? '<span class="spinner"></span>Thinking...'
      : btn.dataset.label;
  }

  document.querySelectorAll('button').forEach(b => {
    b.dataset.label = b.textContent;
  });

  function showError(resultId, msg) {
    const el = document.getElementById(resultId);
    el.className = 'result show error';
    el.innerHTML = '<h3>Error</h3><p>' + msg + '</p>';
  }

  // ── Task A ────────────────────────────────────────────────────────────────
  async function runTaskA() {
    setLoading('btn-a', true);
    document.getElementById('result-a').classList.remove('show');

    const history = parseHistory(document.getElementById('a-history').value);
    const attrs = document.getElementById('a-attrs').value.split(',').map(s => s.trim());

    const body = {
      user_id: 'demo_user_' + Date.now(),
      review_history: history,
      item: {
        item_id: 'demo_item',
        item_name: document.getElementById('a-item-name').value,
        category: document.getElementById('a-category').value,
        attributes: attrs,
        price_range: document.getElementById('a-price').value,
        location: document.getElementById('a-location').value,
      },
      include_reasoning: true,
    };

    try {
      const res = await fetch(TASK_A_URL + '/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'API error');

      document.getElementById('a-rating').textContent = '★ ' + data.predicted_rating;
      document.getElementById('a-review-text').textContent = data.review_text;
      document.getElementById('a-confidence').textContent = 'Confidence: ' + data.confidence;
      document.getElementById('a-reasoning').textContent = data.reasoning ? 'Reasoning: ' + data.reasoning : '';
      document.getElementById('result-a').className = 'result show';
    } catch(e) {
      showError('result-a', e.message);
    } finally {
      setLoading('btn-a', false);
    }
  }

  // ── Task B Warm ───────────────────────────────────────────────────────────
  async function runTaskB() {
    setLoading('btn-b', true);
    document.getElementById('result-b').classList.remove('show');

    const history = parseHistory(document.getElementById('b-history').value);
    let candidates;
    try {
      candidates = JSON.parse(document.getElementById('b-candidates').value);
    } catch(e) {
      showError('result-b', 'Invalid JSON in candidates field');
      setLoading('btn-b', false);
      return;
    }

    const body = {
      user_id: 'demo_user_' + Date.now(),
      review_history: history,
      candidates,
      top_k: 5,
    };

    try {
      const res = await fetch(TASK_B_URL + '/recommend/warm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'API error');
      renderRecs('b-recs', data.recommendations);
      document.getElementById('result-b').className = 'result show';
    } catch(e) {
      showError('result-b', e.message);
    } finally {
      setLoading('btn-b', false);
    }
  }

  // ── Task B Cold Start ─────────────────────────────────────────────────────
  async function runColdStart() {
    setLoading('btn-cold', true);
    document.getElementById('result-cold').classList.remove('show');

    let candidates;
    try {
      candidates = JSON.parse(document.getElementById('cold-candidates').value);
    } catch(e) {
      showError('result-cold', 'Invalid JSON in candidates field');
      setLoading('btn-cold', false);
      return;
    }

    const body = {
      user_id: 'cold_user_' + Date.now(),
      elicitation_answers: {
        q1: document.getElementById('cold-q1').value,
        q2: document.getElementById('cold-q2').value,
        q3: document.getElementById('cold-q3').value,
      },
      candidates,
      top_k: 3,
    };

    try {
      const res = await fetch(TASK_B_URL + '/recommend/cold-start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'API error');
      renderRecs('cold-recs', data.recommendations);
      document.getElementById('result-cold').className = 'result show';
    } catch(e) {
      showError('result-cold', e.message);
    } finally {
      setLoading('btn-cold', false);
    }
  }

  // ── Render recommendations ────────────────────────────────────────────────
  function renderRecs(containerId, recs) {
    const el = document.getElementById(containerId);
    el.innerHTML = recs.map(r => `
      <div class="rec-item">
        <div class="rec-rank">Rank #${r.rank}</div>
        <div class="rec-name">${r.item_name}</div>
        <div class="rec-score">Score: ${(r.score * 100).toFixed(0)}% match</div>
        <div class="rec-explain">${r.explanation}</div>
        ${r.matched_preferences && r.matched_preferences.length
          ? '<div style="margin-top:6px">' + r.matched_preferences.map(p => `<span class="tag">${p}</span>`).join('') + '</div>'
          : ''}
      </div>
    `).join('');
  }
</script>
</body>
</html>
"""