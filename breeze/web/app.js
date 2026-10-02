/* Breeze browser workspace. Chat, credentials, and settings stay in page memory. */

// Incremental SSE parsing: TextDecoder handles split UTF-8 code points; this
// parser handles LF, CRLF, CR, comments, multiline data, and split frame endings.
export class SSEParser {
  constructor(onData) {
    this.onData = onData;
    this.line = '';
    this.data = [];
    this.afterCR = false;
    this.stopped = false;
  }
  push(text) {
    for (const char of text) {
      if (this.stopped) return;
      if (this.afterCR) {
        this.afterCR = false;
        if (char === '\n') continue;
      }
      if (char === '\r' || char === '\n') {
        this.consumeLine();
        this.afterCR = char === '\r';
      } else {
        this.line += char;
        if (this.line.length > 1048576) throw new Error('Invalid stream frame');
      }
    }
  }
  consumeLine() {
    const line = this.line;
    this.line = '';
    if (line === '') {
      if (this.data.length) {
        const data = this.data.join('\n');
        this.data = [];
        if (this.onData(data) === false) this.stopped = true;
      }
      return;
    }
    if (line.startsWith(':')) return;
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'data') {
      this.data.push(value);
      if (this.data.length > 4096) throw new Error('Invalid stream frame');
    }
  }
  end() {
    if (this.stopped) return;
    if (this.line) this.consumeLine();
    this.consumeLine();
  }
}

export function validateBudget(value, limit) {
  if (!/^\d+$/.test(String(value))) return null;
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 1 && number <= limit ? number : null;
}

export function buildMessages(system, history, prompt) {
  return [...(system.trim() ? [{ role: 'system', content: system.trim() }] : []),
    ...history.map(({ role, content }) => ({ role, content })),
    { role: 'user', content: prompt }];
}

const ERROR_MESSAGES = {
  invalid_api_key: 'This service needs a valid API key. Connect again with the key configured on your server.',
  queue_full: 'The server queue is full. Your request was not admitted. Wait for capacity, then use Edit & retry.',
  context_length_exceeded: 'The conversation and requested output do not fit the context window. Lower the output budget, shorten your prompt, or start a new conversation.',
  output_limit_exceeded: 'The output budget exceeds the server limit. Refresh status and choose a smaller budget.',
  deadline_exceeded: 'The request reached the server time limit. Try a shorter prompt or a smaller output budget.',
  cancelled: 'The request was cancelled. Any partial answer is excluded from future conversation context.',
  inference_failed: 'The inference service encountered a problem. Ask the server operator to check it and restart if needed.',
  not_ready: 'The service is not ready. Check the server, then refresh its status.',
  model_not_found: 'The model has changed or is unavailable. Refresh status and start a new conversation.',
  body_too_large: 'This conversation is too large to submit. Shorten the source text or start a new conversation.',
  validation_error: 'The request was rejected. Use non-empty plain text, avoid model control-token delimiters, and shorten long conversations.',
  invalid_request: 'The service could not accept this request. Check the text and output budget, then try again.',
  invalid_origin: 'The service rejected this page origin. Open the workspace at the server’s configured address.',
  invalid_host: 'This server address is not allowed. Check the configured host with your server operator.',
};

class ServiceIssue extends Error {
  constructor(code, status = 0) {
    super(ERROR_MESSAGES[code] || (status === 429 ? ERROR_MESSAGES.queue_full :
      status >= 500 ? 'The service is unavailable. Check the server and retry manually.' :
        'The request could not be completed. Refresh the connection and retry manually.'));
    this.code = code;
    this.status = status;
  }
}

async function responseIssue(response) {
  let code = '';
  try { code = (await response.json())?.error?.code; } catch { /* Never echo response bodies. */ }
  return new ServiceIssue(response.status === 401 ? 'invalid_api_key' : code, response.status);
}

const TEMPLATES = {
  meeting: `Summarize these meeting notes. Separate decisions, open questions, and action items. For each action item, include the owner and due date if stated; do not invent missing details.\n\nSource: Northstar onboarding review, 12 September\nAttendees: Maya (product), Leo (support), Priya (engineering).\nCustomers are getting stuck when inviting teammates. Leo counted 14 related tickets this week. We agreed to replace the invitation email copy before the next release. Maya will send a draft by Tuesday. Priya will check whether expired invitations can be resent and report back on Thursday. Leo will collect three anonymized examples for Maya by Monday. We have not decided whether to change the default invitation expiry. Next check-in: Friday at 10:00.`,
  support: `Draft a concise, empathetic support reply using only the facts below. Acknowledge the inconvenience, explain the next step, and do not promise an unconfirmed resolution date.\n\nCustomer message:\nHi, I invited two teammates yesterday but neither received their invitation email. We need to get our project started today. Can you help? — Sam\n\nKnown facts:\nThe account is active. Invitations show as pending. The customer can ask teammates to check spam and confirm that the email addresses are correct in Settings > Team. An administrator can resend a pending invitation from that page. If the emails still do not arrive, support needs the affected email domains and the approximate resend time to investigate. Do not ask for passwords.`,
  update: `Polish this customer update so it is clear, warm, and easy to scan. Keep the facts and uncertainty intact. Include a subject line and a concise next-steps section.\n\nRough draft:\nHi everyone — quick update on the September reporting changes. The new export filters are ready and we plan to enable them for your workspace on Monday. Your existing saved reports will stay as they are. The scheduled delivery feature needs more testing, so it won't be part of Monday's update. We don't have a confirmed date for that yet. Please try the new filters after Monday and send any feedback to your account contact by Friday. We'll share another update when scheduled delivery is ready. Thanks for helping us get this right.`,
};

function startWorkspace() {
  const $ = (id) => document.getElementById(id);
  const state = {
    key: '', status: null, connection: 'checking', disconnected: false,
    epoch: 0, statusRequest: null, schemaRequest: null, active: null,
    history: [], conversationModel: null, lockedPrompted: false, connecting: false,
  };
  const scroll = $('conversation-scroll');
  let followStream = true;
  let liveTimer;

  function announce(text) {
    clearTimeout(liveTimer);
    liveTimer = setTimeout(() => { $('live-status').textContent = text; }, 80);
  }
  function notify(title, text) {
    $('notice-title').textContent = title;
    $('notice-text').textContent = text;
    $('notice').hidden = false;
    announce(`${title}. ${text}`);
  }
  function headers(key = state.key) {
    return key ? { Authorization: `Bearer ${key}` } : {};
  }
  function ready() { return state.connection === 'ready' && state.status?.status === 'ready'; }
  function setRequestState(text, dot = '') {
    $('request-state').textContent = text;
    $('request-dot').dataset.state = dot;
  }
  function updateControls() {
    const busy = Boolean(state.active);
    $('send').disabled = busy || !ready() || !$('prompt').value.trim();
    $('stop').hidden = !busy;
    $('stop').disabled = Boolean(state.active?.controller.signal.aborted);
    $('system-prompt').disabled = busy;
    $('max-tokens').disabled = busy;
    $('manage-key').disabled = busy;
    $('connect-submit').disabled = busy || state.connecting;
    $('api-key').disabled = busy || state.connecting;
    $('copy-example').disabled = busy || !ready();
    $('load-schema').disabled = busy || !ready() || Boolean(state.schemaRequest);
    $('refresh-status').disabled = state.disconnected || Boolean(state.statusRequest);
    document.querySelectorAll('[data-template]').forEach((button) => { button.disabled = busy; });
  }

  function updateExample() {
    if (!state.status) {
      $('api-example').textContent = 'Connect to load an example with the actual service model.';
      return;
    }
    const budget = validateBudget($('max-tokens').value, state.status.max_output_tokens)
      ?? Math.min(256, state.status.max_output_tokens);
    // JSON-encoded strings are also valid Python string literals for these
    // validated ASCII model IDs and HTTP origins. Never interpolate credentials.
    const model = JSON.stringify(state.status.model);
    const origin = JSON.stringify(location.origin);
    $('api-example').textContent = `import getpass\nimport json\nimport urllib.error\nimport urllib.request\n\nbase_url = ${origin}\nkey = getpass.getpass("API key (blank if not required): ")\nheaders = {"Content-Type": "application/json"}\nif key:\n    headers["Authorization"] = "Bearer " + key\n\npayload = {\n    "model": ${model},\n    "messages": [{"role": "user", "content": "Write a friendly welcome for a new teammate."}],\n    "max_tokens": ${budget},\n    "stream": True,\n    "stream_options": {"include_usage": True},\n    "temperature": 0,\n}\nrequest = urllib.request.Request(\n    base_url + "/v1/chat/completions",\n    data=json.dumps(payload).encode("utf-8"),\n    headers=headers, method="POST",\n)\ntry:\n    with urllib.request.urlopen(request) as response:\n        # This service emits one JSON object per SSE data line.\n        for line in response:\n            text = line.decode("utf-8").strip()\n            if not text.startswith("data:"):\n                continue\n            data = text[5:].strip()\n            if data == "[DONE]":\n                break\n            event = json.loads(data)\n            if "error" in event:\n                print("\\nRequest failed; discard partial history.")\n                break\n            choices = event.get("choices", [])\n            if choices:\n                print(choices[0].get("delta", {}).get("content", ""),\n                      end="", flush=True)\n            if "usage" in event:\n                print("\\nUsage:", event["usage"])\n            if "breeze" in event:\n                print("\\nServer timings:", event["breeze"])\nexcept urllib.error.HTTPError as error:\n    print("Request rejected (HTTP", error.code, "). Retry manually.")\nexcept urllib.error.URLError:\n    print("Could not reach the service.")`;
  }

  function paintConnection() {
    const status = state.status;
    const labels = { checking: 'Connecting', ready: 'Connected', locked: 'Key required',
      offline: 'Offline', unavailable: 'Not ready', disconnected: 'Disconnected' };
    const label = labels[state.connection];
    $('connection-label').textContent = label;
    $('sidebar-status').textContent = state.connection === 'ready' ? 'Service ready' : label;
    $('detail-connection').textContent = label;
    const dot = ready() ? 'ready' : ['locked', 'offline', 'unavailable'].includes(state.connection) ? 'error' : '';
    $('connection-dot').dataset.state = dot;
    $('server-dot').dataset.state = dot;
    $('sidebar-model').textContent = status?.model ?? 'No service details available';
    $('model-label').textContent = status?.model ?? 'No model connected';
    $('demo-banner').hidden = status?.mode !== 'demo';
    if (status?.mode === 'demo') $('rate-metric').hidden = true;
    $('detail-model').textContent = status?.model ?? '—';
    $('detail-mode').textContent = status ? (status.mode === 'demo' ? 'Scripted demo · no model loaded' : 'Native CPU inference') : '—';
    $('detail-threads').textContent = status?.threads ?? '—';
    $('detail-context').textContent = status ? `${status.max_context_tokens.toLocaleString()} tokens` : '—';
    $('detail-auth').textContent = status ? (status.authentication_required ? 'Bearer key required' : 'No key required') : (state.connection === 'locked' ? 'Bearer key required' : 'Not yet known');
    for (const name of ['active', 'waiting', 'capacity']) $('queue-' + name).textContent = status?.queue?.[name] ?? '—';
    if (status) {
      $('max-tokens').max = String(status.max_output_tokens);
      $('token-hint').textContent = `1–${status.max_output_tokens.toLocaleString()} output tokens. Input + output must fit the context window.`;
      $('context-note').textContent = `${status.max_context_tokens.toLocaleString()} context · ${status.threads} CPU threads`;
    } else {
      $('token-hint').textContent = 'Connect to see the server limit.';
      $('context-note').textContent = 'Local service · Streaming responses';
    }
    updateExample();
    updateControls();
  }

  function showConnect() {
    $('connect-error').hidden = true;
    $('api-key').value = '';
    if (!$('connect-dialog').open) $('connect-dialog').showModal();
    updateControls();
  }

  async function refreshStatus({ interactive = false, candidateKey = state.key, connect = false } = {}) {
    if (state.disconnected && !connect) return false;
    if (state.statusRequest) {
      if (!connect) return false;
      state.statusRequest.abort();
    }
    const controller = new AbortController();
    const epoch = state.epoch;
    state.statusRequest = controller;
    const timeout = setTimeout(() => controller.abort(), 10000);
    updateControls();
    try {
      const response = await fetch('/api/status', { headers: headers(candidateKey), signal: controller.signal, cache: 'no-store', credentials: 'omit', redirect: 'error' });
      if (!response.ok) throw await responseIssue(response);
      const status = await response.json();
      if (!status || !['demo', 'native'].includes(status.mode) ||
          !/^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/.test(status.model) ||
          !Number.isInteger(status.max_output_tokens) || status.max_output_tokens < 1 || status.max_output_tokens > 4096 ||
          !Number.isInteger(status.max_context_tokens) || !Number.isInteger(status.threads)) throw new ServiceIssue('invalid_status');
      if (state.epoch !== epoch || state.statusRequest !== controller) return false;
      if (state.conversationModel && state.conversationModel !== status.model && !state.active) {
        state.history = [];
        state.conversationModel = null;
        notify('The model changed', 'Earlier messages remain visible but will not be sent to the new model. Your next message starts fresh context.');
      }
      const firstStatus = !state.status;
      state.status = status;
      state.connection = status.status === 'ready' ? 'ready' : 'unavailable';
      state.disconnected = false;
      if (connect) state.key = candidateKey;
      if (firstStatus && !state.active) $('max-tokens').value = String(Math.min(Number($('max-tokens').value) || 256, status.max_output_tokens));
      if (connect) {
        $('connect-dialog').close();
        $('notice').hidden = true;
        announce('Connected to the Breeze service.');
      } else if (interactive) announce('Service status refreshed.');
      if (!state.active && !$('messages').childElementCount) setRequestState(ready() ? 'Ready when you are' : 'Service not ready', ready() ? 'ready' : 'error');
      return true;
    } catch (error) {
      if (state.epoch !== epoch || state.statusRequest !== controller) return false;
      const previousConnection = state.connection;
      state.status = null;
      const locked = error instanceof ServiceIssue && error.status === 401;
      state.connection = locked ? 'locked' : 'offline';
      const message = locked ? ERROR_MESSAGES.invalid_api_key : error instanceof ServiceIssue ? error.message : 'Could not reach the service. Check that Breeze is running at this address, then reconnect.';
      if (connect) {
        state.key = '';
        $('connect-error').textContent = message;
        $('connect-error').hidden = false;
      } else if (interactive || previousConnection !== state.connection) notify(locked ? 'Connection locked' : 'Service unavailable', message);
      if (locked && !connect && !state.lockedPrompted) {
        state.lockedPrompted = true;
        showConnect();
      }
      if (!state.active) setRequestState(locked ? 'Connect to continue' : 'Service unavailable', 'error');
      return false;
    } finally {
      clearTimeout(timeout);
      if (state.statusRequest === controller) {
        state.statusRequest = null;
        paintConnection();
      }
    }
  }

  function resetMetrics() {
    for (const name of ['usage', 'elapsed', 'ttft', 'queue', 'rate']) $('metric-' + name).textContent = '—';
    $('usage-label').textContent = 'Tokens';
    $('rate-metric').hidden = true;
    $('metric-usage').removeAttribute('title');
  }
  function seconds(value) { return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? `${value.toFixed(2)}s` : '—'; }
  function showMetrics(job) {
    const timing = job.timing;
    const usage = job.usage;
    const estimated = timing?.usage_is_estimate || timing?.mode === 'demo' || job.mode === 'demo';
    $('usage-label').textContent = estimated ? 'Tokens (est.)' : 'Tokens';
    if (usage && ['prompt_tokens', 'completion_tokens', 'total_tokens'].every((key) => Number.isInteger(usage[key]) && usage[key] >= 0)) {
      $('metric-usage').textContent = `${usage.prompt_tokens} in / ${usage.completion_tokens} out`;
      $('metric-usage').title = `${usage.total_tokens} total tokens${estimated ? ' (estimated)' : ''}`;
    }
    $('metric-elapsed').textContent = seconds(timing?.total_seconds);
    $('metric-ttft').textContent = seconds(timing?.time_to_first_token_seconds);
    $('metric-queue').textContent = seconds(timing?.queue_seconds);
    const rate = timing?.decode_tokens_per_second;
    const showRate = !estimated && timing?.mode === 'native' && Number.isFinite(rate) && rate >= 0;
    $('rate-metric').hidden = !showRate;
    $('metric-rate').textContent = showRate ? rate.toFixed(1) : '—';
  }

  function moveToLatest(force = false) {
    if (force || followStream) {
      scroll.scrollTop = scroll.scrollHeight;
      $('jump-latest').hidden = true;
    } else $('jump-latest').hidden = false;
  }

  async function copyText(text, button) {
    const label = button.textContent;
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(text);
      button.textContent = 'Copied';
      announce('Copied to clipboard.');
      setTimeout(() => { if (button.isConnected) button.textContent = label; }, 1600);
    } catch {
      notify('Copy unavailable', 'Your browser did not allow clipboard access. Select the text and copy it manually.');
    }
  }

  function addMessage(role, content) {
    const article = document.createElement('article');
    article.className = `message ${role}`;
    article.setAttribute('aria-label', role === 'user' ? 'Your message' : 'Breeze response');
    const heading = document.createElement('div');
    heading.className = 'message-heading';
    const avatar = document.createElement('span');
    avatar.className = 'message-avatar';
    avatar.setAttribute('aria-hidden', 'true');
    avatar.textContent = role === 'user' ? 'Y' : '✳';
    const name = document.createElement('strong');
    name.textContent = role === 'user' ? 'You' : 'Breeze';
    const badge = document.createElement('span');
    badge.className = 'message-state';
    const body = document.createElement('p');
    body.className = 'message-content';
    body.textContent = content;
    heading.append(avatar, name, badge);
    article.append(heading, body);
    $('messages').append(article);
    $('welcome').hidden = true;
    return { article, heading, badge, body };
  }

  function addCopy(message, text, partial = false) {
    if (!text.trim()) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'message-copy';
    button.textContent = partial ? 'Copy partial' : 'Copy response';
    button.addEventListener('click', () => copyText(text, button));
    message.heading.append(button);
  }

  function setDraft(text) {
    if ($('prompt').value.trim() && $('prompt').value !== text && !confirm('Replace the current unsent draft?')) return;
    $('prompt').value = text;
    $('prompt').focus();
    updateControls();
  }

  async function consumeStream(response, job) {
    if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream')) throw new ServiceIssue('invalid_stream');
    const reader = response.body.getReader();
    job.reader = reader;
    const decoder = new TextDecoder('utf-8', { fatal: true });
    let done = false;
    const parser = new SSEParser((data) => {
      if (job.controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
      if (data === '[DONE]') { done = true; return false; }
      let event;
      try { event = JSON.parse(data); } catch { throw new ServiceIssue('invalid_stream'); }
      if (!event || typeof event !== 'object') throw new ServiceIssue('invalid_stream');
      if (event.error) throw new ServiceIssue(event.error.code);
      const choice = event.choices?.[0];
      if (typeof choice?.delta?.content === 'string') {
        if (job.finished && choice.delta.content) throw new ServiceIssue('invalid_stream');
        job.text += choice.delta.content;
        if (job.text.length > 1048576) throw new ServiceIssue('invalid_stream');
        if (state.active === job) {
          job.message.body.textContent = job.text || 'Waiting for the first token…';
          job.message.article.classList.toggle('pending', !job.text);
          job.message.badge.textContent = job.mode === 'demo' ? 'Scripted demo · streaming' : 'Streaming';
          setRequestState('Receiving response', 'busy');
          moveToLatest();
        }
      }
      if (choice?.finish_reason != null) {
        job.finished = true;
        job.finishReason = choice.finish_reason;
        job.timing = event.breeze ?? null;
      }
      if (event.usage) job.usage = event.usage;
      return true;
    });
    try {
      while (!done) {
        const result = await reader.read();
        if (result.done) {
          parser.push(decoder.decode());
          parser.end();
          break;
        }
        parser.push(decoder.decode(result.value, { stream: true }));
      }
      if (job.controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
      if (!done || !job.finished || !['stop', 'length'].includes(job.finishReason) || !job.text.trim()) throw new ServiceIssue('incomplete_stream');
    } finally {
      try { await reader.cancel(); } catch { /* The transport may already be aborted. */ }
      reader.releaseLock();
      job.reader = null;
    }
  }

  async function sendMessage(event) {
    event?.preventDefault();
    if (state.active) return;
    if (!ready()) { showConnect(); return; }
    const prompt = $('prompt').value.trim();
    if (!prompt) return;
    const budget = validateBudget($('max-tokens').value, state.status.max_output_tokens);
    if (budget === null) {
      notify('Check the output budget', `Enter a whole number from 1 to ${state.status.max_output_tokens}.`);
      $('generation-controls').open = true;
      $('max-tokens').focus();
      return;
    }
    if (state.conversationModel && state.conversationModel !== state.status.model) {
      state.history = [];
      state.conversationModel = null;
    }
    const messages = buildMessages($('system-prompt').value, state.history, prompt);
    if (messages.length > 64 || messages.reduce((sum, message) => sum + [...message.content].length, 0) > 65536) {
      notify('Conversation is too long', 'Shorten your source text or start a new conversation before sending. The text API accepts up to 64 messages and 65,536 combined characters.');
      return;
    }
    if (messages.some((message) => /<\||\|>|\u0000/.test(message.content))) {
      notify('Plain text only', 'Remove model control-token delimiters or NUL characters from your text or system prompt.');
      return;
    }
    const user = addMessage('user', prompt);
    const message = addMessage('assistant', 'Waiting for the server…');
    message.article.classList.add('pending');
    message.badge.textContent = 'Submitted';
    const job = { controller: new AbortController(), text: '', finished: false, timing: null, usage: null,
      model: state.status.model, mode: state.status.mode, message, user, prompt, reader: null };
    state.active = job;
    $('prompt').value = '';
    $('notice').hidden = true;
    resetMetrics();
    setRequestState('Submitted · waiting for server', 'busy');
    updateControls();
    followStream = true;
    moveToLatest(true);
    announce('Request submitted. You can stop at any time.');
    try {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST', headers: { ...headers(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: job.model, messages, max_tokens: budget, stream: true,
          stream_options: { include_usage: true }, temperature: 0 }),
        signal: job.controller.signal, cache: 'no-store', credentials: 'omit', redirect: 'error',
      });
      if (!response.ok) throw await responseIssue(response);
      await consumeStream(response, job);
      if (state.active !== job) return;
      // Commit the pair only after both a finish chunk and [DONE]. Failed or
      // cancelled users/partial assistants never contaminate subsequent turns.
      state.history.push({ role: 'user', content: prompt }, { role: 'assistant', content: job.text });
      state.conversationModel = job.model;
      message.badge.textContent = job.finishReason === 'length' ? 'Output limit reached' : job.mode === 'demo' ? 'Scripted demo · complete' : 'Complete';
      message.article.classList.remove('pending');
      addCopy(message, job.text);
      showMetrics(job);
      setRequestState(job.finishReason === 'length' ? 'Finished · output limit' : 'Response complete', 'ready');
      announce(job.finishReason === 'length' ? 'Response finished at the output limit. You can ask Breeze to continue.' : 'Response complete. Copy response is available.');
    } catch (error) {
      if (state.active !== job) return;
      const cancelled = job.controller.signal.aborted;
      const explanation = cancelled ? 'Stopped. The partial answer is not included in future context.' :
        error instanceof ServiceIssue && error.code !== 'incomplete_stream' ? error.message :
          'The response ended before it was complete. Partial text has not been added to conversation context. Check your connection and retry manually.';
      message.article.classList.remove('pending');
      message.article.classList.add('incomplete');
      message.badge.textContent = cancelled ? 'Stopped · not in context' : 'Incomplete · not in context';
      user.badge.textContent = 'Not in context';
      message.body.textContent = job.text || (cancelled ? 'No response was received before stopping.' : 'No complete response was received.');
      addCopy(message, job.text, true);
      const retry = document.createElement('button');
      retry.className = 'message-copy';
      retry.type = 'button';
      retry.textContent = 'Edit & retry';
      retry.addEventListener('click', () => setDraft(prompt));
      message.heading.append(retry);
      setRequestState(cancelled ? 'Stopped · partial excluded' : 'Request not completed', 'error');
      notify(cancelled ? 'Request stopped' : 'Request interrupted', explanation);
      if (error instanceof ServiceIssue && error.status === 401) {
        state.connection = 'locked';
        state.status = null;
        state.lockedPrompted = true;
        showConnect();
      }
    } finally {
      if (state.active === job) {
        state.active = null;
        updateControls();
        moveToLatest();
        void refreshStatus();
      }
    }
  }

  function abortActive() {
    const job = state.active;
    if (!job) return;
    job.controller.abort();
    if (job.reader) void job.reader.cancel().catch(() => {});
  }

  function clearConversation() {
    abortActive();
    state.active = null;
    state.history = [];
    state.conversationModel = null;
    $('messages').replaceChildren();
    $('prompt').value = '';
    $('welcome').hidden = false;
    $('jump-latest').hidden = true;
    $('notice').hidden = true;
    followStream = true;
    resetMetrics();
    setRequestState(ready() ? 'Ready when you are' : 'Connect to continue', ready() ? 'ready' : '');
    updateControls();
  }

  function disconnect() {
    state.epoch += 1;
    state.disconnected = true;
    state.statusRequest?.abort();
    state.statusRequest = null;
    state.schemaRequest?.abort();
    state.schemaRequest = null;
    state.key = '';
    state.status = null;
    state.connection = 'disconnected';
    state.connecting = false;
    state.lockedPrompted = false;
    $('api-key').value = '';
    $('system-prompt').value = '';
    $('max-tokens').value = '256';
    $('settings-summary').textContent = 'System prompt & output budget';
    $('schema-output').textContent = '';
    $('schema-dialog').close();
    $('connect-dialog').close();
    clearConversation();
    paintConnection();
    announce('Disconnected. API key and conversation cleared from this page.');
  }

  async function loadSchema() {
    if (!ready() || state.active || state.schemaRequest) return;
    const controller = new AbortController();
    const epoch = state.epoch;
    state.schemaRequest = controller;
    updateControls();
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
      const response = await fetch('/api/schema', { headers: headers(), signal: controller.signal, cache: 'no-store', credentials: 'omit', redirect: 'error' });
      if (!response.ok) throw await responseIssue(response);
      const schema = await response.json();
      if (state.epoch !== epoch) return;
      $('schema-output').textContent = JSON.stringify(schema, null, 2);
      $('schema-dialog').showModal();
      announce('API schema loaded.');
    } catch (error) {
      if (state.epoch !== epoch) return;
      notify('Schema unavailable', error instanceof ServiceIssue ? error.message : 'Could not load the schema. Check the connection and try again.');
    } finally {
      clearTimeout(timeout);
      if (state.schemaRequest === controller) state.schemaRequest = null;
      updateControls();
    }
  }

  function changeView() {
    const requested = location.hash.slice(1);
    const name = ['workbench', 'connection', 'setup'].includes(requested) ? requested : 'workbench';
    for (const view of ['workbench', 'connection', 'setup']) $('view-' + view).hidden = view !== name;
    document.querySelectorAll('[data-view]').forEach((link) => {
      if (link.dataset.view === name) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    const title = { workbench: 'Workbench', connection: 'Connection & API', setup: 'Setup guide' }[name];
    $('view-title').textContent = title;
    document.title = `Breeze · ${title}`;
  }

  $('service-origin').textContent = location.origin;
  $('chat-form').addEventListener('submit', sendMessage);
  $('prompt').addEventListener('input', updateControls);
  $('prompt').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
      event.preventDefault();
      if (!state.active) void sendMessage();
    }
  });
  $('stop').addEventListener('click', () => {
    abortActive();
    setRequestState('Stopping request…', 'busy');
    updateControls();
  });
  $('new-conversation').addEventListener('click', () => {
    if (($('messages').childElementCount || $('prompt').value.trim()) && !confirm('Start a new conversation? This clears this chat and draft, and stops any active response.')) return;
    clearConversation();
    $('prompt').focus();
    announce('New conversation. Response settings are unchanged.');
  });
  document.querySelectorAll('[data-template]').forEach((button) => {
    button.addEventListener('click', () => { setDraft(TEMPLATES[button.dataset.template]); announce('Sample text added to the composer. Edit it, then press Send.'); });
  });
  for (const id of ['sidebar-connect', 'connection-pill', 'manage-key', 'setup-connect']) $(id).addEventListener('click', showConnect);
  for (const id of ['disconnect', 'dialog-disconnect']) $(id).addEventListener('click', disconnect);
  $('close-connect').addEventListener('click', () => $('connect-dialog').close());
  $('connect-dialog').addEventListener('close', () => { $('api-key').value = ''; });
  $('connect-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (state.active || state.connecting) return;
    const key = $('api-key').value.trim();
    $('api-key').value = '';
    if (key && !/^[\x21-\x7E]+$/.test(key)) {
      $('connect-error').textContent = 'Use the ASCII API key configured on your server, without spaces.';
      $('connect-error').hidden = false;
      return;
    }
    state.connecting = true;
    const epoch = state.epoch;
    $('connect-error').hidden = true;
    updateControls();
    await refreshStatus({ interactive: true, candidateKey: key, connect: true });
    if (state.epoch === epoch) state.connecting = false;
    updateControls();
  });
  $('refresh-status').addEventListener('click', () => { void refreshStatus({ interactive: true }); });
  $('dismiss-notice').addEventListener('click', () => { $('notice').hidden = true; });
  $('load-schema').addEventListener('click', loadSchema);
  $('close-schema').addEventListener('click', () => $('schema-dialog').close());
  $('copy-example').addEventListener('click', () => { if (!state.active && ready()) void copyText($('api-example').textContent, $('copy-example')); });
  for (const id of ['max-tokens', 'system-prompt']) $(id).addEventListener('input', () => {
    $('settings-summary').textContent = `${$('system-prompt').value.trim() ? 'Custom system prompt' : 'No system prompt'} · ${$('max-tokens').value || '—'} max tokens`;
    updateExample();
  });
  scroll.addEventListener('scroll', () => {
    followStream = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 70;
    $('jump-latest').hidden = followStream || !$('messages').childElementCount;
  }, { passive: true });
  $('jump-latest').addEventListener('click', () => { followStream = true; moveToLatest(true); });
  window.addEventListener('hashchange', changeView);
  window.addEventListener('online', () => { if (!state.disconnected) void refreshStatus(); });
  // Page navigation must not leave a queued/generating request running. Never
  // replay a completion on reconnection, visibility changes, or status polling.
  window.addEventListener('pagehide', disconnect);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && !state.disconnected) void refreshStatus();
  });
  setInterval(() => {
    if (!document.hidden && !state.disconnected && !state.connecting && state.connection !== 'locked') void refreshStatus();
  }, 5000);
  changeView();
  paintConnection();
  void refreshStatus();
}

if (typeof document !== 'undefined') startWorkspace();