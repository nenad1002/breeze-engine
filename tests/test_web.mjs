import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('../breeze/web/app.js', import.meta.url), 'utf8');
const { SSEParser, validateBudget, buildMessages } = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`,
);
const wire = ': heartbeat\r\ndata: {"choices":[{"delta":{"content":"Hello café 🌿"}}]}\r\n\r\n'
  + 'data: {"choices":[{"delta":{"content":"\\ncontinued"}}]}\n\n'
  + 'data: [DONE]\n\n';
const bytes = new TextEncoder().encode(wire);
const reference = [];
new SSEParser((value) => reference.push(value)).push(wire);
assert.equal(JSON.parse(reference[0]).choices[0].delta.content, 'Hello café 🌿');
assert.equal(JSON.parse(reference[1]).choices[0].delta.content, '\ncontinued');
assert.equal(reference[2], '[DONE]');

for (let split = 0; split <= bytes.length; split++) {
  test(`SSE UTF-8 byte split ${split}`, () => {
    const actual = [];
    const parser = new SSEParser((value) => actual.push(value));
    const decoder = new TextDecoder('utf-8', { fatal: true });
    parser.push(decoder.decode(bytes.slice(0, split), { stream: true }));
    parser.push(decoder.decode(bytes.slice(split), { stream: true }));
    parser.push(decoder.decode());
    parser.end();
    assert.deepEqual(actual, reference);
  });
}

test('SSE one byte at a time and DONE stops parsing', () => {
  const values = [];
  const parser = new SSEParser((value) => { values.push(value); return value !== '[DONE]'; });
  const decoder = new TextDecoder();
  for (const byte of bytes) parser.push(decoder.decode(Uint8Array.of(byte), { stream: true }));
  parser.push(decoder.decode());
  parser.push('data: ignored\n\n');
  parser.end();
  assert.deepEqual(values, reference);
});

test('SSE comments, multiline data and CR endings', () => {
  const values = [];
  const parser = new SSEParser((value) => values.push(value));
  parser.push(': note\rdata: first\rdata: second\r\rdata: third');
  parser.end();
  assert.deepEqual(values, ['first\nsecond', 'third']);
});

test('budget validation and context assembly are explicit', () => {
  for (const value of ['0', '-1', '1.2', '1e2', 'true', '', ' 12', '257']) {
    assert.equal(validateBudget(value, 256), null);
  }
  assert.equal(validateBudget('128', 256), 128);
  const history = [{ role: 'user', content: 'First' }, { role: 'assistant', content: 'Answer' }];
  assert.deepEqual(buildMessages(' Be concise ', history, 'Next'), [
    { role: 'system', content: 'Be concise' }, ...history, { role: 'user', content: 'Next' },
  ]);
  assert.equal(history.length, 2);
});

test('workspace avoids persistent secrets and unsafe message rendering', () => {
  assert.doesNotMatch(source, /\b(?:localStorage|sessionStorage)\b|\.innerHTML\s*=|document\.cookie/);
  assert.match(source, /body\.textContent = content/);
  assert.match(source, /credentials: 'omit'/);
});
