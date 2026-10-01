// Run from the repository parent with kwonserver dependencies installed:
// node --test kwonrec/integration/kwonserver.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('../../kwonserver/node_modules/typescript');

function load({ response, error, fallback = [] } = {}) {
  const calls = [];
  const prisma = {
    follow: { findMany: async () => [{ followingId: 'friend' }] },
    post: { findMany: async query => { calls.push(['fallback', query]); return fallback; } },
  };
  const axios = { create: options => {
    calls.push(['config', options]);
    return { post: async (...args) => {
      calls.push(['post', ...args]);
      if (error) throw error;
      return response;
    } };
  } };
  const source = fs.readFileSync(path.join(__dirname, '../../kwonserver/src/services/kwonrec.ts'), 'utf8');
  const output = ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true,
  } }).outputText;
  const exports = {};
  vm.runInNewContext(output, {
    exports, process: { env: { KWONREC_API: 'http://private-service:8001', KWONREC_API_KEY: 'secret' } },
    require: name => {
      if (name === 'axios') return axios;
      if (name === '@/db') return prisma;
      throw new Error('Unexpected runtime import: ' + name);
    },
  });
  return { ...exports, calls };
}

test('private credentials, deadline, following context, and bounded limit', async () => {
  const service = load({ response: { data: { recommendations: [{ id: 'recommended', score: .4 }] } } });
  const result = await service.getRecommendationResponse('authenticated-user', 9999);
  assert.equal(result.data.recommendations[0].id, 'recommended');
  assert.equal(service.calls[0][1].timeout, 1500);
  assert.equal(service.calls[0][1].headers.Authorization, 'Bearer secret');
  const sent = service.calls.find(call => call[0] === 'post')[2];
  assert.equal(sent.user_id, 'authenticated-user');
  assert.equal(sent.limit, 100);
  assert.equal(sent.following_author_ids[0], 'friend');
});

test('service timeout returns visibility-filtered chronological fallback', async () => {
  const service = load({ error: new Error('timeout'), fallback: [{ id: 'safe' }] });
  const result = await service.getRecommendationResponse('reader', 10);
  assert.equal(result.data.degraded, true);
  assert.equal(result.data.recommendations[0].id, 'safe');
  const query = service.calls.find(call => call[0] === 'fallback')[1];
  assert.equal(query.take, 10);
  assert.equal(query.where.scope, 'ANYONE');
  assert.equal(query.where.user.isPrivate, false);
  assert.equal(query.where.user.deletedAt, null);
  assert.equal(query.where.reports.none.userId, 'reader');
  assert.equal(query.where.disinterest.none.userId, 'reader');
  assert.equal(query.where.AND.length, 2);
});

test('invalid upstream payload falls back; invalid limits remain bounded', async () => {
  const service = load({ response: { data: { recommendations: [{ id: 123 }] } } });
  const result = await service.getRecommendationResponse('reader', 'NaN');
  assert.equal(result.data.degraded, true);
  assert.equal(service.calls.find(call => call[0] === 'fallback')[1].take, 21);
});

test('legitimate empty feed stays empty without repeatedly recycling old posts', async () => {
  const service = load({ response: { data: { recommendations: [] } } });
  const result = await service.getRecommendationResponse('reader', 20);
  assert.equal(result.data.recommendations.length, 0);
  assert.equal(service.calls.some(call => call[0] === 'fallback'), false);
});

for (const degraded of [false, true]) {
  test(`newsfeed preserves ranked IDs, authenticated identity and source header (fallback=${degraded})`, async () => {
    const calls = [];
    const source = fs.readFileSync(path.join(__dirname, '../../kwonserver/src/controllers/v1/posts/index.ts'), 'utf8');
    const output = ts.transpileModule(source, { compilerOptions: {
      module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true,
    } }).outputText;
    const exports = {};
    const posts = [{ id: 'second' }, { id: 'first' }];
    vm.runInNewContext(output, { exports, console: { log() {} }, require: name => {
      if (name === '@/utils') return { validateZodInput: () => ({ data: { limit: 2, page: 1 } }) };
      if (name === '@/services/kwonrec') return { getRecommendationResponse: async (user, limit) => {
        calls.push({ user, limit });
        return { data: { recommendations: posts, degraded } };
      } };
      if (name === '@/services/v1/posts') return { getNewsfeed: async (ids, user) => {
        assert.deepEqual(Array.from(ids), ['second', 'first']);
        assert.equal(user.id, 'logged-in');
        return { status: 200, data: posts };
      } };
      return {};
    } });
    const headers = {};
    const res = { setHeader: (k, v) => { headers[k] = v; }, status(code) {
      assert.equal(code, 200); return this;
    }, send(body) { assert.equal(body, posts); return this; } };
    await exports.getNewsfeedController({ user: { id: 'logged-in' }, params: { feedType: 'for-you' },
      query: { user_id: 'attacker-supplied', limit: '2' } }, res);
    assert.deepEqual(calls, [{ user: 'logged-in', limit: 2 }]);
    assert.equal(headers['Cache-Control'], 'private, no-store');
    assert.equal(headers['X-Feed-Source'], degraded ? 'fallback' : 'kwonrec');
  });
}
