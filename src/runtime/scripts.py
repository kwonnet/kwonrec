"""Every script uses one entity's Redis hash slot. Scripts do bounded work."""
POST = r'''
local old = redis.call('GET', KEYS[1])
if old then
  local version = tonumber(cjson.decode(old).version)
  if version > tonumber(ARGV[1]) then return 0 end
  if version == tonumber(ARGV[1]) and old ~= ARGV[2] then return -1 end
end
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
return 1
'''

INDEX = r'''
redis.call('ZADD', KEYS[1], ARGV[1], ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[3])
redis.call('ZREMRANGEBYRANK', KEYS[1], 0, -tonumber(ARGV[4])-1)
redis.call('EXPIRE', KEYS[1], ARGV[5])
return 1
'''

USER_EVENT = r'''
local receipt = redis.call('GET', KEYS[5])
if receipt then return receipt end
local now, ts, weight = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local post, ttl = ARGV[4], tonumber(ARGV[5])
local cutoff = now - ttl
for _, k in ipairs({KEYS[2], KEYS[3], KEYS[4]}) do
  redis.call('ZREMRANGEBYSCORE', k, '-inf', cutoff)
end
-- Impressions suppress repeats, but are not positive training labels.
redis.call('ZADD', KEYS[3], 'GT', ts, post)
redis.call('ZREMRANGEBYRANK', KEYS[3], 0, -5001)
if weight < 0 then
  redis.call('ZADD', KEYS[4], 'GT', ts, post)
  redis.call('ZREM', KEYS[2], post)
end
if redis.call('ZSCORE', KEYS[4], post) and weight > 0 then weight = 0 end
-- Limit repeated signals from the same user/post/type to one per day.
if weight ~= 0 and not redis.call('SET', KEYS[6], '1', 'NX', 'EX', 86400) then weight = 0 end
local history = redis.call('ZREVRANGE', KEYS[2], 0, 19)
local previous = redis.call('ZSCORE', KEYS[2], post)
local edge = 0
if weight > 0 and not previous then edge = 1 end
if weight > 0 then
  redis.call('ZADD', KEYS[2], 'GT', ts, post)
  redis.call('ZREMRANGEBYRANK', KEYS[2], 0, -51)
end
if weight ~= 0 then
  local last = tonumber(redis.call('GET', KEYS[7]) or now)
  local decay = math.exp(-math.max(0, now-last) / 604800)
  local values = redis.call('HGETALL', KEYS[1])
  local profile = {}
  for i=1,#values,2 do profile[values[i]] = tonumber(values[i+1]) * decay end
  local incoming = cjson.decode(ARGV[6])
  local age_decay = math.exp(-math.max(0, now-ts) / 604800)
  for k,v in pairs(incoming) do profile[k] = (profile[k] or 0) + weight*v*age_decay end
  local ordered = {}
  for k,v in pairs(profile) do table.insert(ordered, {k, v}) end
  table.sort(ordered, function(a,b) return math.abs(a[2]) > math.abs(b[2]) end)
  redis.call('DEL', KEYS[1])
  for i=1,math.min(128,#ordered) do redis.call('HSET', KEYS[1], ordered[i][1], ordered[i][2]) end
  redis.call('SET', KEYS[7], now, 'EX', ttl)
end
redis.call('ZREMRANGEBYRANK', KEYS[4], 0, -5001)
for _, k in ipairs({KEYS[1],KEYS[2],KEYS[3],KEYS[4]}) do redis.call('EXPIRE', k, ttl) end
local result = cjson.encode({history=history, weight=weight, edge=edge, fingerprint=ARGV[8]})
redis.call('SET', KEYS[5], result, 'EX', ARGV[7])
return result
'''

ITEM_EVENT = r'''
local saved = redis.call('GET', KEYS[3])
if saved then return saved end
local now, weight = tonumber(ARGV[1]), tonumber(ARGV[2])
local previous = redis.call('HMGET', KEYS[1], 'score', 'updated')
local score = tonumber(previous[1] or 0) * math.exp(-math.max(0,now-tonumber(previous[2] or now))/86400)
score = math.max(0, score + weight)
redis.call('HSET', KEYS[1], 'score', score, 'updated', now)
local neighbors = cjson.decode(ARGV[3])
for _,id in ipairs(neighbors) do redis.call('ZINCRBY', KEYS[2], 1, id) end
redis.call('ZREMRANGEBYRANK', KEYS[2], 0, -201)
redis.call('EXPIRE', KEYS[1], ARGV[4])
redis.call('EXPIRE', KEYS[2], ARGV[4])
redis.call('SET', KEYS[3], tostring(score), 'EX', ARGV[5])
return tostring(score)
'''

RESERVE = r'''
-- Serialize simultaneous feed requests for this user; do not reset seen on refresh.
local selected = {}
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', tonumber(ARGV[2])-tonumber(ARGV[4]))
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', tonumber(ARGV[2])-tonumber(ARGV[4]))
for _,id in ipairs(cjson.decode(ARGV[1])) do
  if not redis.call('ZSCORE', KEYS[1], id) and not redis.call('ZSCORE', KEYS[2], id) then
    redis.call('ZADD', KEYS[1], ARGV[2], id)
    table.insert(selected, id)
    if #selected >= tonumber(ARGV[3]) then break end
  end
end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', tonumber(ARGV[2])-tonumber(ARGV[4]))
redis.call('ZREMRANGEBYRANK', KEYS[1], 0, -5001)
redis.call('EXPIRE', KEYS[1], ARGV[4])
return selected
'''
