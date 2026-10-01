// Read-only source database smoke check. Uses a disposable recommendation profile.
// Run: node kwonrec/integration/live-kwonserver.cjs
const path = require('node:path');
const fs = require('node:fs');
const vm = require('node:vm');
const { randomUUID } = require('node:crypto');
const root = path.resolve(__dirname, '../../kwonserver');
require(path.join(root, 'node_modules/dotenv')).config({ path: path.join(root, '.env') });
const ts = require(path.join(root, 'node_modules/typescript'));
const { PrismaClient } = require(path.join(root, 'node_modules/@prisma/client'));
const prisma = new PrismaClient();
const output = ts.transpileModule(fs.readFileSync(path.join(root, 'src/services/kwonrec.ts'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
}).outputText;
const service = {};
vm.runInNewContext(output, { exports: service, process, require: name => {
  if (name === '@/db') return prisma;
  if (name === 'axios') return require(path.join(root, 'node_modules/axios/dist/node/axios.cjs'));
  throw new Error(`Unexpected import ${name}`);
} });
(async () => {
  const user = `connection-check-${randomUUID()}`;
  const response = await service.getRecommendationResponse(user, 5);
  if (response.data.degraded) throw new Error('kwonserver used fallback: recommendation connection failed');
  const ids = response.data.recommendations.map(item => item.id);
  const visible = await prisma.post.count({ where: { ...service.recommendationVisibility(user), id: { in: ids } } });
  console.log(JSON.stringify({ source: 'kwonrec', recommended: ids.length, visibleInPostgres: visible }));
})().catch(error => { console.error(error.name, 'Recommendation smoke check failed'); process.exitCode = 1; })
  .finally(() => prisma.$disconnect());
