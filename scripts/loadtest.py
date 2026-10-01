"""Bounded-concurrency HTTP load probe. Use a preseeded staging service only."""
import argparse
import asyncio
import json
import os
import time

import httpx


async def run(args):
    queue = asyncio.Queue()
    for index in range(args.requests):
        queue.put_nowait(index)
    latencies, statuses = [], {}
    async with httpx.AsyncClient(base_url=args.url, timeout=5,
                                headers={"Authorization": "Bearer " + os.environ["KWONREC_API_KEY"]}) as client:
        async def worker():
            while not queue.empty():
                index = queue.get_nowait()
                start = time.perf_counter()
                try:
                    response = await client.get(f"/recommend/loadtest-{index % args.users}", params={"limit": 20})
                    label = str(response.status_code)
                    if response.status_code == 200 and not response.json()["recommendations"]:
                        label = "200-empty"
                except httpx.HTTPError:
                    label = "network-error"
                latencies.append((time.perf_counter() - start) * 1000)
                statuses[label] = statuses.get(label, 0) + 1
                queue.task_done()
        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed = time.perf_counter() - start
    latencies.sort()
    def quantile(q):
        return latencies[min(len(latencies) - 1, int(len(latencies) * q))]
    print(json.dumps({"requests": args.requests, "rps": args.requests / elapsed,
                      "p50_ms": quantile(.5), "p95_ms": quantile(.95), "p99_ms": quantile(.99),
                      "statuses": statuses}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--users", type=int, default=1000)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.users) < 1 or args.concurrency > 500:
        parser.error("positive arguments required; concurrency must be <=500")
    asyncio.run(run(args))
