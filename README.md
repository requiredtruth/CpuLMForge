# CpuLMForge

CpuLMForge selects reproducible CPU-only local-model launch profiles from measurements you actually recorded. It groups repeated JSONL runs by model, threads, context, and batch; calculates median and worst observed throughput; applies RAM and minimum-speed gates; and emits a shell-escaped `llama-server` command.

```bash
python -m cpulmforge examples/samples.jsonl --memory-gib 4 --minimum-tps 10
```

Every recommendation includes its sample count, run IDs, median tokens/second, minimum tokens/second, and maximum observed RSS. Rejected profiles retain exact reasons. The tool does not benchmark hardware, infer unmeasured performance, or claim that past measurements guarantee future speed. Feed it runs collected under controlled conditions.

## Test

`python -m unittest discover -s tests -v`

## Fund more development

Donations increase RequiredTruth development production. See [SUPPORT.md](SUPPORT.md); confirmed donors may claim a transaction hash in an issue and request a specific direction.

Apache-2.0 licensed.
