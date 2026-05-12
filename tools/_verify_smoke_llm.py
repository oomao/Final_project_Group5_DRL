"""Verify the llm-path smoke test artifacts."""

import hashlib
import json
import sys


def main() -> int:
    smoke = "runs/smoke_llm"
    with open(f"{smoke}/config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(f"{smoke}/reward_fn.py", "rb") as f:
        src_bytes = f.read()
    computed_sha = hashlib.sha256(src_bytes).hexdigest()

    print(f"reward_source        : {cfg.get('reward_source')}")
    print(f"sha in config        : {cfg.get('reward_fn_sha256')[:16]}...")
    print(f"computed sha-256     : {computed_sha[:16]}...")
    sha_match = cfg.get("reward_fn_sha256") == computed_sha
    print(f"sha match            : {sha_match}")

    with open(f"{smoke}/llm_attempts.jsonl", encoding="utf-8") as f:
        attempts = [json.loads(line) for line in f]
    print(f"\nllm_attempts.jsonl   : {len(attempts)} line(s)")
    for a in attempts:
        print(f"  attempt {a['attempt']}: accepted={a['accepted']} error={a['error']}")

    # The last attempt should be the accepted one
    last_accepted = attempts[-1]["accepted"]
    print(f"\nfinal attempt accepted : {last_accepted}")

    # Check reward_fn.py source matches last accepted response's code block
    import re

    last_response = attempts[-1]["response"]
    py_match = re.search(r"```python\s*\n(.*?)```", last_response, re.DOTALL)
    extracted = py_match.group(1).strip() + "\n" if py_match else ""
    on_disk = src_bytes.decode("utf-8")
    matches = extracted == on_disk
    print(f"reward_fn.py matches last accepted response : {matches}")

    with open(f"{smoke}/episodes.jsonl", encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f]
    print(f"\nepisodes.jsonl       : {len(episodes)} rows")
    print(f"first return         : {episodes[0]['return']:.1f}")
    print(f"last return          : {episodes[-1]['return']:.1f}")

    all_good = sha_match and last_accepted and matches and len(episodes) == 10
    print(f"\nall checks pass      : {all_good}")
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
