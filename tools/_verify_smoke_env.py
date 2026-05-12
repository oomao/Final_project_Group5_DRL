"""Verify the env-path smoke test artifacts. Tools-only helper, not part of the package."""

import hashlib
import json
import sys


def main() -> int:
    smoke = "runs/smoke_env"
    with open(f"{smoke}/config.json") as f:
        cfg = json.load(f)
    with open(f"{smoke}/reward_fn.py", "rb") as f:
        src_bytes = f.read()
    computed_sha = hashlib.sha256(src_bytes).hexdigest()

    print(f"reward_source        : {cfg.get('reward_source')}")
    print(f"reward_fn_sha256     : {cfg.get('reward_fn_sha256')[:16]}...")
    print(f"computed sha-256     : {computed_sha[:16]}...")
    sha_match = cfg.get("reward_fn_sha256") == computed_sha
    print(f"sha match            : {sha_match}")

    with open(f"{smoke}/episodes.jsonl") as f:
        a = [json.loads(line)["return"] for line in f]
    with open("runs/baseline_seed42/episodes.jsonl") as f:
        b = [json.loads(line)["return"] for line in f][:10]
    determ = a == b
    print(f"returns match baseline first 10 : {determ}")

    print(f"\nreward_fn.py contents:")
    print(open(f"{smoke}/reward_fn.py").read())

    return 0 if (sha_match and determ) else 1


if __name__ == "__main__":
    sys.exit(main())
