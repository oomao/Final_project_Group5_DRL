## 1. Update 01-startup.sh

- [ ] 1.1 Add `openspec init` step between git pull and handover read
- [ ] 1.2 Add fallback warning if `openspec` is not in PATH
- [ ] 1.3 Update step counter from [1/3] → [1/4], [2/4], [3/4], [4/4]

## 2. Verify 02-ending.sh

- [ ] 2.1 Confirm tasks.md check and auto-create is present
- [ ] 2.2 Confirm `openspec archive` hint is present
- [ ] 2.3 Confirm handover auto-increment writes correct NN- number
- [ ] 2.4 Confirm git add / commit / push runs at the end

## 3. Smoke Test

- [ ] 3.1 Run `npm run dev:start` and verify all 4 steps execute
- [ ] 3.2 Run `npm run dev:ending` and verify handover is created and pushed
