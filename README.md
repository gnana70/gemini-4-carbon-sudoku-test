Video: https://youtu.be/1nRiEC1B2f4

# Gemini 4 Carbon vs Argon: one hard Sudoku, five tries each

The test from the Singularity Feed video "Gemini 4 Carbon Is Out: I Tested It Against Argon (Same Price)" (10 Nov 2026).

```bash
pip install requests
export GEMINI_API_KEY=...            # paid tier: both models are paid-only
python sudoku_test.py make --seed 7  # writes puzzle.json (unique solution); the video used seed 7 (24 clues)
python sudoku_test.py submit --runs 5 --level high   # starts 10 background Interactions (no connection held open)
python sudoku_test.py poll           # rerun until "pending 0"; then it prints the summary
```

- Same prompt for both models (`PROMPT` in the script), `thinking_level: high`. Each answer is parsed from an `ANSWER ... END`
  block and checked by code: it must keep every given digit and break no row, column or 3x3 box (with a unique puzzle, that means
  it equals the solution).
- Time = submit to the server's `updated` timestamp on completion. Cost = reported input tokens x $2/1M + (output + thinking
  tokens) x $10/1M, Google's introductory rate (until 30 Nov 2026; $4 / $20 after).
- A single try took 6-14 minutes and about 40-60K thinking tokens. Ten tries cost about $5.60.

## Results (10 Nov 2026, `results/`)
| | correct | minutes per try | median | thinking tokens (avg) | cost per try (avg) |
|---|---|---|---|---|---|
| gemini-4-carbon | 5/5 | 6.3, 6.5, 6.6, 8.1, 9.6 | 6.6 | 46,857 | $0.56 |
| gemini-4-argon | 5/5 | 6.4, 6.7, 8.2, 9.2, 13.5 | 8.2 | 47,960 | $0.57 |

Five tries each is a small sample, and one self-contained puzzle doesn't test long multi-step agent work, which is what Google
says Carbon is for.
