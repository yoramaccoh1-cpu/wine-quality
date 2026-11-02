# AI Usage Declaration

We used Claude (Anthropic) as a helper throughout the project. We declare every
use below. We ran all the code ourselves, and every number in the notebook
comes from our own runs.

| Tool | Used for | What we changed or checked |
|------|----------|----------------------------|
| Claude | Step-by-step explanations of the course notebooks (S1, S3, Lab 1) and of the brief | We re-explained each step in our own words and followed the course protocol. |
| Claude | Suggested drafts of tests (`tests/test_data.py`, `tests/test_pipeline.py`) and of the command-line tool (`cli.py`) | We typed and ran them, simplified the pipeline tests, added type hints and docstrings, and checked that all 8 tests pass. |
| Claude | Suggested wording and structure for the notebook (framing, section explanations, interpretation) | We rewrote the text with our own results, fixed the numbers after the final run and changed the threshold method to out-of-fold predictions. 
| Claude | Explaining errors: Git merge and push conflicts, `pip` and build-backend errors, ruff warnings | We applied the fixes ourselves and verified the result with `git status`, `pytest` and `ruff`. |
| ChatGPT (OpenAI) | Final project audit checking completeness against the brief | We reviewed the flagged issues and applied only the ones that were legitimate fixes |

We can explain every line we submit, and we ran the code from top to bottom before submitting.
