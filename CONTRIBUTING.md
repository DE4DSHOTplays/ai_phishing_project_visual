# Contributing to Phishing URL Detector — Visual Edition

Thank you for wanting to contribute! This document explains the easiest way to
get started and how to make high-quality contributions that will be accepted.

1) Report issues
- Use GitHub Issues for bug reports, feature requests, or questions.
- Provide a clear title and steps to reproduce (include sample URL(s), expected
  behaviour, and actual behaviour). Attach screenshots where useful.

2) Discuss major changes first
- Open an issue for larger changes (new features, model/data changes) and
  discuss the design before implementing. This avoids wasted work.

3) Create a branch
- Branch from `main` using a descriptive name:

```bash
git checkout -b feat/add-example-feature
```

4) Make small, focused commits
- Keep commits small and focused; use clear commit messages.
- Suggested format: `type(scope): short description` e.g.
  `fix(app): validate uploaded CSV columns`

5) Code style and tests
- Follow existing Python style in the repo (PEP 8). Use meaningful variable names
  and avoid one-letter names.
- If you add functionality, include a short unit test or a manual test note
  describing how to exercise the change.

6) Pull Request process
- Open a PR from your branch to `main` and include the issue number (if any).
- In the PR description, include what you changed, why, and how to test it.
- Keep PRs focused — large PRs may be split into smaller, reviewable chunks.

7) Review and approvals
- PRs will be reviewed for correctness, clarity, and style. Be responsive to
  feedback and iterate until reviewers approve.

8) Data, models, and sensitive information
- Do NOT commit sensitive data, API keys, or personal information. Use an
  external storage (S3, Drive) or GitHub Releases for large model artifacts.
- If you need to share an example dataset, include a small sanitized sample.

9) Running and testing locally
- To run the Streamlit app locally:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app_visual.py
```

10) Contact and code of conduct
- If you have questions, open an issue or contact the repository owner.
- Treat contributors respectfully. If you'd like, I can add a short
  `CODE_OF_CONDUCT.md` template as well.

Thank you for helping improve this project — contributions large and small
are welcome!
