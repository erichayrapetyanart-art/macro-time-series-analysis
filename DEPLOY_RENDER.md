# Deploying The Dashboard To Render

This project deploys as a Render **Web Service** running the Streamlit dashboard.

## Files Render Uses

- `render.yaml`: Render Blueprint configuration.
- `.python-version`: pins Python to `3.11.9`.
- `requirements.txt`: Python dependencies.
- `dashboard_app.py`: Streamlit entry point.
- `data/*.csv`: local project input data used by the dashboard.
- `outputs/tables/*.csv` and `outputs/tables/*.txt`: precomputed model results used by the dashboard.

The dashboard does not need `FRED_API_KEY` at runtime because it reads the already generated local CSV data and output tables. Add `FRED_API_KEY` on Render only if you want to regenerate data/results in the cloud.

## Render Setup

Recommended path:

1. Push this repository to GitHub.
2. Open Render Dashboard.
3. Choose **New > Blueprint**.
4. Select this repository.
5. Render will read `render.yaml` and create the service.

Manual Web Service setup also works:

- Runtime: `Python`
- Build command: `pip install -r requirements.txt`
- Start command:

```bash
streamlit run dashboard_app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

## Important Git Step

Before deploying, make sure the deployment artifacts are committed:

```bash
git add render.yaml .python-version .streamlit/config.toml DEPLOY_RENDER.md .gitignore
git add data/*.csv outputs/tables/*.csv outputs/tables/*.txt
git commit -m "Configure Render deployment"
git push
```

Do not commit `.venv/`, `.env`, or generated presentation scratch folders.

## Notes

- Render web services must bind to `0.0.0.0` and the `PORT` environment variable.
- The free Render plan can sleep after inactivity, so the first load may be slow.
- The dashboard supports real-time VAR/VARX refitting, but heavy specifications can be slow on the free plan.
- If deployment fails with missing CSV files, check that `data/*.csv` and `outputs/tables/*` were committed.
