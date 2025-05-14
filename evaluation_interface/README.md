# Evaluation Interface

The evaluation interface (in Portuguese) was created using [Streamlit](https://streamlit.io/), using python 3.9.

## How to run

To run the interface, run:

```bash
streamlit run ./1_🏠_Início.py
```

## Configuration

The paths and annotator splits configuration is done in the `./config/config.yaml` file with the following structure:

```yaml
paths:
    data: "../data/evaluation.jsonl" # Path to get data from
    results: "../results/evaluation" # Path to save evaluation results to
splits: # Headline IDs for each annotator to evaluate
    annotator1:
        - 0
        - 1
        - 2
    annotator2:
        - 3
        - 4
        - 5
```

The interface has [Streamlit authentication](https://blog.streamlit.io/streamlit-authenticator-part-1-adding-an-authentication-component-to-your-app/) to know which evaluator is currently working on the task. To configure the credentials, edit the `./config/credentials.yaml` file with the following structure:

```yaml
cookie:
  expiry_days: 30
  key: pun_generation
  name: pun_generation_authentication
credentials:
  usernames:
    annotator1: # Username of the user
      email: null # Required for Stramlit authenticator. Not used
      failed_login_attempts: 0
      logged_in: true
      name: Annotator 1
      password: password1 # A plain text password will be hashed when the system is first run
```
