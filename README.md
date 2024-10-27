# Self-Play LLM for 20 Questions Game
## Tech Stack
- Langgraph: State Graph for LLM Agents
- LLM: OpenAI


## Agent Architecture
### Workflow
![architecture](agent.png)

The game starts with the host generating a topic. The player and host take turns to ask questions and answer questions. The game ends when the player guesses the topic correctly or the number of questions reaches the limit.

### Tools
#### Host
- `generate_topic`: generate a topic for the game
- `answer_question`: answer 'Yes' or 'No' to the question asked by the player
- `check_guess`: check if the guess is correct

#### Player
- `generate_question`: ask a question to the host
- `make_guess`: guess the topic

## Quick Start
1. Create virtual environment and install the dependencies
```
pip install -r requirements.txt
```

2. Create an OpenAI key and put it in the `.env` file
```
OPENAI_API_KEY = sk-...
```

3. Run the code using `python agent.py`


## Test
1. Unit Test for agents
```
python test/unit_test.py
```

2. Parallel Test for multiple games
```
python test/parallel_test.py
```

## TO-DO
- [ ] Logging: improve logging for better debugging, analysis and performance tracking including the prompts and workflow details. Also should have summary report for the test results.
- [ ] Test: add tests to detect if the host and player applies the wrong tools or not use the tools. Consider use behavirour pattern to test the agents.
- [ ] Front-end: add UI to display the game
- [ ] Agent Issues: the player agent repeats same questions and same guesses. Considering introducing a list of guessed topics to avoid repeating guesses. 
- [ ] Code Refactoring: use configuration file for the prompts and other parameters such as LLM. Reduce hard codes and multiple returns, and improve the code readability.