# games-agent
20 Questions Game with LLM-Agent


# Workflow
1. The `hostAgent` needs to come up with a 'topic' of the game and he only answers "YES" or "NO" to the questions asked by the guesserAgent. The topic is an object or a living thing that the guesserAgent needs to guess. The chosen item should be common enough that most people would know it, but not too obvious.
2. The `guesserAgent` asks up to 20 questions (YES or No type questions) to guess the object, he can have multiple attempts until 20 questions have been asked.
3. The `game` is ended when the `guesserAgent` guesses the object or when 20 questions have been asked.