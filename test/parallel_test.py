import threading

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import Game
import uuid
import yaml

def run_game(game_id):
    system_prompt = yaml.safe_load(open("system_prompts.yaml"))
    game = Game(system_prompt, game_id)
    game.run()

def run_parallel_games(num_games):
    threads = []

    for i in range(num_games):
        game_id = str(uuid.uuid4())
        thread = threading.Thread(target=run_game, args=(game_id,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All games completed.")

if __name__ == "__main__":
    num_games = 3  # Number of concurrent games to run
    run_parallel_games(num_games)