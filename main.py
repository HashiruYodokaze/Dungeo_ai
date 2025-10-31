import random
import requests
import sounddevice as sd
import numpy as np
import os
import subprocess
import datetime
import time
import json
import traceback
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scenarios import GENRE_DESCRIPTIONS, ROLE_STARTERS

# ===== CONFIGURATION =====
CONFIG = {
    "ALLTALK_ENABLED": False,
    "ALLTALK_API_URL": "http://localhost:7851",
    "LLM_BACKEND": "KoboldCPP",
    "LLM_API_URL": "http://localhost:5001",
    "LOG_FILE": "error_log.txt",
    "SAVE_FILE": "adventure.txt",
    "DEFAULT_MODEL": "llama3:instruct",
    "REQUEST_TIMEOUT": 120,
    "AUDIO_SAMPLE_RATE": 22050,
    "MAX_CONVERSATION_LENGTH": 10000
}

@dataclass
class GameState:
    conversation: str = ""
    last_ai_reply: str = ""
    last_player_input: str = ""
    current_model: str = CONFIG["DEFAULT_MODEL"]
    character_name: str = "Alex"
    selected_genre: str = "Fantasy"
    selected_role: str = "Adventurer"
    adventure_started: bool = False

# ===== GAME DATA =====
DM_SYSTEM_PROMPT = """
You are a masterful Dungeon Master in an unrestricted SFW adventure. Your role is to **always** narrate the consequences of player actions, no matter how extreme, bizarre, violent or unconventional they may be. Follow these rules:

1. ACTION-CONSEQUENCE SYSTEM:
   - Describe ONLY the consequences of the player's action
   - Never perform actions on behalf of the player
   - Consequences must permanently change the game world
   - Narrate consequences naturally within the story flow
   - All actions are allowed and must always have some kind of reaction

2. RESPONSE STYLE:
   - Describe what happens in the world as a result of the player's action
   - Do not describe the player performing actions - the player has already done that
   - Never ask "what do you do next" or similar
   - Always continue the story in a way that feels immersive

3. WORLD EVOLUTION:
   - NPCs remember player choices and react accordingly
   - Environments change permanently based on actions
   - Player choices open/close future narrative paths
   - Resources, factions, and the world evolve with every decision

4. PLAYER AGENCY:
   - The player can attempt **anything**
   - Never block an action—show the results, good or bad
   - Let the world react dynamically and permanently

Never break character as the Dungeon Master. Always continue the adventure.
"""

class AdventureGame:
    def __init__(self):
        self.state = GameState()
        self._audio_lock = threading.Lock()
        self._setup_directories()
        
    def _setup_directories(self):
        """Ensure necessary directories exist"""
        os.makedirs("logs", exist_ok=True)
        os.makedirs("saves", exist_ok=True)
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None) -> None:
        """Enhanced error logging with rotation"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_path = CONFIG["LOG_FILE"]
            
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"\n--- ERROR [{timestamp}] ---\n")
                log_file.write(f"Message: {error_message}\n")
                
                if exception:
                    log_file.write(f"Exception: {type(exception).__name__}: {str(exception)}\n")
                    traceback.print_exc(file=log_file)
                
                log_file.write(f"Game State: Model={self.state.current_model}, ")
                log_file.write(f"Genre={self.state.selected_genre}, Role={self.state.selected_role}\n")
                log_file.write("--- END ERROR ---\n")
                
        except Exception as e:
            print(f"CRITICAL: Failed to write to error log: {e}")

    def check_server(self, url: str, service_name: str) -> bool:
        """Generic server health check"""
        try:
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.log_error(f"{service_name} check failed", e)
            return False

    def check_llm_server(self) -> bool:
        if CONFIG["LLM_BACKEND"] == "KoboldCPP":
            return self.check_server(f"{CONFIG['LLM_API_URL']}/api/v1/model", "KoboldCPP")
        else:
            return self.check_server(f"{CONFIG['LLM_API_URL']}/api/tags", "Ollama")

    def check_alltalk_server(self) -> bool:
        return self.check_server(f"{CONFIG['ALLTALK_API_URL']}/api/ready", "AllTalk")

    def get_installed_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            if not self.check_llm_server():
                return []

            if CONFIG["LLM_BACKEND"] == "KoboldCPP":
                response = requests.get(f"{CONFIG['LLM_API_URL']}/api/v1/model", timeout=5)
                if response.status_code == 200:
                    return [response.json().get("result", "Unknown model")]

            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=30
            )
            
            models = []
            for line in result.stdout.strip().splitlines()[1:]:
                parts = line.split()
                if parts:
                    models.append(parts[0])
            return models
            
        except subprocess.TimeoutExpired:
            self.log_error("Model list command timed out")
            return []
        except Exception as e:
            self.log_error("Error getting installed models", e)
            return []

    def select_model(self) -> str:
        """Interactive model selection with fallback"""
        models = self.get_installed_models()
        
        if not models:
            print("No models found. Please enter a model name.")
            model_input = input(f"Enter model name [{CONFIG['DEFAULT_MODEL']}]: ").strip()
            return model_input or CONFIG["DEFAULT_MODEL"]

        if len(models) == 1:
            return models[0]

        print("\nAvailable models:")
        for idx, model in enumerate(models, 1):
            print(f"  {idx}: {model}")

        while True:
            try:
                choice = input(f"Select model (1-{len(models)}) or Enter for default [{CONFIG['DEFAULT_MODEL']}]: ").strip()
                
                if not choice:
                    return CONFIG["DEFAULT_MODEL"]
                
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    return models[idx]
                else:
                    print(f"Please enter a number between 1 and {len(models)}")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nUsing default model.")
                return CONFIG["DEFAULT_MODEL"]

    def get_ai_response(self, prompt: str) -> str:
        """Get AI response with enhanced error handling and prompt optimization"""
        try:
            # Truncate conversation if it gets too long to maintain performance
            if len(prompt) > CONFIG["MAX_CONVERSATION_LENGTH"]:
                # Keep system prompt and recent conversation
                system_part = DM_SYSTEM_PROMPT
                recent_conversation = prompt[-4000:]  # Keep last 4000 characters
                prompt = system_part + "\n\n[Earlier conversation truncated...]\n" + recent_conversation

            if CONFIG["LLM_BACKEND"] == "KoboldCPP":
                response = requests.post(
                    CONFIG["LLM_API_URL"] + "/api/v1/generate",
                    json = {
                        "prompt": prompt,
                        "max_length": 500,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "typical_p": 1.0,
                        "rep_pen": 1.1,
                        "rep_pen_range": 256,
                        "stop_sequence": ["\n\n", "Player:", "Dungeon Master:"],
                        "use_default_badwordsids": False
                    },
                    timeout=CONFIG["REQUEST_TIMEOUT"]
                )

            else:
                response = requests.post(
                    CONFIG["LLM_API_URL"] + "/api/generate",
                    json = {
                        "model": self.state.current_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "stop": ["\n\n", "Player:", "Dungeon Master:"],
                            "min_p": 0.05,
                            "top_k": 40,
                            "top_p": 0.9,
                            "num_ctx": 4096
                        }
                    },
                    timeout=CONFIG["REQUEST_TIMEOUT"]
                )

            response.raise_for_status()
            if CONFIG["LLM_BACKEND"] == "KoboldCPP":
                result = response.json()
                if "results" in result and len(result["results"]) > 0:
                    return result["results"][0]["text"].strip()
                return ""
            else:
                return response.json().get("response", "").strip()
            
        except requests.exceptions.Timeout:
            self.log_error("AI request timed out")
            return "The world seems to pause as if time has stopped. What would you like to do?"
        except requests.exceptions.ConnectionError:
            self.log_error("Cannot connect to LLM server")
            return ""
        except Exception as e:
            self.log_error("Error getting AI response", e)
            return ""

    def speak(self, text: str, voice: str = "FemaleBritishAccent_WhyLucyWhy_Voice_2.wav") -> None:
        """Non-blocking text-to-speech with improved error handling"""
        if not text.strip():
            return

        def _speak_thread():
            with self._audio_lock:
                try:
                    if not self.check_alltalk_server():
                        print("[TTS Server unavailable]")
                        return

                    payload = {
                        "text_input": text,
                        "character_voice_gen": voice,
                        "narrator_enabled": "true",
                        "narrator_voice_gen": "narrator.wav",
                        "text_filtering": "none",
                        "output_file_name": "output",
                        "autoplay": "true",
                        "autoplay_volume": "0.8"
                    }

                    response = requests.post(CONFIG["ALLTALK_API_URL"], data=payload, timeout=30)
                    response.raise_for_status()

                    content_type = response.headers.get("Content-Type", "")

                    if content_type.startswith("audio/"):
                        audio_data = np.frombuffer(response.content, dtype=np.int16)
                        sd.play(audio_data, samplerate=CONFIG["AUDIO_SAMPLE_RATE"])
                        sd.wait()
                    elif content_type.startswith("application/json"):
                        error_data = response.json()
                        self.log_error(f"AllTalk API error: {error_data.get('error', 'Unknown error')}")
                    else:
                        self.log_error(f"Unexpected AllTalk response type: {content_type}")

                except Exception as e:
                    self.log_error("Error in TTS", e)

        # Start TTS in background thread
        thread = threading.Thread(target=_speak_thread, daemon=True)
        thread.start()

    def show_help(self) -> None:
        """Display available commands"""
        print("""
Available commands:
/? or /help       - Show this help message
/redo             - Repeat last AI response with a new generation
/save             - Save the full adventure to adventure.txt
/load             - Load the adventure from adventure.txt
/change           - Switch to a different Ollama model
/status           - Show current game status
/exit             - Exit the game
""")

    def show_status(self) -> None:
        """Display current game status"""
        print(f"\n--- Current Game Status ---")
        print(f"Character: {self.state.character_name} the {self.state.selected_role}")
        print(f"Genre: {self.state.selected_genre}")
        print(f"Model: {self.state.current_model}")
        print(f"Adventure: {'Started' if self.state.adventure_started else 'Not started'}")
        if self.state.last_ai_reply:
            print(f"Last action: {self.state.last_player_input[:50]}...")
        print("---------------------------")

    def remove_last_ai_response(self) -> None:
        """Remove the last AI response from conversation"""
        pos = self.state.conversation.rfind("Dungeon Master:")
        if pos != -1:
            self.state.conversation = self.state.conversation[:pos].strip()

    def save_adventure(self) -> bool:
        """Save adventure to file with error handling"""
        try:
            save_data = {
                "conversation": self.state.conversation,
                "metadata": {
                    "character_name": self.state.character_name,
                    "genre": self.state.selected_genre,
                    "role": self.state.selected_role,
                    "model": self.state.current_model,
                    "save_time": datetime.datetime.now().isoformat()
                }
            }
            
            with open(CONFIG["SAVE_FILE"], "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print("Adventure saved successfully!")
            return True
            
        except Exception as e:
            self.log_error("Error saving adventure", e)
            print("Failed to save adventure.")
            return False

    def load_adventure(self) -> bool:
        """Load adventure from file with error handling"""
        try:
            if not os.path.exists(CONFIG["SAVE_FILE"]):
                print("No saved adventure found.")
                return False

            with open(CONFIG["SAVE_FILE"], "r", encoding="utf-8") as f:
                save_data = json.load(f)

            self.state.conversation = save_data["conversation"]
            metadata = save_data.get("metadata", {})
            
            self.state.character_name = metadata.get("character_name", "Alex")
            self.state.selected_genre = metadata.get("genre", "Fantasy")
            self.state.selected_role = metadata.get("role", "Adventurer")
            self.state.current_model = metadata.get("model", CONFIG["DEFAULT_MODEL"])
            
            # Extract last AI reply
            last_dm = self.state.conversation.rfind("Dungeon Master:")
            if last_dm != -1:
                self.state.last_ai_reply = self.state.conversation[last_dm + len("Dungeon Master:"):].strip()
            
            self.state.adventure_started = True
            print("Adventure loaded successfully!")
            return True
            
        except Exception as e:
            self.log_error("Error loading adventure", e)
            print("Failed to load adventure.")
            return False

    def select_genre_and_role(self) -> Tuple[str, str]:
        """Interactive genre and role selection"""

        genres = {}
        for index, genre in enumerate(GENRE_DESCRIPTIONS):
            genres[index] = genre

        print("Choose your adventure genre:")
        for key, name in genres.items():
            print(f"{key}: {name}")
        
        while True:
            genre_choice = int(input("Enter the number of your choice: ").strip())
            selected_genre = genres.get(genre_choice)
            if selected_genre:
                break
            print("Invalid choice. Please try again.")

        # Show genre description
        print(f"\n{selected_genre}: {GENRE_DESCRIPTIONS.get(selected_genre, '')}\n")
        
        # Role selection
        roles = list(ROLE_STARTERS[selected_genre].keys())
        print(f"Choose your role in {selected_genre}:")
        for idx, role in enumerate(roles, 1):
            print(f"{idx}: {role}")
        
        while True:
            role_choice = input("Enter the number of your choice (or 'r' for random): ").strip().lower()
            if role_choice == 'r':
                selected_role = random.choice(roles)
                break
            try:
                idx = int(role_choice) - 1
                if 0 <= idx < len(roles):
                    selected_role = roles[idx]
                    break
            except ValueError:
                pass
            print("Invalid choice. Please try again.")

        return selected_genre, selected_role

    def start_new_adventure(self) -> bool:
        """Start a new adventure with character creation"""
        try:
            self.state.selected_genre, self.state.selected_role = self.select_genre_and_role()
            
            self.state.character_name = input("\nEnter your character's name: ").strip() or "Alex"
            
            starter = ROLE_STARTERS[self.state.selected_genre].get(
                self.state.selected_role, 
                "You find yourself in an unexpected situation when"
            )
            
            print(f"\n--- Adventure Start: {self.state.character_name} the {self.state.selected_role} ---")
            print(f"Starting scenario: {starter}")
            print("Type '/?' or '/help' for commands.\n")
            
            # Initial setup
            initial_context = (
                f"### Adventure Setting ###\n"
                f"Genre: {self.state.selected_genre}\n"
                f"Player Character: {self.state.character_name} the {self.state.selected_role}\n"
                f"Starting Scenario: {starter}\n\n"
                "Dungeon Master: "
            )
            
            self.state.conversation = initial_context
            
            # Get first response
            full_prompt = DM_SYSTEM_PROMPT + "\n\n" + self.state.conversation
            ai_reply = self.get_ai_response(full_prompt)
            if ai_reply:
                print(f"Dungeon Master: {ai_reply}")
                if CONFIG["ALLTALK_ENABLED"]:
                    self.speak(ai_reply)
                self.state.conversation += ai_reply
                self.state.last_ai_reply = ai_reply
                self.state.adventure_started = True
                return True
            else:
                print("Failed to get initial response from AI.")
                return False
                
        except Exception as e:
            self.log_error("Error starting new adventure", e)
            return False

    def process_command(self, command: str) -> bool:
        """Process game commands"""
        cmd = command.lower().strip()
        
        if cmd in ["/?", "/help"]:
            self.show_help()
        elif cmd == "/exit":
            print("Exiting the adventure. Goodbye!")
            return False
        elif cmd == "/redo":
            self._handle_redo()
        elif cmd == "/save":
            self.save_adventure()
        elif cmd == "/load":
            self.load_adventure()
        elif cmd == "/change":
            self._handle_model_change()
        elif cmd == "/status":
            self.show_status()
        else:
            print(f"Unknown command: {command}. Type '/help' for available commands.")
        
        return True

    def _handle_redo(self) -> None:
        """Handle the /redo command"""
        if self.state.last_ai_reply and self.state.last_player_input:
            self.remove_last_ai_response()
            full_prompt = (
                f"{DM_SYSTEM_PROMPT}\n\n"
                f"{self.state.conversation}\n"
                f"Player: {self.state.last_player_input}\n"
                "Dungeon Master:"
            )
            new_reply = self.get_ai_response(full_prompt)
            if new_reply:
                print(f"\nDungeon Master: {new_reply}")
                if CONFIG["ALLTALK_ENABLED"]:
                    self.speak(new_reply)
                self.state.conversation += f"\nPlayer: {self.state.last_player_input}\nDungeon Master: {new_reply}"
                self.state.last_ai_reply = new_reply
            else:
                print("Failed to generate new response.")
        else:
            print("Nothing to redo.")

    def _handle_model_change(self) -> None:
        """Handle model change command"""
        models = self.get_installed_models()
        if models:
            print("Available models:")
            for idx, model in enumerate(models, 1):
                print(f"{idx}: {model}")
            
            while True:
                try:
                    choice = input("Enter number of new model: ").strip()
                    if not choice:
                        break
                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        self.state.current_model = models[idx]
                        print(f"Model changed to: {self.state.current_model}")
                        break
                    print("Invalid selection.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            print("No installed models found.")

    def process_player_input(self, user_input: str) -> None:
        """Process regular player input"""
        self.state.last_player_input = user_input
        formatted_input = f"Player: {user_input}"
        
        prompt = (
            f"{DM_SYSTEM_PROMPT}\n\n"
            f"{self.state.conversation}\n"
            f"{formatted_input}\n"
            "Dungeon Master:"
        )
        
        ai_reply = self.get_ai_response(prompt)
        if ai_reply:
            print(f"\nDungeon Master: {ai_reply}")
            if CONFIG["ALLTALK_ENABLED"]:
                self.speak(ai_reply)
            self.state.conversation += f"\n{formatted_input}\nDungeon Master: {ai_reply}"
            self.state.last_ai_reply = ai_reply
            
            # Auto-save every 5 interactions
            if self.state.conversation.count("Player:") % 5 == 0:
                self.save_adventure()
        else:
            print("Failed to get response from AI. Please try again.")

    def run(self) -> None:
        """Main game loop"""
        print("=== AI Dungeon Master Adventure ===\n")
        
        # Server checks
        if not self.check_llm_server():
            print("LLM server not found.")
            print("Waiting for server to start...")
            time.sleep(3)
            if not self.check_llm_server():
                print("LLM server still not running. Please start it and try again.")
                return
        
        # Model selection
        self.state.current_model = self.select_model()
        print(f"Using model: {self.state.current_model}\n")
        
        # Load or start adventure
        if os.path.exists(CONFIG["SAVE_FILE"]):
            print("A saved adventure exists. Load it now? (y/n)")
            if input().strip().lower() == 'y':
                if self.load_adventure():
                    print(f"\nDungeon Master: {self.state.last_ai_reply}")
                    if CONFIG["ALLTALK_ENABLED"]:
                        self.speak(self.state.last_ai_reply)
        
        if not self.state.adventure_started:
            if not self.start_new_adventure():
                return
        
        # Main game loop
        while self.state.adventure_started:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if not self.process_command(user_input):
                        break
                else:
                    self.process_player_input(user_input)
                    
            except KeyboardInterrupt:
                print("\n\nGame interrupted. Use '/exit' to quit properly.")
            except Exception as e:
                self.log_error("Unexpected error in game loop", e)
                print("An unexpected error occurred. Check the log for details.")

def main():
    """Main entry point with exception handling"""
    try:
        game = AdventureGame()
        game.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        print("Check error_log.txt for details.")

if __name__ == "__main__":
    main()
