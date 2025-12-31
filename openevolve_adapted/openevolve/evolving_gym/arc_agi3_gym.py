"""
ARC-AGI-3 Game Environment for RL Training

This module provides a gym-like interface for the ARC-AGI-3 game environment,
compatible with ThetaEvolve's evolving_gym pattern.

Key features:
- Parallel episode management (N concurrent game sessions)
- History tracking with configurable window for prompting
- Episode persistence across rollout batches
- Distributed reward computation (pluggable)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from requests.cookies import RequestsCookieJar

# Load .env file automatically
try:
    from dotenv import load_dotenv
    # Try multiple locations for .env file
    env_paths = [
        Path.cwd() / ".env",  # Current working directory
        Path(__file__).parent.parent.parent.parent.parent / ".env",  # ThetaEvolve root
        Path.home() / ".env",  # Home directory
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            print(f"[ArcAgi3Gym] Loaded .env from {env_path}")
            break
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures (mirroring ARC-AGI-3-Agents/agents/structs.py)
# ============================================================================

class GameState(str, Enum):
    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


@dataclass
class FrameData:
    """Represents a single frame/state from the game."""
    game_id: str = ""
    frame: List[List[List[int]]] = field(default_factory=list)  # 3D int array
    state: GameState = GameState.NOT_PLAYED
    score: int = 0
    action_id: Optional[int] = None  # Action that led to this frame
    action_data: Optional[Dict[str, Any]] = None  # Action parameters (for complex actions)
    action_reasoning: Optional[str] = None  # Model's reasoning for the action
    guid: Optional[str] = None
    available_actions: List[int] = field(default_factory=list)  # List of valid action IDs
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "frame": self.frame,
            "state": self.state.value if isinstance(self.state, GameState) else self.state,
            "score": self.score,
            "action_id": self.action_id,
            "action_data": self.action_data,
            "action_reasoning": self.action_reasoning,
            "available_actions": self.available_actions,
        }

    def compute_state_hash(self) -> str:
        """
        Compute a hash of the game state for exploration tracking.
        Uses the frame data and score to create a unique identifier.
        """
        import hashlib
        # Convert frame to a stable string representation
        frame_str = json.dumps(self.frame, sort_keys=True) if self.frame else ""
        # Include score in hash to distinguish states with same frame but different scores
        state_str = f"{frame_str}|{self.score}|{self.state.value if isinstance(self.state, GameState) else self.state}"
        return hashlib.md5(state_str.encode()).hexdigest()

    @classmethod
    def from_api_response(cls, data: Dict[str, Any], action_id: Optional[int] = None,
                          action_data: Optional[Dict] = None, action_reasoning: Optional[str] = None) -> "FrameData":
        """Parse API response into FrameData."""
        state_str = data.get("state", "NOT_PLAYED")
        try:
            state = GameState(state_str)
        except ValueError:
            state = GameState.NOT_PLAYED

        available = data.get("available_actions", [])
        # Convert action objects to IDs if needed
        if available and isinstance(available[0], dict):
            available = [a.get("id", a.get("value", 0)) for a in available]
        elif available and hasattr(available[0], "value"):
            available = [a.value for a in available]

        return cls(
            game_id=data.get("game_id", ""),
            frame=data.get("frame", []),
            state=state,
            score=data.get("score", 0),
            action_id=action_id,
            action_data=action_data,
            action_reasoning=action_reasoning,
            guid=data.get("guid"),
            available_actions=available,
        )


@dataclass
class EpisodeState:
    """Tracks the state of a single episode/game session."""
    episode_id: str
    game_id: str
    card_id: str
    guid: Optional[str] = None
    frames: List[FrameData] = field(default_factory=list)
    total_score: int = 0
    is_finished: bool = False
    finish_state: Optional[GameState] = None
    play_count: int = 0  # Number of times this game has been played
    created_at: float = field(default_factory=time.time)
    # Exploration tracking: set of state hashes visited in this episode
    explored_states: set = field(default_factory=set)

    @property
    def current_frame(self) -> Optional[FrameData]:
        return self.frames[-1] if self.frames else None

    @property
    def step_count(self) -> int:
        return len(self.frames)


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    frame: Optional[FrameData] = None
    error: Optional[str] = None
    score_delta: int = 0
    episode_finished: bool = False
    finish_state: Optional[GameState] = None
    # Exploration tracking
    is_new_state: bool = False  # True if this state was not previously explored
    state_hash: Optional[str] = None  # Hash of the new state


@dataclass
class Result:
    """Result object compatible with ThetaEvolve's evolving_gym pattern."""
    parent: Optional[Any] = None  # Episode metadata
    child_metrics: Optional[Dict[str, Any]] = None
    child_program: Optional[Any] = None  # Not used for ARC-AGI-3
    llm_response: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    iteration_time: Optional[float] = None
    runtime_environment_path: Optional[str] = None

    # ARC-AGI-3 specific fields
    episode_id: Optional[str] = None
    step_in_episode: Optional[int] = None
    action_taken: Optional[int] = None
    score_delta: Optional[int] = None
    episode_finished: Optional[bool] = None
    finish_state: Optional[str] = None


# ============================================================================
# Action Parsing
# ============================================================================

# Action ID to name mapping
ACTION_NAMES = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",  # Complex action with x, y
    7: "ACTION7",
}

# Which actions are "complex" (require x, y coordinates)
COMPLEX_ACTIONS = {6}


def parse_action_from_response(response: str) -> Tuple[Optional[int], Optional[Dict], Optional[str]]:
    """
    Parse LLM response to extract action.

    Expected formats:
    - "ACTION3" or "action3" (simple)
    - "ACTION6 x=10 y=20" or {"action": 6, "x": 10, "y": 20} (complex)
    - With reasoning: "I choose ACTION3 because..."

    Returns:
        (action_id, action_data, reasoning)
    """
    response = response.strip()
    reasoning = None
    action_id = None
    action_data = {}

    # Try to extract reasoning (everything after "because", "reason:", etc.)
    for marker in ["because", "reason:", "reasoning:", "my reasoning:"]:
        if marker in response.lower():
            idx = response.lower().find(marker)
            reasoning = response[idx:].strip()
            response = response[:idx].strip()
            break

    # Try JSON format first
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            action_id = data.get("action", data.get("action_id", data.get("id")))
            if "x" in data and "y" in data:
                action_data = {"x": int(data["x"]), "y": int(data["y"])}
            reasoning = data.get("reasoning", reasoning)
            if action_id is not None:
                return int(action_id), action_data or None, reasoning
    except (json.JSONDecodeError, TypeError):
        pass

    # Try text format: "ACTION3" or "action3"
    response_upper = response.upper()
    for aid, name in ACTION_NAMES.items():
        if name in response_upper:
            action_id = aid
            # For complex actions, try to extract x, y
            if aid in COMPLEX_ACTIONS:
                import re
                x_match = re.search(r'x\s*[=:]\s*(\d+)', response, re.IGNORECASE)
                y_match = re.search(r'y\s*[=:]\s*(\d+)', response, re.IGNORECASE)
                if x_match and y_match:
                    action_data = {"x": int(x_match.group(1)), "y": int(y_match.group(1))}
            break

    # Try just a number
    if action_id is None:
        import re
        match = re.search(r'\b([0-7])\b', response)
        if match:
            action_id = int(match.group(1))

    return action_id, action_data or None, reasoning


# ============================================================================
# ARC-AGI-3 Gym
# ============================================================================

class ArcAgi3Gym:
    """
    ARC-AGI-3 game environment for RL training.

    Manages N parallel episodes, tracks history, and provides
    problem_generator() / response_scorer() interface compatible
    with ThetaEvolve's evolving_gym pattern.
    """

    def __init__(
        self,
        game_id: str = "ls20",
        api_key: Optional[str] = None,
        api_url: str = "https://three.arcprize.org",
        num_parallel_episodes: int = 8,
        max_actions_per_episode: int = 80,
        max_plays_per_episode: int = 10,
        history_window: int = 5,
        reward_fn: Optional[Callable[[EpisodeState, ActionResult], float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize ARC-AGI-3 Gym.

        Args:
            game_id: Game identifier (e.g., "ls20")
            api_key: ARC-AGI-3 API key (or from env ARC_API_KEY)
            api_url: API base URL
            num_parallel_episodes: Number of parallel game sessions
            max_actions_per_episode: Max actions before forced reset
            max_plays_per_episode: Max replays of same game before moving on
            history_window: Number of recent frames to include in prompt
            reward_fn: Custom reward function(episode_state, action_result) -> float
            seed: Random seed for reproducibility
        """
        self.game_id = game_id
        self.api_key = api_key or os.getenv("ARC_API_KEY", "")
        self.api_url = api_url.rstrip("/")
        self.num_parallel_episodes = num_parallel_episodes
        self.max_actions_per_episode = max_actions_per_episode
        self.max_plays_per_episode = max_plays_per_episode
        self.history_window = history_window
        self.reward_fn = reward_fn or self._default_reward_fn
        self.seed = seed

        # Validate API key
        if not self.api_key:
            logger.warning("[ArcAgi3Gym] WARNING: ARC_API_KEY is empty! API calls will fail.")
            print(f"[ArcAgi3Gym] WARNING: ARC_API_KEY is empty! Set via env var or api_key parameter.")
        else:
            # Only show first/last few chars for security
            key_preview = f"{self.api_key[:4]}...{self.api_key[-4:]}" if len(self.api_key) > 8 else "***"
            logger.info(f"[ArcAgi3Gym] Using API key: {key_preview}")
            print(f"[ArcAgi3Gym] Using API key: {key_preview}")

        # HTTP session
        self.headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._session: Optional[requests.Session] = None
        self._cookies: RequestsCookieJar = RequestsCookieJar()

        # Episode management
        self.episodes: Dict[str, EpisodeState] = {}  # episode_id -> EpisodeState
        self._episode_queue: List[str] = []  # Round-robin queue of episode IDs
        self._current_episode_idx: int = 0

        # Global tracking
        self.total_steps: int = 0
        self.total_episodes_completed: int = 0
        self.total_wins: int = 0

        # Initialization flag
        self._initialized = False

        # Semaphore for concurrent API calls
        self._api_semaphore = asyncio.Semaphore(num_parallel_episodes)

        # Recorder (optional, for compatibility with evolving_gym pattern)
        self._recorder = None

        logger.info(
            f"ArcAgi3Gym initialized: game={game_id}, parallel_episodes={num_parallel_episodes}, "
            f"history_window={history_window}, max_actions={max_actions_per_episode}"
        )

    # --------------------------------------------------------------------------
    # Session Management
    # --------------------------------------------------------------------------

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(self.headers)
            self._session.cookies = self._cookies
        return self._session

    def _api_post(self, endpoint: str, data: Dict) -> Dict:
        """Make POST request to API."""
        session = self._get_session()
        url = f"{self.api_url}{endpoint}"
        try:
            logger.debug(f"API POST: {url} with data keys: {list(data.keys())}")
            response = session.post(url, json=data, timeout=30)
            
            # Log response details before raising
            if not response.ok:
                logger.error(
                    f"API request failed: {endpoint}\n"
                    f"  Status: {response.status_code}\n"
                    f"  Response: {response.text[:500]}\n"
                    f"  Request data keys: {list(data.keys())}\n"
                    f"  Headers had API key: {bool(self.api_key)}"
                )
            
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {endpoint} - {e}")
            raise

    def _api_get(self, endpoint: str) -> Dict:
        """Make GET request to API."""
        session = self._get_session()
        url = f"{self.api_url}{endpoint}"
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {endpoint} - {e}")
            raise

    def _resolve_game_id(self, game_id_prefix: str) -> str:
        """
        Resolve a game ID prefix to the full game ID.
        e.g., "ls20" -> "ls20-fa137e247ce6"
        
        This matches the behavior of the original ARC-AGI-3-Agents repo.
        """
        # If already looks like a full ID (contains hash), return as-is
        if "-" in game_id_prefix and len(game_id_prefix) > 10:
            return game_id_prefix
        
        try:
            # Fetch available games
            games = self._api_get("/api/games")
            if isinstance(games, list):
                # Find game that starts with the prefix
                matching = [g["game_id"] for g in games if g.get("game_id", "").startswith(game_id_prefix)]
                if matching:
                    full_id = matching[0]
                    logger.info(f"Resolved game ID: {game_id_prefix} -> {full_id}")
                    print(f"[ArcAgi3Gym] Resolved game ID: {game_id_prefix} -> {full_id}")
                    return full_id
                else:
                    available = [g.get("game_id", "") for g in games]
                    logger.warning(f"Game '{game_id_prefix}' not found. Available: {available}")
        except Exception as e:
            logger.warning(f"Failed to resolve game ID: {e}")
        
        return game_id_prefix  # Return original if lookup fails

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize gym by creating parallel episodes."""
        if self._initialized:
            return
        
        # Resolve game ID prefix to full ID (e.g., "ls20" -> "ls20-fa137e247ce6")
        self.game_id = self._resolve_game_id(self.game_id)

        logger.info(f"Initializing {self.num_parallel_episodes} parallel episodes for game {self.game_id}")

        for i in range(self.num_parallel_episodes):
            try:
                episode = await self._create_episode()
                self.episodes[episode.episode_id] = episode
                self._episode_queue.append(episode.episode_id)
                logger.info(f"Created episode {i+1}/{self.num_parallel_episodes}: {episode.episode_id[:8]}")
            except Exception as e:
                logger.error(f"Failed to create episode {i+1}: {e}")
                raise

        self._initialized = True
        logger.info(f"Initialization complete: {len(self.episodes)} episodes ready")

    def initialize_sync(self) -> None:
        """Synchronous initialization wrapper that works from any context."""
        import threading
        
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(self.initialize())
            finally:
                new_loop.close()
        
        try:
            # Check if there's a running loop
            asyncio.get_running_loop()
            # We're inside an async context - run in a separate thread
            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join(timeout=120)  # Wait up to 2 minutes
            if thread.is_alive():
                raise RuntimeError("Initialization timed out")
        except RuntimeError:
            # No running loop - safe to run directly
            run_in_new_loop()

    async def _create_episode(self) -> EpisodeState:
        """Create a new episode by opening scorecard and resetting game."""
        # Open scorecard
        logger.info(f"Opening scorecard for game {self.game_id}...")
        print(f"[ArcAgi3Gym] Opening scorecard for game {self.game_id}...")
        
        card_response = await asyncio.to_thread(
            self._api_post,
            "/api/scorecard/open",
            {"tags": ["thetaevolve", self.game_id]}
        )
        card_id = card_response.get("card_id", "")
        
        if not card_id:
            logger.error(f"Failed to get card_id from scorecard open response: {card_response}")
            raise RuntimeError(f"No card_id in response: {card_response}")
        
        logger.info(f"Got card_id: {card_id}")
        print(f"[ArcAgi3Gym] Got card_id: {card_id}")

        # Reset game to get initial frame
        reset_data = {
            "game_id": self.game_id,
            "card_id": card_id,
        }
        logger.info(f"Resetting game with data: {reset_data}")
        print(f"[ArcAgi3Gym] Resetting game with data: {reset_data}")
        
        frame_response = await asyncio.to_thread(
            self._api_post,
            "/api/cmd/RESET",
            reset_data
        )

        initial_frame = FrameData.from_api_response(frame_response, action_id=0)

        episode = EpisodeState(
            episode_id=str(uuid.uuid4()),
            game_id=self.game_id,
            card_id=card_id,
            guid=initial_frame.guid,
            frames=[initial_frame],
            total_score=initial_frame.score,
            play_count=1,
        )

        # Add initial state to explored set
        episode.explored_states.add(initial_frame.compute_state_hash())

        return episode

    # --------------------------------------------------------------------------
    # Problem Generator
    # --------------------------------------------------------------------------

    def problem_generator(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Generate a problem (prompt) for the LLM.

        Returns:
            (prompt_dict, metadata)
            - prompt_dict: {"system": ..., "user": ...}
            - metadata: Contains episode info, full history, current frame
        """
        # Get next episode (round-robin)
        episode = self._get_next_episode()

        if episode is None:
            raise RuntimeError("No episodes available. Call initialize() first.")

        # Build prompt with history
        prompt = self._build_prompt(episode)

        # Metadata includes full history for reward computation
        metadata = {
            "episode_id": episode.episode_id,
            "game_id": episode.game_id,
            "card_id": episode.card_id,
            "guid": episode.guid,
            "step_in_episode": episode.step_count,
            "current_frame": episode.current_frame.to_dict() if episode.current_frame else None,
            "full_history": [f.to_dict() for f in episode.frames],
            "total_score": episode.total_score,
            "play_count": episode.play_count,
        }

        return prompt, metadata

    def _get_next_episode(self) -> Optional[EpisodeState]:
        """Get next episode in round-robin fashion, skipping finished ones."""
        if not self._episode_queue:
            return None

        # Try to find an active episode
        start_idx = self._current_episode_idx
        for _ in range(len(self._episode_queue)):
            episode_id = self._episode_queue[self._current_episode_idx]
            self._current_episode_idx = (self._current_episode_idx + 1) % len(self._episode_queue)

            episode = self.episodes.get(episode_id)
            if episode and not episode.is_finished:
                return episode

        # All episodes finished, try to reset one
        episode_id = self._episode_queue[start_idx]
        episode = self.episodes.get(episode_id)
        if episode:
            # Will be reset in response_scorer when we detect finished state
            return episode

        return None

    def _build_prompt(self, episode: EpisodeState) -> Dict[str, str]:
        """Build system and user prompt for the episode."""
        current_frame = episode.current_frame

        # System prompt
        system_prompt = f"""You are an AI agent playing the ARC-AGI-3 game "{episode.game_id}".

Your goal is to maximize your score by choosing the best action at each step.
The game ends when you reach a WIN state or GAME_OVER state.

Available actions:
- ACTION1 (1): Simple action 1
- ACTION2 (2): Simple action 2
- ACTION3 (3): Simple action 3
- ACTION4 (4): Simple action 4
- ACTION5 (5): Simple action 5
- ACTION6 (6): Complex action with coordinates (requires x and y between 0-63)
- ACTION7 (7): Simple action 7

For simple actions, respond with just the action name (e.g., "ACTION3").
For ACTION6, include coordinates: "ACTION6 x=10 y=20"

Think step by step about which action will best improve your score based on the current game state."""

        # User prompt with frame history
        user_parts = []

        # Add history (up to history_window recent frames)
        # Note: frames[-0:] returns all frames in Python, so handle 0 specially
        if self.history_window > 0:
            history_frames = episode.frames[-(self.history_window):]
        else:
            history_frames = []  # No history when window is 0
        if len(history_frames) > 1:
            user_parts.append("=== Recent History ===")
            for i, frame in enumerate(history_frames[:-1]):  # Exclude current
                action_str = ACTION_NAMES.get(frame.action_id, f"ACTION{frame.action_id}") if frame.action_id else "RESET"
                user_parts.append(f"Step {i+1}: Action={action_str}, Score={frame.score}, State={frame.state.value}")
                if frame.frame:
                    user_parts.append(f"Frame: {self._format_frame(frame.frame)}")
            user_parts.append("")

        # Current state
        user_parts.append("=== Current State ===")
        user_parts.append(f"Step: {episode.step_count}")
        user_parts.append(f"Current Score: {episode.total_score}")
        user_parts.append(f"Game State: {current_frame.state.value if current_frame else 'UNKNOWN'}")

        if current_frame and current_frame.available_actions:
            available = [ACTION_NAMES.get(a, f"ACTION{a}") for a in current_frame.available_actions]
            user_parts.append(f"Available Actions: {', '.join(available)}")

        if current_frame and current_frame.frame:
            user_parts.append(f"Current Frame:\n{self._format_frame(current_frame.frame)}")

        user_parts.append("")
        user_parts.append("Choose your next action:")

        return {
            "system": system_prompt,
            "user": "\n".join(user_parts),
        }

    def _format_frame(self, frame: List[List[List[int]]]) -> str:
        """Format 3D frame array as text."""
        if not frame:
            return "[]"

        # Frame is list of 2D layers; format each layer
        layers = []
        for layer_idx, layer in enumerate(frame):
            if not layer:
                continue
            # Format as grid
            rows = []
            for row in layer:
                rows.append(" ".join(str(v).rjust(3) for v in row))
            layer_str = "\n".join(rows)
            if len(frame) > 1:
                layers.append(f"Layer {layer_idx}:\n{layer_str}")
            else:
                layers.append(layer_str)

        return "\n".join(layers)

    # --------------------------------------------------------------------------
    # Response Scorer
    # --------------------------------------------------------------------------

    async def response_scorer(self, response: str, metadata: Dict[str, Any]) -> Optional[Result]:
        """
        Score the LLM's response by executing the action.

        Args:
            response: LLM response containing action choice
            metadata: Episode metadata from problem_generator()

        Returns:
            Result with reward metrics
        """
        iteration_start = time.time()

        episode_id = metadata.get("episode_id")
        episode = self.episodes.get(episode_id)

        if episode is None:
            logger.error(f"Episode not found: {episode_id}")
            return None

        # Check if episode needs reset (finished in previous step)
        if episode.is_finished or episode.current_frame.state in [GameState.WIN, GameState.GAME_OVER]:
            await self._handle_episode_end(episode)

        # Parse action from response
        action_id, action_data, reasoning = parse_action_from_response(response)

        if action_id is None:
            logger.warning(f"Failed to parse action from response: {response[:100]}")
            # Return penalty for unparseable action
            result = Result(
                parent=metadata,
                child_metrics={
                    "reward": -0.1,  # Small penalty for invalid action
                    "parse_error": True,
                },
                llm_response=response,
                episode_id=episode_id,
                step_in_episode=episode.step_count,
                action_taken=None,
                iteration_time=time.time() - iteration_start,
            )
            return result

        # Execute action
        async with self._api_semaphore:
            action_result = await self._execute_action(episode, action_id, action_data, reasoning)

        if not action_result.success:
            logger.warning(f"Action execution failed: {action_result.error}")
            result = Result(
                parent=metadata,
                child_metrics={
                    "reward": -0.1,
                    "execution_error": True,
                    "error": action_result.error,
                },
                llm_response=response,
                episode_id=episode_id,
                step_in_episode=episode.step_count,
                action_taken=action_id,
                iteration_time=time.time() - iteration_start,
            )
            return result

        # Compute reward
        reward = self.reward_fn(episode, action_result)

        # Update global stats
        self.total_steps += 1
        if action_result.episode_finished:
            self.total_episodes_completed += 1
            if action_result.finish_state == GameState.WIN:
                self.total_wins += 1

        result = Result(
            parent=metadata,
            child_metrics={
                "reward": reward,
                "score": action_result.frame.score if action_result.frame else 0,
                "score_delta": action_result.score_delta,
                "step_in_episode": episode.step_count,
                "episode_finished": action_result.episode_finished,
                "finish_state": action_result.finish_state.value if action_result.finish_state else None,
                # Exploration tracking
                "is_new_state": action_result.is_new_state,
                "explored_states_count": len(episode.explored_states),
            },
            llm_response=response,
            artifacts={
                "frame": action_result.frame.to_dict() if action_result.frame else None,
                "action_id": action_id,
                "action_data": action_data,
                "reasoning": reasoning,
            },
            episode_id=episode_id,
            step_in_episode=episode.step_count,
            action_taken=action_id,
            score_delta=action_result.score_delta,
            episode_finished=action_result.episode_finished,
            finish_state=action_result.finish_state.value if action_result.finish_state else None,
            iteration_time=time.time() - iteration_start,
        )

        return result

    async def _execute_action(
        self,
        episode: EpisodeState,
        action_id: int,
        action_data: Optional[Dict] = None,
        reasoning: Optional[str] = None,
    ) -> ActionResult:
        """Execute action in the game and update episode state."""
        action_name = ACTION_NAMES.get(action_id, f"ACTION{action_id}")

        # Build request data
        request_data = {
            "game_id": episode.game_id,
        }
        if episode.guid:
            request_data["guid"] = episode.guid
        if action_data:
            request_data.update(action_data)
        if reasoning:
            request_data["reasoning"] = reasoning

        try:
            response = await asyncio.to_thread(
                self._api_post,
                f"/api/cmd/{action_name}",
                request_data
            )
        except Exception as e:
            return ActionResult(success=False, error=str(e))

        if "error" in response:
            return ActionResult(success=False, error=response.get("error"))

        # Parse frame
        new_frame = FrameData.from_api_response(
            response,
            action_id=action_id,
            action_data=action_data,
            action_reasoning=reasoning,
        )

        # Update episode
        old_score = episode.total_score
        episode.frames.append(new_frame)
        episode.total_score = new_frame.score
        episode.guid = new_frame.guid or episode.guid

        score_delta = new_frame.score - old_score

        # Exploration tracking: check if this state is new
        state_hash = new_frame.compute_state_hash()
        is_new_state = state_hash not in episode.explored_states
        if is_new_state:
            episode.explored_states.add(state_hash)

        # Check for episode end
        episode_finished = new_frame.state in [GameState.WIN, GameState.GAME_OVER]
        if episode_finished:
            episode.is_finished = True
            episode.finish_state = new_frame.state

        # Check for max actions
        if episode.step_count >= self.max_actions_per_episode:
            episode.is_finished = True
            episode.finish_state = GameState.GAME_OVER

        return ActionResult(
            success=True,
            frame=new_frame,
            score_delta=score_delta,
            episode_finished=episode_finished,
            finish_state=new_frame.state if episode_finished else None,
            is_new_state=is_new_state,
            state_hash=state_hash,
        )

    async def _handle_episode_end(self, episode: EpisodeState) -> None:
        """Handle episode end by resetting or creating new episode."""
        logger.info(
            f"Episode {episode.episode_id[:8]} ended: state={episode.finish_state}, "
            f"score={episode.total_score}, steps={episode.step_count}"
        )

        # Check if we should replay or move on
        if episode.play_count < self.max_plays_per_episode:
            # Reset same game
            try:
                reset_data = {
                    "game_id": episode.game_id,
                    "card_id": episode.card_id,
                }
                response = await asyncio.to_thread(
                    self._api_post,
                    "/api/cmd/RESET",
                    reset_data
                )
                initial_frame = FrameData.from_api_response(response, action_id=0)

                episode.frames = [initial_frame]
                episode.total_score = initial_frame.score
                episode.guid = initial_frame.guid
                episode.is_finished = False
                episode.finish_state = None
                episode.play_count += 1
                # Clear explored states for new play
                episode.explored_states.clear()
                # Add initial state to explored set
                episode.explored_states.add(initial_frame.compute_state_hash())

                logger.info(f"Reset episode {episode.episode_id[:8]}, play {episode.play_count}")
            except Exception as e:
                logger.error(f"Failed to reset episode: {e}")
        else:
            # Close scorecard and create new episode
            try:
                await asyncio.to_thread(
                    self._api_post,
                    "/api/scorecard/close",
                    {"card_id": episode.card_id}
                )
                new_episode = await self._create_episode()

                # Replace in tracking
                old_id = episode.episode_id
                self.episodes[new_episode.episode_id] = new_episode
                del self.episodes[old_id]

                # Update queue
                idx = self._episode_queue.index(old_id)
                self._episode_queue[idx] = new_episode.episode_id

                logger.info(f"Created new episode {new_episode.episode_id[:8]} to replace {old_id[:8]}")
            except Exception as e:
                logger.error(f"Failed to create new episode: {e}")

    # --------------------------------------------------------------------------
    # Reward Function
    # --------------------------------------------------------------------------

    def _default_reward_fn(self, episode: EpisodeState, action_result: ActionResult) -> float:
        """
        Default reward function. Can be replaced via reward_fn parameter.

        This is designed to be easily modified for different reward strategies.
        
        Current strategy (exploration-based):
        - New state explored: +0.01
        - Revisited state: -0.01 (default penalty)
        - Additional rewards for score changes and episode completion
        """
        reward = 0.0

        # === Exploration-based reward (primary) ===
        if action_result.is_new_state:
            reward += 0.01  # Reward for exploring a new state
        else:
            reward -= 0.01  # Penalty for revisiting an already-explored state

        # === Score delta reward (secondary) ===
        if action_result.score_delta > 0:
            reward += action_result.score_delta * 0.1  # Positive reward for score increase
        elif action_result.score_delta < 0:
            reward += action_result.score_delta * 0.05  # Smaller penalty for score decrease

        # === Episode completion rewards ===
        if action_result.episode_finished:
            if action_result.finish_state == GameState.WIN:
                reward += 1.0  # Bonus for winning
            elif action_result.finish_state == GameState.GAME_OVER:
                reward -= 0.5  # Penalty for game over

        return reward

    def set_reward_fn(self, reward_fn: Callable[[EpisodeState, ActionResult], float]) -> None:
        """Set custom reward function."""
        self.reward_fn = reward_fn

    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get current gym statistics."""
        return {
            "total_steps": self.total_steps,
            "total_episodes_completed": self.total_episodes_completed,
            "total_wins": self.total_wins,
            "win_rate": self.total_wins / max(1, self.total_episodes_completed),
            "active_episodes": len([e for e in self.episodes.values() if not e.is_finished]),
        }

    @property
    def recording_enabled(self) -> bool:
        return self._recorder is not None

    def enable_recording(self, output_dir: str = "./arc_agi3_output") -> None:
        """Enable recording (placeholder for compatibility)."""
        # TODO: Implement recording if needed
        pass

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._session:
            self._session.close()
            self._session = None

        # Close any open scorecards
        for episode in self.episodes.values():
            try:
                self._api_post("/api/scorecard/close", {"card_id": episode.card_id})
            except Exception:
                pass

        self.episodes.clear()
        self._episode_queue.clear()

