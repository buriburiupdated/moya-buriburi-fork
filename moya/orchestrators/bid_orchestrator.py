"""
BidOrchestrator for Moya.

A radical implementation that creates a collection of agents that:
- Bid on tasks based on their confidence and historical performance
- Form dynamic teams to solve complex problems
- Learn from past interactions to improve future responses
- Process requests in parallel where possible
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

from moya.orchestrators.base_orchestrator import BaseOrchestrator
from moya.registry.agent_registry import AgentRegistry
from moya.agents.base_agent import Agent

class AgentBid:
    """Represents a bid from an agent to handle a specific task."""
    
    def __init__(self, agent_name: str, confidence: float, processing_time: float = 1.0):
        self.agent_name = agent_name
        self.confidence = confidence  # 0.0 to 1.0
        self.processing_time = processing_time  # Estimated processing time in seconds

class BidOrchestrator(BaseOrchestrator):
    """
    An orchestrator where agents bid on tasks,
    form teams, and compete/collaborate to provide the best responses.
    """

    def __init__(
            self,
            agent_registry: AgentRegistry,
            default_agent_name: Optional[str] = None,
            config: Optional[dict] = {},
    ):
        """
        Initialize the BidOrchestrator.
        
        :param agent_registry: The AgentRegistry to retrieve agents from.
        :param default_agent_name: Fallback agent if no bids are received.
        :param config: Configuration dictionary with bid parameters.
        """
        super().__init__(agent_registry=agent_registry, config=config)
        self.default_agent_name = default_agent_name

        # Performance tracking
        self.agent_performance: Dict[str, Dict[str, float]] = {}  # agent_name → {success_rate, avg_response_time, etc.}
        
        # Configuration
        self.parallel_bidding = self.config.get("parallel_bidding", True)
        self.min_confidence_threshold = self.config.get("min_confidence", 0.6)
        self.team_formation_threshold = self.config.get("team_threshold", 0.4)
        self.max_team_size = self.config.get("max_team_size", 3)
        self.enable_learning = self.config.get("enable_learning", True)
        self.verbose = self.config.get("verbose", False)

    def orchestrate(self, thread_id: str, user_message: str, stream_callback=None, **kwargs) -> str:
        """
        The main orchestration method that implements the bid approach.
        
        :param thread_id: The conversation thread ID.
        :param user_message: The message from the user.
        :param stream_callback: Optional callback for streaming responses.
        :param kwargs: Additional context.
        :return: The final response.
        """
        self.log(f"Received user message: {user_message[:50]}...")
        
        # Phase 1: Collect bids from agents
        bids = self._collect_agent_bids(thread_id, user_message, **kwargs)
        if not bids:
            return self._handle_no_bids(thread_id, user_message, **kwargs)
        
        # Phase 2: Select winning agent(s) based on bids
        if self._should_form_team(bids, user_message):
            self.log("Forming an agent team to handle complex request")
            return self._process_with_team(thread_id, user_message, bids, stream_callback, **kwargs)
        else:
            winning_bid = self._select_winning_bid(bids)
            self.log(f"Selected agent {winning_bid.agent_name} with confidence {winning_bid.confidence:.2f}")
            return self._process_with_single_agent(thread_id, user_message, winning_bid, stream_callback, **kwargs)

    def _collect_agent_bids(self, thread_id: str, user_message: str, **kwargs) -> List[AgentBid]:
        """Collect bids from all available agents."""
        available_agents = self.agent_registry.list_agents()
        
        if not available_agents:
            return []
        
        bids = []
        
        if self.parallel_bidding:
            # Parallel bid collection
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._get_agent_bid, agent_info.name, thread_id, user_message, **kwargs)
                    for agent_info in available_agents
                ]
                
                for future in futures:
                    bid = future.result()
                    if bid:
                        bids.append(bid)
        else:
            # Sequential bid collection
            for agent_info in available_agents:
                bid = self._get_agent_bid(agent_info.name, thread_id, user_message, **kwargs)
                if bid:
                    bids.append(bid)
        
        return sorted(bids, key=lambda x: x.confidence, reverse=True)

    def _get_agent_bid(self, agent_name: str, thread_id: str, user_message: str, **kwargs) -> Optional[AgentBid]:
        """Get a bid from a specific agent."""
        agent = self.agent_registry.get_agent(agent_name)
        
        # Check if agent exists
        if agent is None:
            self.log(f"Agent {agent_name} not found in registry")
            return None
        
        # Check if agent has a bid_on_task method
        if hasattr(agent, 'bid_on_task') and callable(getattr(agent, 'bid_on_task')):
            try:
                bid_info = agent.bid_on_task(user_message, thread_id=thread_id, **kwargs)
                return AgentBid(
                    agent_name=agent_name,
                    confidence=bid_info.get('confidence', 0.0),
                    processing_time=bid_info.get('processing_time', 1.0)
                )
            except Exception as e:
                self.log(f"Error getting bid from {agent_name}: {str(e)}")
                return None
        
        # Fall back to estimating confidence from agent description
        confidence = self._estimate_agent_confidence(agent, user_message)
        return AgentBid(agent_name=agent_name, confidence=confidence)
    
    def _estimate_agent_confidence(self, agent: Agent, user_message: str) -> float:
        """Estimate agent confidence based on keywords in the message and agent description."""
        # Get agent name safely
        agent_name = getattr(agent, 'name', str(agent))
        
        # Get description safely
        description = getattr(agent, 'description', "") or ""
        description = description.lower()
        message = user_message.lower()
        
        # Count overlapping words between description and message
        desc_words = set(description.split())
        msg_words = set(message.split())
        overlap = len(desc_words.intersection(msg_words))
        
        # Factor in historical performance
        perf = self.agent_performance.get(agent_name, {}).get('success_rate', 0.5)
        
        # More discriminating confidence calculation - start with a lower base
        # and require more overlapping words to achieve high confidence
        base_confidence = 0.1  # Reduced from 0.3
        overlap_factor = min(overlap * 0.15, 0.5)  # Capped at 0.5, requires ~3-4 words to max out
        perf_factor = perf * 0.4  # Reduced from 0.6
        
        confidence = min(base_confidence + overlap_factor + perf_factor, 1.0)
        
        return confidence

    def _select_winning_bid(self, bids: List[AgentBid]) -> AgentBid:
        """Select the winning bid from the list."""
        if not bids:
            # This shouldn't happen as we check earlier, but just in case
            available_agents = self.agent_registry.list_agents()
            if available_agents:
                agent_name = self.default_agent_name or available_agents[0].name
            else:
                agent_name = self.default_agent_name or "default_agent"
            return AgentBid(agent_name=agent_name, confidence=0.1)
            
        return bids[0]  # We already sorted by confidence
    
    def _estimate_task_complexity(self, user_message: str) -> float:
        """
        Estimate the complexity of a task based on multiple factors.
        Returns a normalized complexity score between 0.0 (very simple) and 1.0 (extremely complex).
        """
        # Initialize base complexity
        complexity = 0.0
        
        # 1. Length-based complexity - longer messages often indicate more complex tasks
        message_length = len(user_message)
        # Normalize using a sigmoid function to handle very long messages
        length_factor = min(0.3, 1 / (1 + math.exp(-message_length / 500 + 3)))
        complexity += length_factor
        
        # 2. Question complexity - multiple questions indicate higher complexity
        question_count = user_message.count('?')
        question_factor = min(0.2, question_count * 0.05)
        complexity += question_factor
        
        # 3. Linguistic complexity indicators
        complexity_indicators = [
            "complex", "difficult", "advanced", "sophisticated", 
            "analyze", "evaluation", "compare", "synthesis",
            "integration", "optimization", "trade-off"
        ]
        
        indicator_count = sum(1 for word in complexity_indicators if word.lower() in user_message.lower())
        indicator_factor = min(0.15, indicator_count * 0.03)
        complexity += indicator_factor
        
        # 4. Domain-specific complexity - look for technical terms
        technical_domains = {
            "programming": ["code", "function", "algorithm", "implementation", "debug"],
            "mathematics": ["equation", "theorem", "proof", "calculation", "formula"],
            "finance": ["investment", "portfolio", "asset", "liability", "valuation"],
            "science": ["experiment", "hypothesis", "analysis", "research", "methodology"]
        }
        
        domain_matches = {}
        for domain, terms in technical_domains.items():
            matches = sum(1 for term in terms if term.lower() in user_message.lower())
            if matches > 0:
                domain_matches[domain] = matches
        
        # Higher complexity if multiple domains are involved (interdisciplinary)
        domain_count = len(domain_matches)
        domain_factor = min(0.2, domain_count * 0.1)
        complexity += domain_factor
        
        # 5. Task structure complexity - multiple subtasks indicated by numbered lists, bullets, etc.
        structure_indicators = re.findall(r'(\d+\.\s|\*\s|-\s|•\s|Step \d+)', user_message)
        structure_factor = min(0.15, len(structure_indicators) * 0.025)
        complexity += structure_factor
        
        # 6. Dependency indicators - tasks that depend on other tasks or have constraints
        dependency_terms = ["if", "when", "after", "before", "only if", "unless", "depending on", "given that"]
        dependency_count = sum(1 for term in dependency_terms if term.lower() in user_message.lower())
        dependency_factor = min(0.1, dependency_count * 0.02)
        complexity += dependency_factor
        
        # 7. Temporal complexity - deadlines, scheduling, timing requirements
        temporal_terms = ["deadline", "schedule", "timeline", "by tomorrow", "within", "hours", "minutes"]
        temporal_count = sum(1 for term in temporal_terms if term.lower() in user_message.lower())
        temporal_factor = min(0.1, temporal_count * 0.025)
        complexity += temporal_factor
        
        # Optional: Historical complexity adjustment
        # If we've seen similar tasks before, adjust based on past performance
        historical_adjustment = self._get_historical_complexity_adjustment(user_message)
        complexity += historical_adjustment
        
        # Ensure final value is between 0 and 1
        return max(0.0, min(1.0, complexity))


    
    def _should_form_team(self, bids: List[AgentBid], user_message: str) -> bool:
        """Determine if we should form a team of agents."""
        if len(bids) < 2:
            return False
        
        task_complexity = self._estimate_task_complexity(user_message)

        if task_complexity < 0.7:
            return False
        
        best_confidence = bids[0].confidence
        runner_up_confidence = bids[1].confidence if len(bids) > 1 else 0
        
        # If one agent is clearly better, use just that one
        confidence_gap = best_confidence - runner_up_confidence
        if best_confidence > 0.7 and confidence_gap > 0.2:
            return False
        
        # If the best agent isn't very confident, try a team approach
        if best_confidence < self.team_formation_threshold:
            return True
        
        # Only form a team if multiple agents have high and similar confidence
        high_confidence_agents = [b for b in bids if b.confidence > self.team_formation_threshold]
        similar_confidence = all(
            abs(b.confidence - bids[0].confidence) < 0.25 
            for b in high_confidence_agents[:min(3, len(high_confidence_agents))]
        )
        
        return len(high_confidence_agents) >= 2 and similar_confidence

    def _process_with_single_agent(self, thread_id: str, user_message: str, 
                                  bid: AgentBid, stream_callback=None, **kwargs) -> str:
        """Process the request with a single agent."""
        start_time = time.time()
        agent = self.agent_registry.get_agent(bid.agent_name)
        
        if stream_callback:
            response = ""
            message_stream = agent.handle_message_stream(user_message, thread_id=thread_id, **kwargs)
            if message_stream is None:
                message_stream = []

            for chunk in message_stream:
                stream_callback(chunk)
                response += chunk
        else:
            response = agent.handle_message(user_message, thread_id=thread_id, **kwargs)
            
        # Update performance metrics
        processing_time = time.time() - start_time
        self._update_agent_performance(bid.agent_name, processing_time)
        
        return response

    def _process_with_team(self, thread_id: str, user_message: str, 
                      bids: List[AgentBid], stream_callback=None, **kwargs) -> str:
        """Process the request with a team of agents."""
        # Select the top N agents
        team_bids = bids[:min(len(bids), self.max_team_size)]
        team_agents = [self.agent_registry.get_agent(bid.agent_name) for bid in team_bids]
        
        self.log(f"Team formed with agents: {[bid.agent_name for bid in team_bids]}")
        
        # Get responses from all team members in parallel
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(agent.handle_message, user_message, thread_id=thread_id, **kwargs)
                for agent in team_agents
            ]
            
            team_responses = []
            for future, bid in zip(futures, team_bids):
                try:
                    response = future.result()
                    team_responses.append((bid, response))
                except Exception as e:
                    self.log(f"Error from team agent {bid.agent_name}: {str(e)}")
        
        # Synthesize the final response
        final_response = self._synthesize_team_responses(user_message, team_responses)
        
        # If streaming is requested, stream the final response
        if stream_callback:
            # Fix: Stream by words or small sentences instead of every 5th character
            words = final_response.split(' ')
            for i in range(0, len(words), 3):
                chunk = ' '.join(words[i:i+3]) + ' '
                stream_callback(chunk)
                time.sleep(0.01)
        
        return final_response

    def _synthesize_team_responses(self, user_message: str, 
                                  team_responses: List[Tuple[AgentBid, str]]) -> str:
        """Synthesize multiple agent responses into a single coherent response."""
        if not team_responses:
            return "[No team responses received]"
            
        # Sort by confidence
        team_responses.sort(key=lambda x: x[0].confidence, reverse=True)
        
        # For simplicity, we'll weight responses by confidence and combine them
        highest_confidence_response = team_responses[0][1]
        
        if len(team_responses) == 1:
            return highest_confidence_response
            
        # Simple synthesis strategy - take the highest confidence response
        # and augment with insights from other responses
        synthesized = f"{highest_confidence_response}\n\n"
        
        # Check if we have an agent specifically for synthesis
        synthesis_agent = self._get_synthesis_agent()
        if synthesis_agent:
            # If we have a dedicated synthesis agent, use it
            synthesis_input = {
                "user_query": user_message,
                "responses": [
                    {"agent": bid.agent_name, "confidence": bid.confidence, "response": response}
                    for bid, response in team_responses
                ]
            }
            return synthesis_agent.handle_message(str(synthesis_input))
        
        synthesized += "Additional insights:\n"
        for bid, response in team_responses[1:]:
            insight = self._extract_insight(response)
            if insight:
                synthesized += f"- {insight}\n"
        
        return synthesized

    def _extract_insight(self, response: str) -> str:
        """Extract a key insight from a response."""
        if len(response) > 100:
            return response[:100].rsplit(' ', 1)[0] + "..."
        return response

    def _get_synthesis_agent(self) -> Optional[Agent]:
        """Get an agent specialized in synthesis if available."""
        agents = self.agent_registry.list_agents()
        synthesis_agents = [
            agent_info for agent_info in agents 
            if "synthesis" in agent_info.name.lower() or "combine" in agent_info.name.lower()
        ]
        if synthesis_agents:
            return self.agent_registry.get_agent(synthesis_agents[0].name)
        return None
    
    def _handle_no_bids(self, thread_id: str, user_message: str, **kwargs) -> str:
        """Handle the case when no bids are received."""
        if self.default_agent_name:
            agent = self.agent_registry.get_agent(self.default_agent_name)
            return agent.handle_message(user_message, thread_id=thread_id, **kwargs)
        return "[No agents available to handle this request.]"

    def _update_agent_performance(self, agent_name: str, processing_time: float) -> None:
        """Update performance metrics for an agent."""
        if not self.enable_learning:
            return
            
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = {
                'success_rate': 0.5,  # Start with neutral rating
                'avg_response_time': processing_time,
                'request_count': 1
            }
            return
            
        perf = self.agent_performance[agent_name]
        req_count = perf['request_count'] + 1
        
        perf['avg_response_time'] = (
            (perf['avg_response_time'] * perf['request_count']) + processing_time
        ) / req_count
        
        perf['request_count'] = req_count
        

    def log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[BidOrchestrator] {message}")