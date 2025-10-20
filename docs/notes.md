## RL + ART + RULER notes

### 1. **Verifiable Domains**
Problems where success can be objectively measured programmatically.

**Examples:**
- Games (2048, Chess, Go) - clear win/loss
- Math problems - check if answer matches ground truth
- Coding - run tests, check if code executes correctly
- Formal systems - compilers, type checkers

**Reward Assignment:** Deterministic function
```python
trajectory.reward = calculate_score(outcome)  # No LLM needed
```

### 2. **Non-Verifiable (Subjective) Domains**
Problems where quality requires judgment and can't be objectively measured.

**Examples:**
- Email/customer support - "Did this answer the question well?"
- Content generation - "Is this title engaging?"
- Dialogue - "Was this response helpful?"
- Creative tasks - "Is this design appealing?"

**Reward Assignment:** LLM-as-judge (via RULER)
```python
judged_group = await ruler_score_group(group, "openai/o4-mini")
# RULER ranks trajectories and assigns relative rewards
```

---

## Core RL Concepts

### **Environment**
The system that defines:
- **State representation** (e.g., `TwentyFortyEightGame` with board state)
- **Action space** (e.g., left/right/up/down moves)
- **Transition rules** (e.g., `condense_board()`, `populate_random_cell()`)
- **Terminal conditions** (e.g., `check_game_finished()`)

**Key insight:** Environments should be as simple as possible - complexity should be in the strategy, not the environment.

### **Rollout**
A single episode of agent interaction with environment.

**Process:**
1. Initialize environment state
2. Agent observes state, takes action
3. Environment transitions to new state
4. Repeat until terminal condition
5. Evaluate outcome and assign reward

```python
async def rollout(model, scenario) -> Trajectory:
    # Play one complete episode
    # Return trajectory with reward
```

### **Trajectory**
The complete record of one rollout episode.

**Contains:**
- `messages_and_choices`: All system/user/assistant messages
- `reward`: Single scalar reward for this episode
- `metadata`: Optional tracking info (game_id, step, etc.)
- `metrics`: Additional measurements (max_value, move_count, etc.)

```python
trajectory = art.Trajectory(
    messages_and_choices=[...],  # Conversation history
    reward=<calculated_value>,   # Success measure
    metadata={...},              # Tracking
)
```

---

## RULER: LLM-as-Judge for Trajectory Scoring

### How It Works

**Input: Unranked Trajectories**
```python
# Create trajectories with placeholder rewards
group = art.TrajectoryGroup([
    art.Trajectory(messages=[...], reward=0),  # Unscored
    art.Trajectory(messages=[...], reward=0),  # Unscored
    art.Trajectory(messages=[...], reward=0),  # Unscored
])
```

**Process: RULER Ranks & Scores**
```python
# RULER uses LLM judge to compare and rank all trajectories
judged_group = await ruler_score_group(
    group,             # Your unranked trajectories
    "openai/o4-mini",  # LLM judge model
    debug=True         # Optional: see judge's reasoning
)
```

**Output: Scored Trajectories**
```python
# Same trajectories, now with relative rewards assigned
judged_group.trajectories[0].reward  # e.g., 1.0 (best)
judged_group.trajectories[1].reward  # e.g., 0.5 (medium)
judged_group.trajectories[2].reward  # e.g., 0.0 (worst)
```

### Key Properties

**No Pre-Ranking Required:**
- You pass unranked trajectories
- RULER does all ranking via LLM judge
- No labeled data or expert rankings needed

**Relative Scoring:**
- Rewards are relative within the group, not absolute
- Only ranking matters: "A is better than B"
- Perfect fit for GRPO which needs group comparisons

**Efficient Judging:**
- One LLM call judges entire group (N trajectories)
- Much cheaper than N separate absolute scoring calls
- Works with smaller/cheaper judge models (e.g., Qwen 2.5-32B)

### Performance Optimizations

**1. Batch Judging (GRPO's Natural Fit)**
```python
# Instead of: judge each trajectory individually (N calls)
# RULER does: judge all N trajectories together (1 call)
```

**2. Smaller Judge Models Work**
- Don't need GPT-4/Claude for judging
- Qwen 2.5-32B works fine for relative comparisons
- "Which is better?" is easier than "Is this good?"

**3. Prompt Caching**
```
System: How to evaluate... [CACHED]
User: Task description... [CACHED]
User: Rank these N attempts [ONLY THIS CHANGES]
```

**4. Parallel Rollout Generation**
```python
# 18 agent rollouts run in parallel
# Then 1 judge call ranks them all
art.TrajectoryGroup(rollout(model, scenario) for _ in range(18))
```

**5. Judge Once Per Episode**
- 50-turn conversation = 50 agent calls + 1 judge call
- Not judging every single turn

### Cost Example (from Kyle's Talk)
- **Email Agent Training:** $80 total (including all LLM judge calls)
- **Result:** 96% accuracy vs 90% for o3
- **60% error reduction** - judge cost is negligible compared to value

---

## The Complete Training Flow

### Verifiable Domain (2048)
```python
# 1. Generate trajectories with deterministic rewards
for i in range(20):
    groups = [
        art.TrajectoryGroup(
            rollout(model, scenario) for _ in range(18)
        )
    ]
    
    # 2. Train directly (rewards already set in rollout)
    await model.train(groups, config=art.TrainConfig(...))
```

### Non-Verifiable Domain (Email Agent)
```python
# 1. Generate trajectories WITHOUT scores
group = art.TrajectoryGroup(
    rollout(model, scenario) for _ in range(18)
)

# 2. RULER judges and assigns rewards
judged_group = await ruler_score_group(group, "openai/o4-mini")

# 3. Train with scored trajectories
await model.train([judged_group], config=art.TrainConfig(...))
```

---

## The Distinction That Matters

**Verifiable** → Environment provides ground truth → Fast, deterministic rewards → Direct training

**Non-Verifiable** → Need LLM judge → Adds latency but unlocks subjective domains → RULER + training

Both use the same core machinery (environments, rollouts, trajectories), just differ in **how the reward gets assigned**.

**GRPO's Key Insight:** Only need to know "which trajectory is better" within a group, not "how good is this trajectory" in absolute terms. This makes LLM judging practical and effective even with smaller models.