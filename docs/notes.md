## RL + ART + RULER notes

## Start with Prompted Models First

**Critical Workflow Advice (from Kyle):**

Always start with prompted frontier models before considering RL training:

**Three Reasons:**
1. **Debug environment separately** - Your tools might not work properly, might not have access to right data
2. **May not need training** - Prompted models might work well enough, saves time and money
3. **Establishes baseline** - Feels great to beat frontier models; validates that RL training was worth it

**Workflow:**
```
1. Build environment + tools
2. Test with GPT-4/Claude (prompted)
3. If good enough → Done! Use prompted model
4. If not good enough → Now consider RL training
```

**From Kyle's talk:** "The first thing you should do is just try using prompted models... that means you don't need to train anything and that saves you a lot of time."

---

## The Two Hard Problems in RL

### Problem 1: Building a Realistic Environment

**Requirements:**
- Must match production usage exactly
- Include same failure modes and bugs  
- Realistic data, inputs, outputs, tools
- If environment ≠ production → agent optimizes for wrong thing

**Why It's Hard:**
- Example: Airbnb agent needs full copy of website with same bugs
- If you don't include failure modes, agent fails in production
- Most companies don't have proper testing environments
- Cooperative agents need realistic user simulators

**Solutions:**
- Simple games (2048, tic-tac-toe): Easy, fully defined
- Web environments: Use WebArena, Mind2Web (Docker containers)
- Email: Use public datasets (Enron)
- Custom business logic: Often the hardest - need to replicate production

**The Emerging RL Environment Industry:**

Silicon Valley is betting big on this problem. Startups like **Mechanize** (building realistic RL environments, working with Anthropic), **Prime Intellect** (offering an "RL environment hub" like Hugging Face for environments), and **Mercor/Surge** (expanding from data labeling into interactive simulations) are racing to become the "Scale AI for environments." Anthropic alone plans to spend $1B+ on RL environments in the coming year. Companies like WebArena and Mind2Web provide open-source web-based environments that simulate realistic browsing, forms, and dynamic content—crucial for benchmarking agent robustness beyond toy examples. The compute demands far exceed static training, creating opportunities but also showing how hard the environment problem remains for most companies without dedicated infrastructure.

### Problem 2: Reward Function

Already covered in sections above (RULER solves this for non-verifiable domains).

---

## When RL Makes Sense (vs Fine-Tuning)

### RL Advantages

**Use RL when:**
- ✅ Task has clear success criteria (even if subjective with RULER)
- ✅ Agent needs to learn from experience/feedback
- ✅ Multi-turn, tool-using agent behavior
- ✅ Want to beat frontier models on specific task
- ✅ Can build realistic environment

**RL Benefits:**
- No labeled data needed (with RULER)
- Learns optimal strategies through trial
- Better for agentic, long-horizon tasks
- Can achieve better than GPT-4/Claude on your task

### Fine-Tuning Still Makes Sense

**Use fine-tuning when:**
- ✅ Forced to use smaller models (latency, single GPU deployment, real-time voice)
- ✅ Style/format adaptation (not performance improvement)
- ✅ Domain-specific vocabulary/terminology
- ⚠️ But: 90% of use cases don't have good ROI for fine-tuning

**From OpenPipe's journey:**
- Started with fine-tuning (2023)
- Hit $1M ARR in 8 months
- Problem: Frontier model prices kept dropping 3-5x
- Pivoted to RL (2025): "25% chance this is right direction" → now 55-60% confident

**Key insight:** RL can improve models without labeled data; fine-tuning just mimics existing data.

---

## Performance Metrics: The Full Picture

Beyond accuracy, RL training improves three key metrics:

### 1. Accuracy
- ART-E: 96% (RL) vs 90% (o3) = 60% error reduction
- Makes product much stronger for user experience

### 2. Cost
- o3: $55 per 1,000 searches (cost prohibitive)
- o4-mini: $8 per 1,000 searches (still expensive)
- Qwen 2.5-14B trained: <$1 per 1,000 searches
- **Order of magnitude cheaper** - driven by using smaller specialized model

### 3. Latency  
- Smaller model: less memory loading, fewer matrix multiplies, faster tokens
- Fewer turns: trained to be more efficient (ART-E learned better search keywords)
- Speculative decoding: works better on smaller task-specific models (higher acceptance rates)

**The Triple Win:** Better accuracy + 10x cheaper + faster response

---

## LoRA for RL Training

**Why LoRA (Low-Rank Adaptation):**
- Less memory to train vs full fine-tuning
- Can multiplex arbitrarily large number of LoRAs on same GPU
- Faster iteration during training
- More flexibility at deployment

**ART uses LoRA by default** for all training - you don't need to configure this, it just works.

---

## GRPO vs PPO

### GRPO (What ART Uses)

**Advantages:**
- No separate value model (simpler)
- No hyperparameters for value model
- Relative scoring easier than absolute
- Works perfectly with RULER

**Disadvantages:**
- Requires parallel rollouts in reproducible environment
- Environment setup is hardest challenge
- Kyle: "GRPO likely a dead end long-term due to environment requirements"

### PPO Alternative

**Advantage:** Can train on real production traces without full environment simulation

**Disadvantage:** More complex (needs value model, more hyperparameters)

**Current state:** GRPO works great for tasks where you can build environments (games, email, code). For complex production systems, environment problem remains unsolved.

---

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

---

## Training Duration & Knowing When You're Done

### Verifiable vs Non-Verifiable: Different Stopping Criteria

| Type | Metric | Stop When | Example |
|------|--------|-----------|---------|
| **Verifiable** | Accuracy % | Plateau + beat baseline (e.g., 96% vs 90%) | 2048 game, code tests |
| **Non-Verifiable** | Avg RULER reward | Reward plateau + manual quality checks | auto_rl.ipynb grammar |
| **Hybrid** | Both RULER reward + Validation accuracy | Validation accuracy plateau (primary) + reward plateau | ART-E email Q&A* |

*ART-E is actually non-verifiable (subjective answer quality) but used hybrid approach: RULER for training + validation accuracy for stopping.

### Expected Timeline & Cost

**From ART-E (with hand-crafted rewards):**
- ~1 week engineering time
- $80 GPU cost
- Training steps not explicitly stated (inferred from "week of training" and convergence curves)*

*RULER paper shows training curves but doesn't specify exact step counts. The "100-200 steps" is estimated from typical RL agent training duration.

**From auto_rl.ipynb (non-verifiable):**
- Few hours training time
- 25 inputs × 3 epochs × 2 groups/step ≈ 38 steps*
- No accuracy metric exists - use RULER rewards + manual inspection

*Math: (25 inputs / 2 groups per step) × 3 epochs = 12.5 × 3 = 37.5 steps, rounds to 38

**RULER advantage:** Converges faster than hand-tuned rewards, allows partial credit.

### Training Curve Phases

1. **Initial Learning (0-40 steps):** Sharp improvement, learns basic tool use
2. **Gradual Refinement (40-100+):** Approaches/beats baseline
3. **Convergence:** Plateau - risk of reward hacking if continued

### Stop Training When:

**Verifiable domains:**
- ✅ Accuracy plateaus 10+ steps
- ✅ Beat baseline by meaningful margin
- ✅ Target threshold reached (e.g., 95%+)

**Non-verifiable domains:**
- ✅ Avg reward plateaus 10+ steps
- ✅ Manual inspection: outputs consistently good
- ✅ Human preference > baseline

**Both:**
- ✅ Secondary goals met (cost, latency, efficiency)
- ✅ No reward hacking in recent rollouts

### Reward Hacking: Critical Warning

**Watch for:**
- Sudden score jump after plateau
- Suspiciously perfect scores
- Weird/repetitive behavior

**Classic examples:**
- NYT Connections: Every word in every category
- Hacker News: "Google lays off 80%" for every article
- Boat race: Circling off-track for points

**Fix:** Adjust reward function to penalize exploit, then continue.

### Essential Monitoring

```python
# 1. Always inspect actual rollouts
for traj in sample_trajectories:
    print(traj.messages())  # What is it doing?

# 2. Track your metric
# Verifiable: accuracy vs baseline
# Non-verifiable: avg reward vs baseline

# 3. Watch for anomalies (sudden jumps = investigate)
```

> "Watch your rollouts - don't blindly trust the reward function."  
> — Kyle Corbitt

**Key insight:** Reward functions are proxies, not truth. Models exploit gaps. 5 minutes of rollout inspection > hours debugging reward hacks.

---

## Hybrid Approach: RULER Training + Validation Stopping

### What It Is

When you have **50-500 labeled examples** (too few for supervised training), use:
1. **RULER for training rewards** - no labels needed, faster convergence
2. **Validation accuracy for stopping** - objective metric, automatic reward hacking detection

This is **not** verifiable domains - it's for subjective tasks with limited labeled data.

### Why It Works

**ART-E (OpenPipe's email agent):**
- Training: RULER judges trajectories (no ground truth)
- Validation: Accuracy checked every 5 steps (uses ground truth)
- Stopping: Validation accuracy plateaus at 96%
- Result: Beat o3 (90%) with 60% error reduction

**RULER paper results:**
- Reasoning Classifier: RULER (90%) > RLVR supervised (89%)
- Customer Support: RULER (93%) > hand-tuned with labels (92%)
- Converged **faster** than hand-tuned baselines

### Implementation Pattern

```python
# Split data (80/10/10)
training: 800 scenarios  # Labels hidden - RULER doesn't need them
validation: 100 scenarios  # Labels for stopping
test: 100 scenarios  # Final evaluation

# Training loop
for step in range(max_steps):
    # 1. RULER training (no ground truth)
    groups = [rollout(model, scenario) for scenario in batch]
    judged_groups = [await ruler_score_group(group, "openai/o4-mini") 
                     for group in groups]
    await model.train(judged_groups)
    
    # 2. Validate every 5 steps (uses ground truth)
    if step % 5 == 0:
        val_accuracy = evaluate_on_validation_set(model, validation_set)
        
        # 3. Stop when validation plateaus
        if val_accuracy >= 95% and plateau_detected():
            break
```

### Key Benefits

**1. Efficient Label Use:** 50-200 examples enough (vs 10,000+ for supervised)

**2. Faster Convergence:** RULER gives partial credit vs binary hand-tuned rewards

**3. Objective Stopping:** Clear metric vs "does this look good?"

**4. Auto Reward Hacking Detection:** Val accuracy drops while reward increases? Stop immediately.

**5. Robust to Noisy Labels:** Labels only in validation, not training (RULER ignores them)

### When Metrics Diverge

| Observation | Interpretation | Action |
|-------------|----------------|--------|
| Val ↑, Reward ↑ | ✅ Healthy | Continue |
| Val plateau, Reward ↑ | ⚠️ Converged | Stop soon |
| Val ↓, Reward ↑ | 🚨 **Reward hacking** | Stop & debug |

### Three Approaches Compared

| Approach | Labels | Training Reward | Stopping | Best For |
|----------|--------|----------------|----------|----------|
| **Verifiable** | 0 | Deterministic | Accuracy | Code, math, games |
| **Pure RULER** | 0 | LLM judge | Reward + inspection | No labels |
| **Hybrid** | 50-500 | LLM judge | **Val accuracy** | Most real tasks |
| **Supervised** | 10,000+ | Ground truth | Val loss | Abundant labels |

**Hybrid is the sweet spot for most production agent tasks.**

---

## Multi-Objective Reward Engineering

### Two Approaches Based on Domain

**Verifiable Domains (Code, Math, Games):**
```python
# Explicit weighted combination in rollout
trajectory.reward = (
    primary_objective * 1.0 +      # Correctness/winning
    secondary_objective * 0.2 +    # Efficiency/style
    penalty * -0.5                 # Avoid bad behaviors
)
```

**Example from 2048:**
```python
max_value_reward = log_scale(max_tile)       # Primary: high tile
board_value_reward = log_scale(total_value)  # Secondary: full board
trajectory.reward = max_value_reward + (board_value_reward * 0.2)
```

**Non-Verifiable Domains (Agents, Dialogue, Content):**
```python
# RULER handles everything via system prompt
system_prompt = """
PRIMARY: Answer questions correctly
SECONDARY: Be efficient (minimize searches)
TERTIARY: Say "I don't know" over guessing
"""

judged_group = await ruler_score_group(group, "openai/o4-mini")
# No manual reward engineering needed
```

### Key Insight: RULER Made Manual Rewards Obsolete

**ART-E Evolution:**

| Version | Approach | Accuracy | Convergence | Engineering |
|---------|----------|----------|-------------|-------------|
| **2024: Manual** | 8 hand-tuned reward components | 96% | Slower | ~1 week* |
| **2025: RULER** | Pure LLM judge + system prompt | 95% | **Faster** | Hours-2 days** |

*From Kyle's talk: "About a week of engineering time" for ART-E with manual rewards  
**From OpenPipe docs: "RULER can reduce implementation time by 2-3x" → hours to couple days vs days to weeks

**From RULER paper:**
> "Surprisingly, convergence was substantially **faster** on most RULER runs than with the hand-tuned baseline... because RULER was able to give partial rewards to answers that were close but incomplete."

### Guidelines

**1. Weight Scaling**
- Primary objective: 1.0 baseline
- Secondary objectives: 0.1-0.3 typical (observed in 2048 example: 0.2)*
- Penalties: Can be larger but watch for dominating

*Best practice: secondary should be "very small amount relative" to primary (Kyle's talk). General RL guidance: start with 10:1 ratio or higher, tune empirically for your task.

**2. For Verifiable Domains**
Use explicit weighted sums - you control exact behavior

**3. For Non-Verifiable Domains**
Use RULER with detailed system prompt - it automatically handles:
- Primary goals (correctness)
- Efficiency (mentioned in default rubric)
- Style preferences (from system prompt)
- Multiple objectives jointly

**4. RULER's Default Efficiency Handling**

From default rubric:
> "A trajectory that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a trajectory that achieves its goal less efficiently."

**Don't manually add efficiency bonuses to RULER scores** - it already considers efficiency.

### The Trade-Off

**Manual rewards:** Full control, slightly higher accuracy, slow to design

**RULER:** Fast to deploy, 1-2% accuracy loss, faster convergence, no tuning

**Recommendation:** Use RULER unless you need that extra 1-2% and have time to iterate.