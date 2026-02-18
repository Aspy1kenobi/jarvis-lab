# Constitutional AI: Harmlessness from AI Feedback

**Authors:** Yuntao Bai, Saurav Kadavath, et al. (Anthropic)  
**Year:** 2022  
**Link:** https://arxiv.org/abs/2212.08073  
**Read Date:** 2025-02-18

## 🎯 Main Contribution
AI can be trained to critique and improve its own outputs based on constitutional principles, without human feedback in the loop.

## 💡 Key Insights
- **Self-critique loop**: Model generates response → critiques itself → revises
- **Constitutional principles**: Simple rules like "be helpful and harmless"
- **RLAIF (RL from AI Feedback)**: Alternative to RLHF that scales better
- Critique quality improves with model size
- Multiple rounds of revision compound improvements

## 🔬 Methodology
1. Start with base model
2. Give it constitutional principles (written rules)
3. Generate initial response
4. Ask model to critique based on principles
5. Revise based on critique
6. Use AI preferences (not human) to train with RL

## 📊 Results
- Models get significantly more harmless
- Quality doesn't degrade
- Scales better than human feedback
- Works even with simple principles

## 🤔 Relevance to My Research
**EXTREMELY RELEVANT!**
- This is basically what I'm building with debate agents
- Proposer = initial generation
- Critic = constitutional review
- Refiner = revision step
- Could use their prompting strategies directly
- Shows that critique → revision DOES improve outputs

## 💭 Key Differences
- They use same model for all roles (I might use separate agents)
- They focus on safety/harmlessness (I'm focusing on quality/correctness)
- They do RL training (I'm doing inference-time improvement)

## 📌 Quotes
> "We find that when chain-of-thought critiques are used, larger models can write more useful critiques and revisions of their own outputs."

> "Constitutional AI methods scale to more and more capable models without requiring expensive additional human oversight."

## 🔗 Implementation Ideas for My Work
- Use their critique prompts as templates
- Try multi-round revision (not just one debate cycle)
- Measure if quality compounds with more rounds
- Could this replace need for separate critic agent?

## 🎯 Next Steps
- [ ] Try their exact critique prompts with GPT-2
- [ ] Compare: separate agents vs same model different prompts
- [ ] Test if multiple revision rounds help

## Related Papers
- RLHF papers (they're improving on this)
- "Self-Refine" paper (similar idea)
- Debate papers (Anthropic has others)
