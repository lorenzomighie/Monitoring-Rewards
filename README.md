# Monitoring Rewards via Transducers

Project for Elective in AI - Reasoning Agents

An implementation of Monitoring Rewards based on the paper
[Temporal Logic Monitoring Rewards via Transducers](http://www.diag.uniroma1.it/degiacom/papers/2020draft/kr2020dfipr.pdf)


It is developed to be easily integrated with [OpenAI Gym](https://gym.openai.com/) environments.


Experiments are made for the following environment and with the following Monitoring Rewards:
 - Cliff Walking 
    * (Not 'cliff' Until 'goal', reward=0, cost=-1, success=1, failure=-100)
    * (Always Eventually 'safe_path', reward=0, cost=-1, success=0, failure=0)
 - Taxi Domain
    * (Eventually('correct_pickup' And 'goal'), reward=0, cost=-1, success=20, failure=0
    * (Eventually(Always(Not 'illegal_action'), reward=0, cost=-9, success=0, failure=0
 - Cartpole 
 - Breakout 

## Install 
- [flloat](https://github.com/whitemech/flloat) for LTLf formulas and DFA
- [OpenAI Gym](https://gym.openai.com/) for Taxi, Cartpole and Breakout environments
- [OpenAI Baselines](https://github.com/openai/baselines) for Cartpole and Breakout


See also https://github.com/fabriziocacicia/monitoring_rewards for a guide into monitoring rewards
