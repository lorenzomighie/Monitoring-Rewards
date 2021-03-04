# Monitoring Rewards via Transducers

Project for Elective in AI - Reasoning Agents

An implementation and application of Monitoring Rewards based on the paper
[Temporal Logic Monitoring Rewards via Transducers](http://www.diag.uniroma1.it/degiacom/papers/2020draft/kr2020dfipr.pdf) to specify Non-Markovian Rewards in a Reinforcement Learning setup using Linear Time Logic over finite traces (LTLf) and constructing a Monitor Reward capable of finding the state for which the formula given is temporary true, temporary false, permanently true or permanently false (corresponding to assigning reward, cost, success or failure), based on a trace containing the 'state' as a truth assignment to propositional atoms.


It is developed to be easily integrated with [OpenAI Gym](https://gym.openai.com/) environments.


Experiments are made for the following environment and with the following Monitoring Rewards:
 - Cliff Walking 
    * (Not 'cliff' Until 'goal', reward=0, cost=-1, success=1, failure=-100)
    * (Always Eventually 'safe_path', reward=0, cost=-1, success=0, failure=0)
 - Taxi Domain
    * (Eventually('correct_pickup' And 'goal'), reward=0, cost=-1, success=20, failure=0)
    * (Always(Eventually(Not 'illegal_action'), reward=0, cost=-9, success=0, failure=0)

## Install 
- [flloat](https://github.com/whitemech/flloat) for LTLf formulas and DFA
- [OpenAI Gym](https://gym.openai.com/) 


See also https://github.com/fabriziocacicia/monitoring_rewards for a guide into monitoring rewards and https://github.com/whitemech/LTLf2DFA for a guida into Linear Time Logic and Deterministic Finite state Automaton.
