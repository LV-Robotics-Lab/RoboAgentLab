MotorMind mode.

The session's model is `motormind/<base>`: a provider that plays the MotorMind mission in code. It plans the task into clauses and subgoals, runs each subgoal through the executor one action at a time, judges every attempt (code measurements first, the verifier otherwise), and retries, backs up to the grasp or replans by fixed rules. Every pi turn is exactly one robot call: `mm_observe` (look), `mm_act` (one action in the robot's base frame: robot0_base, mm and deg; forward +X, left +Y, up +Z) or `finish` when the mission ends.

This text describes the mode; the provider does not read it. The configuration evidence of a run is the hash of the role prompts (`robot_result.motormind.prompts`), not this file.
